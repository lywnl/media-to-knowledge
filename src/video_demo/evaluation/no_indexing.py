from __future__ import annotations

import ast
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

PROHIBITED_PACKAGE_PARTS = (
    "chromadb",
    "elasticsearch",
    "faiss",
    "langchain",
    "llama_index",
    "opensearch",
    "pgvector",
    "pinecone",
    "pymilvus",
    "qdrant",
    "sentence_transformers",
    "weaviate",
)
PROHIBITED_SYMBOL_PARTS = (
    "bm25",
    "chroma",
    "embedding",
    "faiss",
    "milvus",
    "opensearch",
    "pgvector",
    "pinecone",
    "qdrant",
    "sentence_transform",
    "vector_index",
    "weaviate",
)
PROHIBITED_STAGE_NAMES = {"EMBEDDING_BUILD", "BM25_BUILD", "VECTOR_INDEX_BUILD"}
PROHIBITED_VECTOR_HOST_PARTS = (
    "chroma",
    "milvus",
    "opensearch",
    "pinecone",
    "qdrant",
    "weaviate",
)
PROHIBITED_VECTOR_PORTS = {6333, 6334, 19530}
INDEX_OPERATION_WORDS = {"add_documents", "add_texts", "index", "upsert"}
INDEX_CONTEXT_WORDS = {"collection", "collections", "index", "indexes", "points", "vectors"}
COMPOUND_INDEX_OPERATIONS = {
    "create_collection",
    "create_index",
    "upsert_points",
    "upsert_vectors",
}
INDEX_OPERATION_LITERALS = (
    "add_documents",
    "add_texts",
    "create_collection",
    "create_index",
    "upsert",
    "upsert_points",
    "upsert_vectors",
)
INDEX_VECTOR_WRITE_PATH_PARTS = {"points", "upsert", "vectors"}
CONFIG_SUFFIXES = {
    ".bash",
    ".conf",
    ".env",
    ".ini",
    ".json",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
    ".zsh",
}
URL_PATTERN = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)

_ConstantScalar = str | int | float | bool
_ConstantFormatValue = (
    _ConstantScalar
    | tuple[_ConstantScalar, ...]
    | dict[str, _ConstantScalar]
)
@dataclass(frozen=True, slots=True)
class IndexingViolation:
    rule: str
    relative_path: str
    line: int
    detail: str


def audit_no_indexing_capability(project_root: Path) -> tuple[IndexingViolation, ...]:
    """审计生产源码、依赖和阶段常量，防止 Demo 越界实现检索索引。"""

    root = project_root.expanduser().resolve(strict=True)
    source_root = root / "src" / "video_demo"
    violations: list[IndexingViolation] = []
    if source_root.is_dir():
        for path in sorted(source_root.rglob("*.py")):
            violations.extend(_audit_python(path, root))
        for path in sorted(source_root.rglob("*")):
            if path.is_file() and path.suffix.casefold() in CONFIG_SUFFIXES:
                violations.extend(_audit_configuration(path, root))
    violations.extend(_audit_project_configuration(root))
    violations.extend(_audit_scripts(root))
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        violations.extend(_audit_dependencies(pyproject, root))
    uv_lock = root / "uv.lock"
    if uv_lock.is_file():
        violations.extend(_audit_resolved_dependencies(uv_lock, root))
    return tuple(
        sorted(
            set(violations),
            key=lambda item: (item.relative_path, item.line, item.rule, item.detail),
        ),
    )


def _audit_python(path: Path, project_root: Path) -> list[IndexingViolation]:
    relative_path = path.relative_to(project_root).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[IndexingViolation] = []
    policy_nodes = (
        _policy_definition_nodes(tree)
        if path.resolve() == Path(__file__).resolve()
        else set()
    )
    dynamic_import_functions = _dynamic_import_function_names(tree)
    for node in ast.walk(tree):
        if id(node) in policy_nodes:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                violations.extend(_package_violation(alias.name, node, relative_path))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            violations.extend(_package_violation(node.module, node, relative_path))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_symbol_violation(node.name, node, relative_path))
        elif isinstance(node, ast.Attribute):
            violations.extend(_symbol_violation(_dotted_name(node), node, relative_path))
            violations.extend(_index_reference_violation(node, relative_path))
        elif isinstance(node, ast.Name):
            violations.extend(_symbol_violation(node.id, node, relative_path))
        elif isinstance(node, ast.Call):
            violations.extend(
                _dynamic_import_violation(
                    node,
                    relative_path,
                    dynamic_import_functions,
                ),
            )
            violations.extend(_index_operation_violation(node, relative_path))
            value = _constant_string(node)
            if value is not None:
                violations.extend(_string_violations(value, node, relative_path))
        elif isinstance(node, (ast.Constant, ast.BinOp, ast.JoinedStr)):
            value = _constant_string(node)
            if value is not None:
                violations.extend(_string_violations(value, node, relative_path))
    return violations


def _audit_dependencies(path: Path, project_root: Path) -> list[IndexingViolation]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    project = payload.get("project", {})
    dependencies = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    for values in optional.values():
        dependencies.extend(values)
    violations: list[IndexingViolation] = []
    for dependency in dependencies:
        normalized = str(dependency).casefold().replace("-", "_")
        prohibited = _matching_part(normalized, PROHIBITED_PACKAGE_PARTS)
        if prohibited is not None:
            violations.append(
                IndexingViolation(
                    rule="prohibited-dependency",
                    relative_path=path.relative_to(project_root).as_posix(),
                    line=1,
                    detail=prohibited,
                ),
            )
    return violations


def _audit_resolved_dependencies(
    path: Path,
    project_root: Path,
) -> list[IndexingViolation]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    packages = payload.get("package", [])
    violations: list[IndexingViolation] = []
    if not isinstance(packages, list):
        return violations
    for package in packages:
        if not isinstance(package, dict):
            continue
        normalized = str(package.get("name", "")).casefold().replace("-", "_")
        prohibited = _matching_part(normalized, PROHIBITED_PACKAGE_PARTS)
        if prohibited is not None:
            violations.append(
                IndexingViolation(
                    rule="prohibited-dependency",
                    relative_path=path.relative_to(project_root).as_posix(),
                    line=1,
                    detail=prohibited,
                ),
            )
    return violations


def _audit_configuration(path: Path, project_root: Path) -> list[IndexingViolation]:
    relative_path = path.relative_to(project_root).as_posix()
    text = path.read_text(encoding="utf-8")
    normalized = text.casefold().replace("-", "_")
    violations: list[IndexingViolation] = []
    prohibited = _matching_part(normalized, PROHIBITED_SYMBOL_PARTS)
    package = _matching_part(normalized, PROHIBITED_PACKAGE_PARTS)
    operation = _matching_part(normalized, INDEX_OPERATION_LITERALS)
    for detail in {value for value in (prohibited, package) if value is not None}:
        violations.append(
            IndexingViolation("prohibited-config-text", relative_path, 1, detail),
        )
    if operation is not None:
        violations.append(
            IndexingViolation("prohibited-index-operation", relative_path, 1, operation),
        )
    for match in URL_PATTERN.finditer(text):
        if _is_vector_url(match.group(0)):
            violations.append(
                IndexingViolation("prohibited-vector-url", relative_path, 1, match.group(0)),
            )
    return violations


def _audit_project_configuration(project_root: Path) -> list[IndexingViolation]:
    candidates = [
        path
        for path in project_root.iterdir()
        if path.is_file() and _is_configuration_path(path)
    ]
    violations: list[IndexingViolation] = []
    for path in sorted(set(candidates)):
        violations.extend(_audit_configuration(path, project_root))
    return violations


def _audit_scripts(project_root: Path) -> list[IndexingViolation]:
    scripts_root = project_root / "scripts"
    if not scripts_root.is_dir():
        return []
    violations: list[IndexingViolation] = []
    for path in sorted(item for item in scripts_root.rglob("*") if item.is_file()):
        if path.suffix.casefold() == ".py":
            violations.extend(_audit_python(path, project_root))
        else:
            violations.extend(_audit_configuration(path, project_root))
    return violations


def _is_configuration_path(path: Path) -> bool:
    return path.name.startswith(".env") or path.suffix.casefold() in CONFIG_SUFFIXES


def _package_violation(
    package: str,
    node: ast.AST,
    relative_path: str,
) -> list[IndexingViolation]:
    normalized = package.casefold().replace("-", "_")
    if normalized in {"operator", "_operator"}:
        return [
            _violation(
                "prohibited-index-operation",
                relative_path,
                node,
                "operator-dynamic-attribute-factory",
            ),
        ]
    prohibited = _matching_part(normalized, PROHIBITED_PACKAGE_PARTS)
    if prohibited is None:
        return []
    return [_violation("prohibited-import", relative_path, node, prohibited)]


def _symbol_violation(
    symbol: str | None,
    node: ast.AST,
    relative_path: str,
) -> list[IndexingViolation]:
    if symbol is None:
        return []
    normalized = symbol.casefold().replace("-", "_")
    operation = _matching_part(normalized, INDEX_OPERATION_LITERALS)
    if operation is not None:
        return [_violation("prohibited-index-operation", relative_path, node, operation)]
    prohibited = _matching_part(normalized, PROHIBITED_SYMBOL_PARTS)
    if prohibited is None:
        return []
    return [_violation("prohibited-symbol", relative_path, node, prohibited)]


def _dynamic_import_violation(
    node: ast.Call,
    relative_path: str,
    function_names: set[str],
) -> list[IndexingViolation]:
    function_name = _dotted_name(node.func)
    if function_name not in function_names:
        return []
    package = _dynamic_import_package(node)
    if package is None:
        return []
    normalized = package.casefold().replace("-", "_")
    if normalized in {"operator", "_operator"}:
        return [
            _violation(
                "prohibited-index-operation",
                relative_path,
                node,
                "operator-dynamic-attribute-factory",
            ),
        ]
    prohibited = _matching_part(normalized, PROHIBITED_PACKAGE_PARTS)
    if prohibited is None:
        return []
    return [_violation("prohibited-dynamic-import", relative_path, node, prohibited)]


def _dynamic_import_function_names(tree: ast.Module) -> set[str]:
    function_names = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                if alias.name == "importlib" or alias.name.startswith("importlib."):
                    function_names.add(f"{imported_name}.import_module")
                elif alias.name == "builtins":
                    function_names.add(f"{imported_name}.__import__")
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == "importlib"
        ):
            for alias in node.names:
                if alias.name == "import_module":
                    function_names.add(alias.asname or alias.name)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == "builtins"
        ):
            for alias in node.names:
                if alias.name == "__import__":
                    function_names.add(alias.asname or alias.name)
    return function_names


def _dynamic_import_package(node: ast.Call) -> str | None:
    positional_package, has_positional_argument = _possible_first_positional_string(
        node.args,
    )
    keyword_states: list[tuple[set[str], bool]] = []
    for keyword in node.keywords:
        if keyword.arg is None:
            keyword_states.append(_literal_mapping_string_state(keyword.value, "name"))
        elif keyword.arg == "name":
            value = _constant_string(keyword.value)
            keyword_states.append(({value} if value is not None else set(), True))

    guaranteed_states = [state for state in keyword_states if state[1]]
    if len(guaranteed_states) > 1 or (
        has_positional_argument and guaranteed_states
    ):
        return None
    if has_positional_argument:
        return positional_package
    candidates = (
        guaranteed_states[0][0]
        if guaranteed_states
        else set().union(*(state[0] for state in keyword_states))
    )
    return next(iter(candidates)) if len(candidates) == 1 else None


def _possible_first_positional_string(
    arguments: list[ast.expr],
) -> tuple[str | None, bool]:
    for argument in arguments:
        if not isinstance(argument, ast.Starred):
            return _constant_string(argument), True
        expanded = argument.value
        if not isinstance(expanded, (ast.List, ast.Tuple)):
            continue
        candidate, is_nonempty = _possible_first_positional_string(expanded.elts)
        if is_nonempty:
            return candidate, True
    return None, False


def _literal_mapping_string_state(
    node: ast.AST,
    target_key: str,
) -> tuple[set[str], bool]:
    if not isinstance(node, ast.Dict):
        return set(), False
    candidates: set[str] = set()
    guaranteed = False
    for key_node, value_node in zip(node.keys, node.values, strict=True):
        if key_node is None:
            nested_candidates, nested_guaranteed = _literal_mapping_string_state(
                value_node,
                target_key,
            )
            if nested_guaranteed:
                candidates = nested_candidates
                guaranteed = True
            else:
                candidates.update(nested_candidates)
            continue
        key = _constant_string(key_node)
        if key != target_key:
            continue
        value = _constant_string(value_node)
        candidates = {value} if value is not None else set()
        guaranteed = True
    return candidates, guaranteed


def _index_operation_violation(
    node: ast.Call,
    relative_path: str,
) -> list[IndexingViolation]:
    dynamic_getattr = _dynamic_getattr_violation(node, relative_path)
    if dynamic_getattr:
        return dynamic_getattr
    tokens = _call_tokens(node)
    compound_operation = tokens & COMPOUND_INDEX_OPERATIONS
    if compound_operation:
        return [
            _violation(
                "prohibited-index-operation",
                relative_path,
                node,
                ".".join(sorted(compound_operation)),
            ),
        ]
    operation = tokens & INDEX_OPERATION_WORDS
    context = tokens & INDEX_CONTEXT_WORDS
    if not operation or not context or operation == {"index"} == context:
        return []
    detail = ".".join(sorted(tokens))
    return [_violation("prohibited-index-operation", relative_path, node, detail)]


def _dynamic_getattr_violation(
    node: ast.Call,
    relative_path: str,
) -> list[IndexingViolation]:
    function = node.func
    if not isinstance(function, ast.Call) or _dotted_name(function.func) != "getattr":
        return []
    if len(function.args) < 2:
        return [_violation("prohibited-index-operation", relative_path, node, "dynamic-getattr")]
    method = _constant_string(function.args[1])
    if method is None:
        detail = "dynamic-getattr"
    else:
        matched_operation = _matching_part(
            method.casefold().replace("-", "_"),
            INDEX_OPERATION_LITERALS,
        )
        if matched_operation is None:
            return []
        detail = matched_operation
    return [_violation("prohibited-index-operation", relative_path, node, detail)]


def _index_reference_violation(
    node: ast.Attribute,
    relative_path: str,
) -> list[IndexingViolation]:
    dotted = _dotted_name(node)
    if dotted is None:
        return []
    tokens = {part.casefold() for part in dotted.split(".")}
    operation = tokens & INDEX_OPERATION_WORDS
    context = tokens & INDEX_CONTEXT_WORDS
    if not operation or not context or operation == {"index"} == context:
        return []
    return [
        _violation(
            "prohibited-index-operation",
            relative_path,
            node,
            ".".join(sorted(tokens)),
        ),
    ]


def _string_violations(
    value: str,
    node: ast.AST,
    relative_path: str,
) -> list[IndexingViolation]:
    normalized = value.casefold().replace("-", "_")
    violations: list[IndexingViolation] = []
    if value in PROHIBITED_STAGE_NAMES:
        violations.append(_violation("prohibited-stage", relative_path, node, value))
    prohibited = _matching_part(normalized, PROHIBITED_SYMBOL_PARTS)
    if prohibited is not None:
        violations.append(_violation("prohibited-text", relative_path, node, prohibited))
    operation = _matching_part(normalized, INDEX_OPERATION_LITERALS)
    if operation is not None:
        violations.append(
            _violation("prohibited-index-operation", relative_path, node, operation),
        )
    if value.startswith(("http://", "https://")) and _is_vector_url(value):
        violations.append(
            _violation("prohibited-vector-url", relative_path, node, value),
        )
    return violations


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _call_tokens(node: ast.Call) -> set[str]:
    tokens: set[str] = set()

    def visit(current: ast.AST) -> None:
        if isinstance(current, ast.Attribute):
            tokens.add(current.attr.casefold())
            visit(current.value)
        elif isinstance(current, ast.Name):
            tokens.add(current.id.casefold())
        elif isinstance(current, ast.Call):
            if _dotted_name(current.func) == "getattr" and len(current.args) >= 2:
                visit(current.args[0])
                attribute = _constant_string(current.args[1])
                if attribute is not None:
                    tokens.add(attribute.casefold())
            visit(current.func)
            for argument in current.args:
                visit(argument)

    visit(node)
    return tokens


def _policy_definition_nodes(tree: ast.Module) -> set[int]:
    node_ids: set[int] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if not node.targets[0].id.startswith(("PROHIBITED_", "INDEX_", "COMPOUND_")):
            continue
        if not _is_literal_policy_value(node.value):
            continue
        node_ids.update(id(item) for item in ast.walk(node))
    return node_ids


def _is_literal_policy_value(node: ast.AST) -> bool:
    return all(
        isinstance(item, (ast.Constant, ast.Load, ast.Set, ast.Tuple))
        for item in ast.walk(node)
    )


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_string(node.left)
        right = _constant_string(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                formatted = _constant_formatted_value(value)
                if formatted is None:
                    return None
                parts.append(formatted)
                continue
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                return None
            parts.append(value.value)
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        template = _constant_string(node.left)
        values = _constant_format_values(node.right)
        if template is None or values is None:
            return None
        try:
            formatted = template % values
        except (TypeError, ValueError):
            return None
        return formatted if isinstance(formatted, str) else None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and len(node.args) == 1
    ):
        separator = _constant_string(node.func.value)
        joined_values = node.args[0]
        if separator is None or not isinstance(joined_values, (ast.List, ast.Tuple)):
            return None
        resolved_parts: list[str] = []
        for value in joined_values.elts:
            part = _constant_string(value)
            if part is None:
                return None
            resolved_parts.append(part)
        return separator.join(resolved_parts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        template = _constant_string(node.func.value)
        arguments = [_constant_scalar(value) for value in node.args]
        if template is None or any(value is None for value in arguments):
            return None
        keywords = _constant_format_keywords(node.keywords)
        if keywords is None:
            return None
        try:
            return template.format(*arguments, **keywords)
        except (IndexError, KeyError, TypeError, ValueError):
            return None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "decode"
        and isinstance(node.func.value, ast.Constant)
        and isinstance(node.func.value.value, bytes)
    ):
        decode_arguments = _constant_decode_arguments(node)
        if decode_arguments is None:
            return None
        encoding, errors = decode_arguments
        try:
            return node.func.value.value.decode(encoding=encoding, errors=errors)
        except (LookupError, UnicodeDecodeError):
            return None
    return None


def _constant_formatted_value(node: ast.FormattedValue) -> str | None:
    value = _constant_scalar(node.value)
    if value is None:
        return None
    if node.conversion == -1:
        converted: str | int | float | bool = value
    elif node.conversion == ord("s"):
        converted = str(value)
    elif node.conversion == ord("r"):
        converted = repr(value)
    elif node.conversion == ord("a"):
        converted = ascii(value)
    else:
        return None
    format_spec = ""
    if node.format_spec is not None:
        resolved_spec = _constant_string(node.format_spec)
        if resolved_spec is None:
            return None
        format_spec = resolved_spec
    try:
        return format(converted, format_spec)
    except (TypeError, ValueError):
        return None


def _constant_decode_arguments(node: ast.Call) -> tuple[str, str] | None:
    if len(node.args) > 2 or any(keyword.arg is None for keyword in node.keywords):
        return None
    names = ("encoding", "errors")
    values: dict[str, str] = {}
    for name, argument in zip(names, node.args, strict=False):
        resolved = _constant_string(argument)
        if resolved is None:
            return None
        values[name] = resolved
    for keyword in node.keywords:
        if keyword.arg not in names or keyword.arg in values:
            return None
        resolved = _constant_string(keyword.value)
        if resolved is None:
            return None
        values[keyword.arg] = resolved
    return values.get("encoding", "utf-8"), values.get("errors", "strict")


def _constant_scalar(node: ast.AST) -> _ConstantScalar | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
        return node.value
    return _constant_string(node)


def _constant_format_values(
    node: ast.AST,
) -> _ConstantFormatValue | None:
    if isinstance(node, ast.Tuple):
        values = tuple(_constant_scalar(item) for item in node.elts)
        if any(value is None for value in values):
            return None
        return tuple(value for value in values if value is not None)
    if isinstance(node, ast.Dict):
        return _constant_literal_mapping(node)
    return _constant_scalar(node)


def _constant_format_keywords(
    keywords: list[ast.keyword],
) -> dict[str, _ConstantScalar] | None:
    resolved: dict[str, _ConstantScalar] = {}
    for keyword in keywords:
        if keyword.arg is None:
            expanded = _constant_literal_mapping(keyword.value)
            if expanded is None or resolved.keys() & expanded.keys():
                return None
            resolved.update(expanded)
            continue
        if keyword.arg in resolved:
            return None
        value = _constant_scalar(keyword.value)
        if value is None:
            return None
        resolved[keyword.arg] = value
    return resolved


def _constant_literal_mapping(node: ast.AST) -> dict[str, _ConstantScalar] | None:
    if not isinstance(node, ast.Dict):
        return None
    resolved: dict[str, _ConstantScalar] = {}
    for key_node, value_node in zip(node.keys, node.values, strict=True):
        if key_node is None:
            expanded = _constant_literal_mapping(value_node)
            if expanded is None:
                return None
            resolved.update(expanded)
            continue
        key = _constant_string(key_node)
        value = _constant_scalar(value_node)
        if key is None or value is None:
            return None
        resolved[key] = value
    return resolved


def _matching_part(value: str, candidates: tuple[str, ...]) -> str | None:
    return next((part for part in candidates if part in value), None)


def _is_vector_url(value: str) -> bool:
    parsed = urlsplit(value)
    host = (parsed.hostname or "").casefold()
    if _matching_part(host, PROHIBITED_VECTOR_HOST_PARTS) is not None:
        return True
    try:
        port = parsed.port
    except ValueError:
        return False
    path = parsed.path.casefold()
    path_parts = {part for part in path.split("/") if part}
    high_signal_path = bool(path_parts & {"collections", "indexes"}) and bool(
        path_parts & INDEX_VECTOR_WRITE_PATH_PARTS,
    )
    return high_signal_path or (
        port in PROHIBITED_VECTOR_PORTS
        and bool(path_parts & {"collections", "indexes", "points", "vectors"})
    )


def _violation(
    rule: str,
    relative_path: str,
    node: ast.AST,
    detail: str,
) -> IndexingViolation:
    return IndexingViolation(
        rule=rule,
        relative_path=relative_path,
        line=getattr(node, "lineno", 1),
        detail=detail,
    )
