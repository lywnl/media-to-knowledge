from __future__ import annotations

import ast
from collections import deque
from pathlib import Path

_PROJECT_CONTRACT_FILES = (Path("pyproject.toml"), Path("uv.lock"))
_BACKEND_SUFFIXES = frozenset({".py", ".json"})
_BACKEND_ROOT = Path("src/video_demo")
_EXCLUDED_BACKEND_ROOTS = (Path("src/video_demo/web"),)


def prediction_implementation_files(workspace_root: Path) -> tuple[Path, ...]:
    """返回完整产品预测使用的后端源码、静态契约和依赖锁。"""

    root = workspace_root.expanduser().resolve(strict=False)
    source_root = root / _BACKEND_ROOT
    if (
        not source_root.is_dir()
        or (root / "src").is_symlink()
        or source_root.is_symlink()
    ):
        raise ValueError("预测实现源码目录缺失或不安全")
    for relative in _PROJECT_CONTRACT_FILES:
        contract = root / relative
        if not contract.is_file() or contract.is_symlink():
            raise ValueError(f"预测实现契约缺失或不安全：{relative.as_posix()}")
    files = list(_PROJECT_CONTRACT_FILES)
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("预测实现源码不得包含符号链接")
        if not path.is_file() or path.suffix not in _BACKEND_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(
            relative.is_relative_to(excluded) for excluded in _EXCLUDED_BACKEND_ROOTS
        ):
            continue
        if "__pycache__" in relative.parts:
            continue
        files.append(relative)
    return tuple(sorted(set(files), key=lambda item: item.as_posix()))


def implementation_import_closure(
    reference_root: Path,
    entry_files: tuple[Path, ...],
    *,
    extra_files: tuple[Path, ...] = _PROJECT_CONTRACT_FILES,
    excluded_files: frozenset[Path] = frozenset(),
    leaf_files: frozenset[Path] = frozenset(),
) -> tuple[Path, ...]:
    """从入口源码静态追踪工作区内 ``video_demo`` 导入闭包。"""

    root = reference_root.expanduser().resolve(strict=False)
    pending = deque(entry_files)
    discovered = set(extra_files)
    while pending:
        relative = pending.popleft()
        if relative in discovered or relative in excluded_files:
            continue
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"实现入口或依赖文件缺失：{relative.as_posix()}")
        discovered.add(relative)
        if relative.suffix != ".py" or relative in leaf_files:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as error:
            raise ValueError(f"实现源码无法解析：{relative.as_posix()}") from error
        for module_name in _imported_video_demo_modules(tree, relative):
            for imported in _module_files(root, module_name):
                if imported not in discovered and imported not in excluded_files:
                    pending.append(imported)
    return tuple(sorted(discovered, key=lambda item: item.as_posix()))


def _imported_video_demo_modules(
    tree: ast.AST,
    relative: Path,
) -> tuple[str, ...]:
    current_module = _module_name(relative)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.name
                for alias in node.names
                if alias.name == "video_demo" or alias.name.startswith("video_demo.")
            )
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        imported_from = _absolute_import_from(current_module, relative, node)
        if imported_from is None or not (
            imported_from == "video_demo" or imported_from.startswith("video_demo.")
        ):
            continue
        modules.add(imported_from)
        for alias in node.names:
            if alias.name != "*":
                modules.add(f"{imported_from}.{alias.name}")
    return tuple(sorted(modules))


def _absolute_import_from(
    current_module: str,
    relative: Path,
    node: ast.ImportFrom,
) -> str | None:
    if node.level == 0:
        return node.module
    package_parts = current_module.split(".")
    if relative.name != "__init__.py":
        package_parts.pop()
    remove = node.level - 1
    if remove > len(package_parts):
        return None
    base = package_parts[: len(package_parts) - remove]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _module_name(relative: Path) -> str:
    module_path = relative.with_suffix("")
    parts = list(module_path.parts)
    if not parts or parts[0] != "src":
        raise ValueError("实现源码必须位于 src 目录")
    parts = parts[1:]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_files(root: Path, module_name: str) -> tuple[Path, ...]:
    if not (module_name == "video_demo" or module_name.startswith("video_demo.")):
        return ()
    module_parts = module_name.split(".")
    resolved: list[Path] = []
    for length in range(1, len(module_parts) + 1):
        partial = Path("src", *module_parts[:length])
        module_file = partial.with_suffix(".py")
        package_file = partial / "__init__.py"
        if (root / module_file).is_file():
            resolved.append(module_file)
        elif (root / package_file).is_file():
            resolved.append(package_file)
        else:
            break
    return tuple(resolved)
