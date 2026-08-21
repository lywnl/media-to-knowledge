from __future__ import annotations

from pathlib import Path

import pytest

import video_demo.evaluation.no_indexing as no_indexing_module
from video_demo.evaluation.no_indexing import audit_no_indexing_capability


def test_production_source_and_dependencies_contain_no_indexing_capability() -> None:
    project_root = Path(__file__).parents[2]

    assert audit_no_indexing_capability(project_root) == ()


@pytest.mark.parametrize(
    ("source", "expected_rule"),
    [
        ("import qdrant_client\n", "prohibited-import"),
        ("from weaviate import Client\n", "prohibited-import"),
        (
            "import importlib\nbackend = importlib.import_module('pinecone')\n",
            "prohibited-dynamic-import",
        ),
        (
            "import importlib\nbackend = importlib.import_module('qdr' + 'ant_client')\n",
            "prohibited-dynamic-import",
        ),
        (
            "import importlib\nbackend = importlib.import_module('oper' + 'ator')\n",
            "prohibited-index-operation",
        ),
        (
            "import importlib as il\nbackend = il.import_module('operator')\n",
            "prohibited-index-operation",
        ),
        (
            "import importlib.util\nbackend = importlib.import_module('operator')\n",
            "prohibited-index-operation",
        ),
        (
            "from importlib import import_module\n"
            "backend = import_module('operator')\n",
            "prohibited-index-operation",
        ),
        ("backend = __import__('opensearchpy')\n", "prohibited-dynamic-import"),
        ("backend = __import__('operator')\n", "prohibited-index-operation"),
        ("backend = __import__(name='operator')\n", "prohibited-index-operation"),
        ("backend = __import__(*('operator',))\n", "prohibited-index-operation"),
        (
            "backend = __import__(**{'name': 'operator'})\n",
            "prohibited-index-operation",
        ),
        (
            "backend = __import__(**{}, **{'name': 'operator'})\n",
            "prohibited-index-operation",
        ),
        (
            "backend = __import__(**{'level': 0}, **{'name': 'operator'})\n",
            "prohibited-index-operation",
        ),
        (
            "backend = __import__(*(), **{}, name='operator')\n",
            "prohibited-index-operation",
        ),
        (
            "args = ()\nbackend = __import__(*args, name='operator')\n",
            "prohibited-index-operation",
        ),
        (
            "kwargs = {}\nbackend = __import__(**kwargs, name='operator')\n",
            "prohibited-index-operation",
        ),
        (
            "kwargs = {}\nbackend = __import__(name='operator', **kwargs)\n",
            "prohibited-index-operation",
        ),
        (
            "args = ()\nbackend = __import__(*args, 'operator')\n",
            "prohibited-index-operation",
        ),
        (
            "globals_value = {}\n"
            "backend = __import__(*('operator', globals_value))\n",
            "prohibited-index-operation",
        ),
        (
            "globals_value = {}\n"
            "backend = __import__("
            "**{'name': 'operator', 'globals': globals_value}"
            ")\n",
            "prohibited-index-operation",
        ),
        (
            "kwargs = {}\n"
            "backend = __import__(**{'name': 'operator', **kwargs})\n",
            "prohibited-index-operation",
        ),
        (
            "import builtins\nbackend = builtins.__import__('operator')\n",
            "prohibited-index-operation",
        ),
        (
            "import importlib\n"
            "backend = importlib.import_module(**{'name': 'operator'})\n",
            "prohibited-index-operation",
        ),
        (
            "import importlib\n"
            "backend = importlib.import_module(**{}, **{'name': 'operator'})\n",
            "prohibited-index-operation",
        ),
        (
            "import importlib\n"
            "backend = importlib.import_module("
            "**{'package': 'ignored'}, **{'name': 'operator'}"
            ")\n",
            "prohibited-index-operation",
        ),
        (
            "import importlib\n"
            "args = ()\n"
            "backend = importlib.import_module(*args, name='operator')\n",
            "prohibited-index-operation",
        ),
        (
            "import importlib\n"
            "kwargs = {}\n"
            "backend = importlib.import_module(name='operator', **kwargs)\n",
            "prohibited-index-operation",
        ),
        (
            "from _operator import attrgetter\nbackend = attrgetter(method)\n",
            "prohibited-index-operation",
        ),
        ("result = client.embeddings.create(input=['正文'])\n", "prohibited-symbol"),
        ("client.index.upsert(records)\n", "prohibited-index-operation"),
        ("client.collection('docs').upsert(points)\n", "prohibited-index-operation"),
        (
            "getattr(client.collection('docs'), 'up' + 'sert')(points)\n",
            "prohibited-index-operation",
        ),
        (
            "ops = {'write': client.index.upsert}\nops['write'](records)\n",
            "prohibited-index-operation",
        ),
        (
            "method = ''.join(['up', 'sert'])\n"
            "getattr(client.collection('docs'), method)(points)\n",
            "prohibited-index-operation",
        ),
        (
            "def write():\n"
            "    method = ''.join(['up', 'sert'])\n"
            "    getattr(client.collection('docs'), method)(points)\n",
            "prohibited-index-operation",
        ),
        (
            "method = ''.join(['up', 'sert'])\n"
            "getattr(client.collection('docs'), method)(points)\n"
            "method = 'fetch'\n",
            "prohibited-index-operation",
        ),
        (
            "method = ''.join(['up', 'sert'])\n"
            "if use_fetch:\n"
            "    method = 'fetch'\n"
            "getattr(client.collection('docs'), method)(points)\n",
            "prohibited-index-operation",
        ),
        (
            "method = ''.join(['up', 'sert'])\n"
            "for item in records:\n"
            "    method = 'fetch'\n"
            "getattr(client.collection('docs'), method)(points)\n",
            "prohibited-index-operation",
        ),
        (
            "method = ''.join(['up', 'sert'])\n"
            "try:\n"
            "    method = resolve_method()\n"
            "except LookupError:\n"
            "    pass\n"
            "getattr(client.collection('docs'), method)(points)\n",
            "prohibited-index-operation",
        ),
        (
            "with context():\n"
            "    method = ''.join(['up', 'sert'])\n"
            "    getattr(client.collection('docs'), method)(points)\n",
            "prohibited-index-operation",
        ),
        (
            "try:\n"
            "    method = ''.join(['up', 'sert'])\n"
            "    may_fail()\n"
            "    method = 'fetch'\n"
            "except RuntimeError:\n"
            "    pass\n"
            "getattr(client.collection('docs'), method)(points)\n",
            "prohibited-index-operation",
        ),
        (
            "source = ''.join(['up', 'sert'])\n"
            "method = 'fetch'\n"
            "match source:\n"
            "    case method:\n"
            "        getattr(client.collection('docs'), method)(points)\n",
            "prohibited-index-operation",
        ),
        (
            "def write(method=''.join(['up', 'sert'])):\n"
            "    getattr(client.collection('docs'), method)(points)\n",
            "prohibited-index-operation",
        ),
        (
            "method = 'fetch'\n"
            "def write():\n"
            "    getattr(client.collection('docs'), method)(points)\n"
            "method = ''.join(['up', 'sert'])\n"
            "write()\n",
            "prohibited-index-operation",
        ),
        (
            "method = ''.join(['up', 'sert'])\n"
            "class Writer:\n"
            "    method = 'fetch'\n"
            "    def write(self):\n"
            "        getattr(client.collection('docs'), method)(points)\n",
            "prohibited-index-operation",
        ),
        (
            "getattr(client, f\"{'up'}sert\")(\n"
            "    collection_name='docs', points=records\n"
            ")\n",
            "prohibited-index-operation",
        ),
        (
            "getattr(client, '{}{}'.format('up', 'sert'))(records)\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "method = '{prefix}sert'.format(**{'prefix': 'up'})\n"
            "operator.attrgetter(method)(client)(\n"
            "    collection_name='docs', points=records\n"
            ")\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "method = '{prefix}sert'.format(\n"
            "    **{'prefix': 'fetch', 'prefix': 'up'},\n"
            ")\n"
            "operator.attrgetter(method)(client)(\n"
            "    collection_name='docs', points=records\n"
            ")\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "method = '{prefix}sert'.format(\n"
            "    **{**{'prefix': 'up'}},\n"
            ")\n"
            "operator.attrgetter(method)(client)(\n"
            "    collection_name='docs', points=records\n"
            ")\n",
            "prohibited-index-operation",
        ),
        (
            "getattr(client, '%s%s' % ('up', 'sert'))(records)\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "method = '%(prefix)ssert' % {'prefix': 'up'}\n"
            "operator.attrgetter(method)(client)(\n"
            "    collection_name='docs', points=records\n"
            ")\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "method = '%(prefix)ssert' % {\n"
            "    'prefix': 'fetch', 'prefix': 'up',\n"
            "}\n"
            "operator.attrgetter(method)(client)(\n"
            "    collection_name='docs', points=records\n"
            ")\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "method = '%(prefix)ssert' % {\n"
            "    **{'prefix': 'up'},\n"
            "}\n"
            "operator.attrgetter(method)(client)(\n"
            "    collection_name='docs', points=records\n"
            ")\n",
            "prohibited-index-operation",
        ),
        (
            "getattr(client, b'upsert'.decode('ascii'))(records)\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "method = b'\\xff\\xfeu\\x00p\\x00s\\x00e\\x00r\\x00t\\x00'.decode(\n"
            "    encoding='utf-16',\n"
            ")\n"
            "operator.attrgetter(method)(client)(\n"
            "    collection_name='docs', points=records\n"
            ")\n",
            "prohibited-index-operation",
        ),
        (
            "target = client.collection('docs')\n"
            "method = read_method()\n"
            "getattr(target, method)(records)\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "method = read_method()\n"
            "operator.attrgetter(method)(client)(records)\n",
            "prohibited-index-operation",
        ),
        (
            "from operator import attrgetter as pick\n"
            "method = read_method()\n"
            "pick(method)(client)(\n"
            "    collection_name='docs', points=records\n"
            ")\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "pick = operator.attrgetter\n"
            "again = pick\n"
            "method = read_method()\n"
            "again(method)(client)(records)\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "method = read_method()\n"
            "operator.methodcaller(\n"
            "    method, collection_name='docs', points=records\n"
            ")(client)\n",
            "prohibited-index-operation",
        ),
        (
            "from operator import methodcaller as invoke\n"
            "again = invoke\n"
            "method = read_method()\n"
            "again(method, points=records)(client)\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "pick: object = operator.attrgetter\n"
            "method = read_method()\n"
            "pick(method)(client)(points=records)\n",
            "prohibited-index-operation",
        ),
        (
            "from operator import methodcaller\n"
            "invoke: object = methodcaller\n"
            "method = read_method()\n"
            "invoke(method, points=records)(client)\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "method = read_method()\n"
            "(pick := operator.attrgetter)(method)(client)(points=records)\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "pick = operator.attrgetter\n"
            "writer = lambda method: pick(method)(client)(points=records)\n",
            "prohibited-index-operation",
        ),
        (
            "pick = safe_factory\n"
            "def write():\n"
            "    pick(method)(client)(points=records)\n"
            "import operator\n"
            "pick = operator.attrgetter\n"
            "write()\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "pick = operator.attrgetter\n"
            "class C:\n"
            "    pick = safe_factory\n"
            "    def write(self):\n"
            "        pick(method)(client)(points=records)\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "try:\n"
            "    pick = operator.attrgetter\n"
            "    may_fail()\n"
            "    pick = safe_factory\n"
            "except RuntimeError:\n"
            "    pass\n"
            "pick(method)(client)(points=records)\n",
            "prohibited-index-operation",
        ),
        (
            "import operator\n"
            "match operator.attrgetter:\n"
            "    case pick:\n"
            "        pick(method)(client)(points=records)\n",
            "prohibited-index-operation",
        ),
        ("client.create_collection('docs')\n", "prohibited-index-operation"),
        ("VECTOR_BACKEND = 'pgvector'\n", "prohibited-text"),
        (
            "endpoint = 'https://demo.svc.us-east-1.pinecone.io/vectors/upsert'\n",
            "prohibited-vector-url",
        ),
        (
            "endpoint = 'http://127.0.0.1:6333/collections/demo/points'\n",
            "prohibited-vector-url",
        ),
        (
            "endpoint = 'https://api.example.com/indexes/docs/upsert'\n",
            "prohibited-vector-url",
        ),
        (
            "endpoint = 'https://api.example.com/indexes/' + 'docs/upsert'\n",
            "prohibited-vector-url",
        ),
        ("class MilvusWriter:\n    pass\n", "prohibited-symbol"),
    ],
)
def test_static_audit_detects_known_indexing_escape_routes(
    tmp_path: Path,
    source: str,
    expected_rule: str,
) -> None:
    source_root = tmp_path / "src" / "video_demo"
    source_root.mkdir(parents=True)
    (source_root / "module.py").write_text(source, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n",
        encoding="utf-8",
    )

    violations = audit_no_indexing_capability(tmp_path)

    assert expected_rule in {violation.rule for violation in violations}


def test_static_audit_detects_prohibited_dependency_and_stage(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "video_demo"
    domain_root = source_root / "domain"
    domain_root.mkdir(parents=True)
    (source_root / "__init__.py").write_text("", encoding="utf-8")
    (domain_root / "run.py").write_text(
        "from enum import StrEnum\n"
        "class RunStage(StrEnum):\n"
        "    EMBEDDING_BUILD = 'EMBEDDING_BUILD'\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = ['chromadb>=1']\n",
        encoding="utf-8",
    )

    violations = audit_no_indexing_capability(tmp_path)

    assert {violation.rule for violation in violations} >= {
        "prohibited-dependency",
        "prohibited-stage",
    }


def test_static_audit_detects_prohibited_resolved_dependency_in_uv_lock(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src" / "video_demo"
    source_root.mkdir(parents=True)
    (source_root / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = ['safe-wrapper>=1']\n",
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text(
        "version = 1\n"
        "[[package]]\n"
        "name = 'safe-wrapper'\n"
        "version = '1.0.0'\n"
        "dependencies = [{ name = 'pymilvus' }]\n"
        "[[package]]\n"
        "name = 'pymilvus'\n"
        "version = '2.5.0'\n",
        encoding="utf-8",
    )

    violations = audit_no_indexing_capability(tmp_path)

    assert any(
        violation.rule == "prohibited-dependency"
        and violation.relative_path == "uv.lock"
        and violation.detail == "pymilvus"
        for violation in violations
    )


def test_static_audit_scans_non_python_configuration_files(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "video_demo"
    source_root.mkdir(parents=True)
    (source_root / "settings.json").write_text(
        '{"backend": "qdrant", "url": "http://127.0.0.1:6333/collections/demo"}',
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n",
        encoding="utf-8",
    )

    violations = audit_no_indexing_capability(tmp_path)

    assert {violation.rule for violation in violations} >= {
        "prohibited-config-text",
        "prohibited-vector-url",
    }


def test_static_audit_scans_root_environment_and_scripts(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "video_demo"
    source_root.mkdir(parents=True)
    scripts_root = tmp_path / "scripts"
    scripts_root.mkdir()
    (tmp_path / ".env.example").write_text(
        "VECTOR_URL=http://127.0.0.1:6333/collections/demo/points\n",
        encoding="utf-8",
    )
    (scripts_root / "bootstrap.sh").write_text(
        "curl https://api.example.com/indexes/docs/upsert\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n",
        encoding="utf-8",
    )

    violations = audit_no_indexing_capability(tmp_path)

    assert {violation.relative_path for violation in violations} >= {
        ".env.example",
        "scripts/bootstrap.sh",
    }


def test_static_audit_scans_python_and_extensionless_scripts(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "video_demo"
    source_root.mkdir(parents=True)
    scripts_root = tmp_path / "scripts"
    scripts_root.mkdir()
    (scripts_root / "index_data.py").write_text(
        "client.collection('docs').upsert(points)\n",
        encoding="utf-8",
    )
    (scripts_root / "bootstrap").write_text(
        "curl https://api.example.com/indexes/docs/upsert\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n",
        encoding="utf-8",
    )

    violations = audit_no_indexing_capability(tmp_path)

    assert {violation.relative_path for violation in violations} >= {
        "scripts/index_data.py",
        "scripts/bootstrap",
    }


def test_static_audit_policy_constants_cannot_hide_executable_indexing(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src" / "video_demo" / "evaluation"
    source_root.mkdir(parents=True)
    (source_root / "no_indexing.py").write_text(
        "PROHIBITED_BACKDOOR = client.index.upsert(records)\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n",
        encoding="utf-8",
    )

    violations = audit_no_indexing_capability(tmp_path)

    assert "prohibited-index-operation" in {violation.rule for violation in violations}


def test_policy_literal_on_same_line_cannot_hide_executable_indexing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "src" / "video_demo" / "evaluation"
    source_root.mkdir(parents=True)
    policy = source_root / "no_indexing.py"
    policy.write_text(
        "INDEX_SAFE = ('allowed',); client.collection('docs').upsert(records)\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(no_indexing_module, "__file__", str(policy))

    violations = audit_no_indexing_capability(tmp_path)

    assert "prohibited-index-operation" in {violation.rule for violation in violations}


def test_policy_literal_cannot_hide_executable_chained_assignment_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "src" / "video_demo" / "evaluation"
    source_root.mkdir(parents=True)
    policy = source_root / "no_indexing.py"
    policy.write_text(
        "INDEX_SAFE = sink[\n"
        "    client.collection('docs').upsert(records)\n"
        "] = ('allowed',)\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(no_indexing_module, "__file__", str(policy))

    violations = audit_no_indexing_capability(tmp_path)

    assert "prohibited-index-operation" in {violation.rule for violation in violations}


@pytest.mark.parametrize("format_clause", ("!r", "!a", ":!<4"))
def test_constant_f_string_conversion_uses_runtime_semantics(
    tmp_path: Path,
    format_clause: str,
) -> None:
    source_root = tmp_path / "src" / "video_demo"
    source_root.mkdir(parents=True)
    (source_root / "module.py").write_text(
        f'label = f"{{\'up\'{format_clause}}}sert"\n',
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n",
        encoding="utf-8",
    )

    violations = audit_no_indexing_capability(tmp_path)

    assert violations == ()


@pytest.mark.parametrize(
    "source",
    (
        "def attrgetter(name):\n"
        "    return lambda obj: obj\n"
        "reader = attrgetter(dynamic_name)\n",
    ),
)
def test_static_operator_attribute_factory_with_safe_name_is_allowed(
    tmp_path: Path,
    source: str,
) -> None:
    source_root = tmp_path / "src" / "video_demo"
    source_root.mkdir(parents=True)
    (source_root / "module.py").write_text(source, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n",
        encoding="utf-8",
    )

    violations = audit_no_indexing_capability(tmp_path)

    assert violations == ()


@pytest.mark.parametrize(
    "source",
    (
        "args = ()\nbackend = __import__('json', *args, name='operator')\n",
        "backend = __import__(*('json', 'operator'))\n",
        "kwargs = {}\nbackend = __import__(**{'name': 'operator', 'name': 'json'})\n",
        "kwargs = {}\nbackend = __import__(**{'name': 'operator', **kwargs, 'name': 'json'})\n",
    ),
)
def test_dynamic_import_parser_respects_guaranteed_safe_argument_precedence(
    tmp_path: Path,
    source: str,
) -> None:
    source_root = tmp_path / "src" / "video_demo"
    source_root.mkdir(parents=True)
    (source_root / "module.py").write_text(source, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n",
        encoding="utf-8",
    )

    violations = audit_no_indexing_capability(tmp_path)

    assert violations == ()


@pytest.mark.parametrize(
    "source",
    (
        "import operator\ngetter = operator.attrgetter('title')\n",
        "import operator\ncaller = operator.methodcaller('fetch', limit=1)\n",
        "from operator import attrgetter as pick\npick = safe_factory\n",
    ),
)
def test_operator_dynamic_attribute_factory_capability_is_denied(
    tmp_path: Path,
    source: str,
) -> None:
    source_root = tmp_path / "src" / "video_demo"
    source_root.mkdir(parents=True)
    (source_root / "module.py").write_text(source, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n",
        encoding="utf-8",
    )

    violations = audit_no_indexing_capability(tmp_path)

    assert "prohibited-index-operation" in {violation.rule for violation in violations}


def test_relative_local_operator_module_is_not_treated_as_standard_library(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src" / "video_demo"
    source_root.mkdir(parents=True)
    (source_root / "module.py").write_text(
        "from .operator import safe_helper\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = []\n",
        encoding="utf-8",
    )

    violations = audit_no_indexing_capability(tmp_path)

    assert violations == ()
