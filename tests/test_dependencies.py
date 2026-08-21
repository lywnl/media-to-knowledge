from __future__ import annotations

import warnings

from video_demo.audio import yamnet as yamnet_module


def test_tensorflow_hub_import_suppresses_only_known_pkg_resources_warning() -> None:
    import_tensorflow_hub = getattr(yamnet_module, "import_tensorflow_hub", None)
    assert callable(import_tensorflow_hub)

    def importer(name: str) -> object:
        assert name == "tensorflow_hub"
        warnings.warn(
            "pkg_resources is deprecated as an API. See upstream migration guidance.",
            UserWarning,
            stacklevel=2,
        )
        warnings.warn("保留其他依赖警告", RuntimeWarning, stacklevel=2)
        return object()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        imported = import_tensorflow_hub(importer=importer)

    assert imported is not None
    assert [str(item.message) for item in captured] == ["保留其他依赖警告"]
