"""Document compressor interfaces with lazy optional baseline imports."""

from importlib import import_module
from .base import BaseCompressor, SearchResult


_LAZY_IMPORTS = {
    "CompActCompressor": (".baselines.compact.compressor", "CompActCompressor"),
    "EXITCompressor": (".baselines.exit.compressor", "EXITCompressor"),
    "RefinerCompressor": (".baselines.refiner.compressor", "RefinerCompressor"),
    "RecompAbstractiveCompressor": (
        ".baselines.recomp_abst.compressor",
        "RecompAbstractiveCompressor",
    ),
    "RecompExtractiveCompressor": (
        ".baselines.recomp_extr.compressor",
        "RecompExtractiveCompressor",
    ),
    "LongLLMLinguaCompressor": (
        ".baselines.longllmlingua.compressor",
        "LongLLMLinguaCompressor",
    ),
}


def __getattr__(name):
    """Load a baseline only when it is explicitly requested."""

    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "BaseCompressor",
    "SearchResult",
    *_LAZY_IMPORTS,
]
