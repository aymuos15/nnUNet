"""Aggregated utilities namespace for nnUNetv2 with lazy submodule loading."""

from importlib import import_module
from typing import Any

__all__ = ("core", "dataset_io", "planning", "training_runtime", "visualization")


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
