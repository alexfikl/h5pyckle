from h5pyckle.base import PickleGroup
from h5pyckle.base import dump, load, dump_to_group, load_from_group, load_by_pattern
from h5pyckle.base import dumper, loader

# NOTE: importing to have the types registered
import h5pyckle.interop_builtins        # noqa: F401
import h5pyckle.interop_numpy           # noqa: F401

__all__ = [
    "dump", "load", "dump_to_group", "load_from_group", "load_by_pattern",
    "PickleGroup",

    "dumper", "loader",
]
