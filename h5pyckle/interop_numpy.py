from typing import Any, Optional, Sequence

import numpy as np

from h5pyckle.base import dumper, loader
from h5pyckle.base import PickleGroup, load_from_type


def make_obj_array(arrays: Sequence[Any]) -> np.ndarray:
    result = np.empty((len(arrays),), dtype=object)

    # 'result[:] = res_list' may look tempting, however:
    # https://github.com/numpy/numpy/issues/16564
    for i, ary in enumerate(arrays):
        result[i] = ary

    return result


def load_numpy_dataset(parent, name):
    ds = parent[name]
    if ds.shape == ():
        ds = np.array(ds[()])
    else:
        ds = ds[:]

    if issubclass(parent.pycls, np.ndarray):
        ds = ds.view(parent.pycls)

    return ds


# {{{ dtype


@dumper.register(np.dtype)
def _dump_dtype(
    obj: np.dtype, parent: PickleGroup, *, name: Optional[str] = None
) -> None:
    if name is None:
        parent.attrs["dtype"] = np.array(obj.str.encode())
    else:
        grp = parent.create_group(name)
        grp.append_type(obj, force_cls=np.dtype)
        grp.attrs["dtype"] = np.array(obj.str.encode())


@loader.register(np.dtype)
def _load_dtype(parent: PickleGroup) -> np.dtype:
    return np.dtype(parent.attrs["dtype"])


# }}}


# {{{ ndarray


@dumper.register(np.ndarray)
def _dump_ndarray(
    obj: np.ndarray, parent: PickleGroup, *, name: Optional[str] = None
) -> None:
    grp = parent.create_type(name, obj)

    dumper(obj.dtype, grp)
    if hasattr(obj, "__dict__"):
        dumper(obj.__dict__, grp, name="__dict__")

    if obj.dtype.char == "O":
        for i, ary in enumerate(obj):
            dumper(ary, grp, name=f"entry_{i}")
    else:
        grp.create_dataset("entry", data=obj)


@loader.register(np.ndarray)
def _load_ndarray(parent: PickleGroup) -> np.ndarray:
    dtype = load_from_type(parent, cls=np.dtype)

    if dtype.char == "O":
        obj = make_obj_array(
            [load_from_type(parent[name]) for name in sorted(parent)]
        )
    else:
        obj = load_numpy_dataset(parent, "entry")

    if "__dict__" in parent:
        fields = load_from_type(parent["__dict__"])
        if not hasattr(obj, "__dict__"):
            obj.__dict__ = {}

        obj.__dict__.update(fields)

    return obj


# }}}
