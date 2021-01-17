"""
In general, ``h5py`` has very good interopability with :mod:`numpy`. This
module offers some basic handling of :mod:`numpy` types

* :class:`~numpy.dtype`
* :class:`~numpy.ndarray` of all the supported types. Object arrays are also
  supported and are stored largely the same as a Python :class:`list`.

This is very much a work in progress and additional support is encouraged.
"""

from typing import Optional

import numpy as np

from h5pyckle.base import dumper, loader
from h5pyckle.base import PickleGroup, load_from_type


def make_obj_array(arrays):
    result = np.empty((len(arrays),), dtype=object)

    # 'result[:] = res_list' may look tempting, however:
    # https://github.com/numpy/numpy/issues/16564
    for i, ary in enumerate(arrays):
        result[i] = ary

    return result


# {{{ dtype

@dumper.register(np.dtype)
def _(obj: np.dtype, parent: PickleGroup, *, name: Optional[str] = None):
    if name is None:
        parent.attrs["dtype"] = np.array(obj.str.encode())
    else:
        grp = parent.create_type(name, obj)
        grp.attrs["dtype"] = np.array(obj.str.encode())


@loader.register(np.dtype)
def _(parent: PickleGroup) -> np.dtype:
    return np.dtype(parent.attrs["dtype"])

# }}}


# {{{ ndarray

@dumper.register(np.ndarray)
def _(obj: np.ndarray, parent: PickleGroup, *, name: Optional[str] = None):
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
def _(parent: PickleGroup) -> np.ndarray:
    dtype = load_from_type(parent, obj_type=np.dtype)

    if dtype.char == "O":
        obj = make_obj_array([
            load_from_type(parent[name]) for name in sorted(parent)
            ])
    else:
        obj = parent["entry"][:].view(parent.type)

    if "__dict__" in parent:
        fields = load_from_type(parent["__dict__"])
        if not hasattr(obj, "__dict__"):
            obj.__dict__ = {}

        obj.__dict__.update(fields)

    return obj

# }}}
