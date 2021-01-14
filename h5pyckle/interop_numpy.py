from typing import Optional

import numpy as np

from h5pyckle.base import dumper, loader
from h5pyckle.base import PickleGroup, load_from_type, create_type


def make_obj_array(arrays):
    result = np.empty((len(arrays),), dtype=object)

    # 'result[:] = res_list' may look tempting, however:
    # https://github.com/numpy/numpy/issues/16564
    for i, ary in enumerate(arrays):
        result[i] = ary

    return result


# {{{ dtype

@dumper.register(np.dtype)
def _(obj: np.dtype, pkl: PickleGroup, *, name: Optional[str] = None):
    if name is None:
        pkl.attrs["dtype"] = np.array(obj.str.encode())
    else:
        grp = create_type(obj, pkl, name=name)
        grp.attrs["dtype"] = np.array(obj.str.encode())


@loader.register(np.dtype)
def _(pkl: PickleGroup) -> np.dtype:
    return np.dtype(pkl.attrs["dtype"])

# }}}


# {{{ ndarray

@dumper.register(np.ndarray)
def _(obj: np.ndarray, pkl: PickleGroup, *, name: Optional[str] = None):
    grp = create_type(obj, pkl, name=name)
    dumper(obj.dtype, grp)

    if obj.dtype.char == "O":
        for i, ary in enumerate(obj):
            dumper(ary, grp, name=f"entry_{i}")
    else:
        grp.create_dataset("entry", data=obj)


@loader.register(np.ndarray)
def _(pkl: PickleGroup) -> np.ndarray:
    dtype = load_from_type(pkl, obj_type=np.dtype)

    if dtype.char == "O":
        return make_obj_array([
            load_from_type(pkl[name]) for name in sorted(pkl)
            ])

    return pkl["entry"][:]

# }}}
