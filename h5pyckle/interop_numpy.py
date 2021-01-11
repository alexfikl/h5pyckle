from typing import Optional

import numpy as np

from h5pyckle.base import H5Group
from h5pyckle.base import dump, loader, load_from_type, create_type


def make_obj_array(arrays):
    result = np.empty((len(arrays),), dtype=object)

    # 'result[:] = res_list' may look tempting, however:
    # https://github.com/numpy/numpy/issues/16564
    for i, ary in enumerate(arrays):
        result[i] = ary

    return result


# {{{ dtype

@dump.register(np.dtype)
def _(obj: np.dtype, h5: H5Group, *, name: Optional[str] = None):
    if name is None:
        h5.attrs["dtype"] = np.array(obj.str.encode())
    else:
        grp = create_type(obj, h5, name=name)
        grp.attrs["dtype"] = np.array(obj.str.encode())


@loader.register(np.dtype)
def _(h5: H5Group) -> np.dtype:
    return np.dtype(h5.attrs["dtype"])

# }}}


# {{{ ndarray

@dump.register(np.ndarray)
def _(obj: np.ndarray, h5: H5Group, *, name: Optional[str] = None):
    grp = create_type(obj, h5, name=name)
    dump(obj.dtype, grp)

    if obj.dtype.char == "O":
        for i, ary in enumerate(obj):
            dump(ary, grp, name=f"entry_{i}")
    else:
        grp.create_dataset("entry", data=obj)


@loader.register(np.ndarray)
def _(h5: H5Group) -> np.ndarray:
    dtype = load_from_type(h5, obj_type=np.dtype)

    if dtype.char == "O":
        return make_obj_array([
            load_from_type(h5[name]) for name in sorted(h5)
            ])

    return h5["entry"][:]

# }}}
