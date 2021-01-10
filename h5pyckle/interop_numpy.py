from typing import Optional

import numpy as np

from h5pyckle import H5Group, dump, loader, load_from_type


# {{{ dtype

@dump.register(np.dtype)
def _(obj: np.dtype, h5: H5Group, *, name: Optional[str] = None):
    if name is None:
        h5.attrs["dtype"] = np.array(obj.str.encode())
    else:
        grp = h5.create_group(name)
        dump(type(obj), grp)
        grp.attrs["dtype"] = np.array(obj.str.encode())


@loader.register(np.dtype)
def _(h5: H5Group) -> np.dtype:
    return np.dtype(h5.attrs["dtype"])

# }}}


# {{{ ndarray

@dump.register(np.ndarray)
def _(obj: np.ndarray, h5: H5Group, *, name: str):
    grp = h5.create_group(name)
    dump(type(obj), grp)
    dump(obj.dtype, grp)

    if obj.dtype.char == "O":
        for i, ary in enumerate(obj):
            dump(ary, grp, name=f"{name}_{i:05d}")
    else:
        h5.create_dataset(name, data=obj)


@loader.register(np.ndarray)
def _(h5: H5Group) -> np.ndarray:
    dtype = loader.dispatch(np.dtype)(h5)

    if dtype.char == "O":
        from pystopt.obj_array import make_obj_array
        return make_obj_array([
            load_from_type(h5[name]) for name in h5
            ])

    return h5[:]

# }}}
