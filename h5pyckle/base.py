import os
import pickle

from functools import singledispatch
from contextlib import AbstractContextManager
from typing import Any, List, Optional

import h5py
import numpy as np

# https://docs.h5py.org/en/stable/high/attr.html#attributes
_MAX_ATTRIBUTE_SIZE = 2**13


# {{{ wrapper

class H5Group(h5py.Group):
    def __init__(self, pickler, gid):
        super().__init__(gid)
        self.pickler = pickler

    # pylint: disable=arguments-differ
    def create_group(self, name, *, track_order=None):
        grp = super().create_group(name, track_order=track_order)
        return H5Group(self.pickler, grp.id)

    # pylint: disable=arguments-differ
    def create_dataset(self, name, *, shape=None, dtype=None, data=None):
        return super().create_dataset(name,
                shape=shape, dtype=dtype, data=data,
                **self.pickler.h5_dset_options)

    def __getitem__(self, name):
        grp = super().__getitem__(name)

        if isinstance(grp, h5py.Group):
            return H5Group(self.pickler, grp.id)

        # it's probably a dataset
        return grp

# }}}


# {{{ context manager

class Pickler(AbstractContextManager):
    def __init__(self,
            filename: os.PathLike, *,
            mode: str = "w",
            h5_file_options: Optional[dict] = None,
            h5_dset_options: Optional[dict] = None):
        if h5_file_options is None:
            h5_file_options = {}

        if h5_dset_options is None:
            h5_dset_options = {}

        self.filename = filename
        self.mode = mode

        self.h5 = None
        self.h5_file_options = h5_file_options
        self.h5_dset_options = h5_dset_options

    def __enter__(self):
        if self.h5 is not None:
            raise RuntimeError("cannot nest pickling contexts")

        self.h5 = h5py.File(self.filename,
                mode=self.mode,
                **self.h5_file_options)

        return H5Group(self, self.h5["/"].id)

    def __exit__(self, exc_type, exc_value, traceback):
        self.h5.close()
        self.h5 = None

# }}}


# {{{ interface

@singledispatch
def dump(obj: Any, h5: H5Group, *, name: Optional[str] = None):
    raise NotImplementedError(f"pickling of '{type(obj).__name__}'")


@singledispatch
def loader(h5: H5Group) -> Any:
    if "type" in h5.attrs:
        cls = load_from_type(type)(h5)
        raise NotImplementedError(f"unpickling of '{cls.__name__}'")

    raise NotImplementedError


def load(h5: H5Group, *,
        pattern: Optional[str] = None,
        exclude: Optional[List[str]] = None):
    """
    :param h5: an open :class:`h5py.File`.
    :param pattern: if *None*, all the data from *h5* is loaded. Otherwise,
        the given pattern is searched for using :meth:`h5py.Group.visit`
        and only the match is returned. Currently this only returns one match.
    """
    if exclude is None:
        exclude = []

    groups = {}
    if pattern is None:
        for name in h5:
            if any(ex in name for ex in exclude):
                continue

            obj = h5[name]
            if "type" in obj.attrs:
                groups[name] = load_from_type(obj)
            elif isinstance(obj, h5py.Dataset):
                groups[name] = obj[:]
            else:
                raise TypeError(f"cannot unpickle '{name}'")

        for name, value in h5.attrs.items():
            if any(ex in name for ex in exclude):
                continue

            if isinstance(value, np.void):
                value = value.tobytes()

            if isinstance(value, bytes):
                try:
                    real_value = pickle.loads(value)
                except pickle.PicklingError:
                    real_value = value
            else:
                real_value = value

            groups[name] = real_value
    else:
        groups = load_from_pattern(h5, pattern)

    return groups


def dump_to_file(obj, filename: os.PathLike, *, name: Optional[str] = None):
    with Pickler(filename, mode="w") as root:
        dump(obj, root, name=name)


def load_from_file(filename: os.PathLike):
    with Pickler(filename, mode="r") as root:
        return load(root)

# }}}


# {{{ loading helpers

def load_from_type(h5: H5Group, *, obj_type=None):
    if obj_type is None:
        if "type" not in h5.attrs:
            raise ValueError(f"cannot find type information in group '{h5.name}'")

        obj_type = loader.dispatch(type)(h5)

    return loader.dispatch(obj_type)(h5)


def load_from_pattern(h5: H5Group, pattern: str):
    def callback(name, obj):
        if pattern in name:
            return name

        found = None
        for key in obj.attrs:
            if pattern in key:
                found = name
                break

        return found

    name = h5.visititems(callback)
    if name is None:
        raise ValueError(f"could not find any match for '{pattern}'")

    h5grp = h5[name]
    if pattern in name:
        if "type" in h5grp.attrs:
            return load_from_type(h5grp)

        return load(h5grp)

    found = None
    for key, value in h5grp.attrs.items():
        if pattern in key:
            found = value
            break

    if found is None:
        # NOTE: this should really not fail since the patter was found somewhere
        raise RuntimeError(f"attribute matching '{pattern}' not found in '{name}'")

    return found


def create_type(obj, h5, *, name=None, only_with_name=False):
    if name is not None:
        h5 = h5.create_group(name)
        dump(type(obj), h5)
    else:
        if not only_with_name:
            dump(type(obj), h5)

    return h5

# }}}
