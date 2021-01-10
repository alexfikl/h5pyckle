import os
import pickle

from functools import singledispatch
from contextlib import AbstractContextManager
from typing import Any, Dict, Optional

import h5py
import numpy as np


# {{{ context manager

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
    # TODO: piggyback on __getstate__ or something
    h5.attrs[name] = np.array(pickle.dumps(obj))


@singledispatch
def loader(h5: H5Group) -> Any:
    raise NotImplementedError


def load(h5: H5Group, *, pattern: Optional[str] = None):
    """
    :param h5: an open :class:`h5py.File`.
    :param pattern: if *None*, all the data from *h5* is loaded. Otherwise,
        the given pattern is searched for using :meth:`h5py.Group.visit`
        and only the match is returned. Currently this only returns one match.
    """
    groups = {}

    if pattern is None:
        for name in h5:
            if "type" in h5[name].attrs:
                groups[name] = load_from_type(h5[name])

        for name, value in h5.attrs.items():
            groups[name] = value
    else:
        groups = load_from_pattern(h5, pattern)

    return groups

# }}}


# {{{

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
        if "type" not in h5grp.attrs:
            raise RuntimeError(f"could not load '{name}' due to missing 'type'")

        return load_from_type(h5grp)

    found = None
    for key, value in h5grp.attrs.items():
        if pattern in key:
            found = value
            break

    if found is None:
        raise RuntimeError(f"attribute matching '{pattern}' not found in '{name}'")

    return found

# }}}


# {{{ python types

@dump.register(type)
def _(obj: type, h5: H5Group, *, name: Optional[str] = None):
    if name is None:
        name = "type"
    h5.attrs[name] = np.array(pickle.dumps(obj))


@loader.register(type)
def _(h5: H5Group) -> type:
    return pickle.loads(h5.attrs["type"])


@dump.register(dict)
def _(obj: Dict[str, Any], h5: H5Group, *, name: Optional[str] = None):
    for key, value in obj.items():
        try:
            dump(value, h5, name=key)
        except pickle.PicklingError:
            h5.attrs[key] = np.array(pickle.dumps(value))

# }}}
