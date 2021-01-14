"""
.. currentmodule:: h5pyckle

.. autoclass:: PickleFile
    :no-show-inheritance:
.. autoclass:: PickleGroup
    :no-show-inheritance:

.. autofunction:: dump
.. autofunction:: dump_to_group
.. autofunction:: load
.. autofunction:: load_from_group
.. autofunction:: load_by_pattern

.. function:: dumper(obj: Any, parent: PickleGroup, *, name: Optional[str] = None)

    Function to implement for pickling various non-standard types. It is based
    on :func:`functools.singledispatch`, so new types can be registered
    with ``dumper.register(MyFancyType)`` or using optional typing in newer
    Python versions.

.. function:: loader(group: PickleGroup)

    Similar to :func:`dumper`, the loader is based on :func:`functools.singledispatch`.
    Unlike :func:`dumper`, this must register types explicitly with
    ``loader.register(MyFancyType)``.

Canonical Names
^^^^^^^^^^^^^^^

.. currentmodule:: h5pyckle.base

.. class:: PickleFile

    See :class:`h5pyckle.PickleFile`.

.. class:: PickleGroup

    See :class:`h5pyckle.PickleGroup`.
"""

import os
import pickle

from functools import singledispatch
from contextlib import AbstractContextManager
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np

# https://docs.h5py.org/en/stable/high/attr.html#attributes
_MAX_ATTRIBUTE_SIZE = 2**13


# {{{ wrapper for h5py.Group

class PickleGroup(h5py.Group):
    """
    .. attribute:: type

        If the group has type information, this attribute will return the
        corresponding class.

    .. attribute:: is_type

        If *True*, the group has type information that can be used to
        reconstruct the object.

    .. attribute:: context

        The instance of the :class:`~h5pyckle.PickleFile` used for I/O.

    .. automethod:: __init__
    .. automethod:: create_group
    .. automethod:: create_dataset
    .. automethod:: __getitem__

    .. automethod:: create_type
    .. automethod:: append_type

    """

    def __init__(self,
            gid: Union["h5py.Group", h5py.h5g.GroupID], *
            context: Optional["h5pyckle.PickleFile"] = None):
        if isinstance(gid, h5py.Group):
            gid = gid.id

        super().__init__(gid)
        self.context = context

        if context is None:
            self.h5_dset_options = {}
        else:
            self.h5_dset_options = self.context.h5_dset_options

    # {{{ h5py.Group overwrites

    # pylint: disable=arguments-differ
    def create_group(self, name: str, *,
            track_order: Optional[bool] = None) -> "h5pyckle.PickleGroup":
        """Thin wrapper around :meth:`h5py.Group.create_group`.

        :param name: name of the new group.
        :param track_order: If *True*, creation order in the group is preserved.
        """

        grp = super().create_group(name, track_order=track_order)
        return PickleGroup(self.context, grp)

    # pylint: disable=arguments-differ
    def create_dataset(self, name, *, shape=None, dtype=None, data=None):
        """Thin wrapper around :meth:`h5py.Group.create_dataset`. It uses
        the options from :attr:`~h5pyckle.PickleFile.h5_dset_options` to create
        the dataset.
        """

        return super().create_dataset(name,
                shape=shape, dtype=dtype, data=data,
                **self.h5_dset_options)

    def __getitem__(self, name: str) -> Any:
        """Retrieves an object in the group.

        :param name: a relative or absolute path in the group.
        """
        grp = super().__getitem__(name)

        if isinstance(grp, h5py.Group):
            return PickleGroup(self.context, grp)

        return grp

    # }}}

    # {{{ type handling

    def create_type(self, name: str, obj: Any) -> "h5pyckle.PickleGroup":
        """Creates a new group and adds appropriate type information.

        :param name: name of the new group.
        :param obj: object that will be pickled in the new group.
        """
        grp = self.create_group(name)
        grp.attrs["__type"] = np.void(pickle.dumps(type(obj)))

        return grp

    def append_type(self, name: str, obj: Any):
        """Append type information to the current group."""
        pkl.attrs["__type"] = np.void(pickle.dumps(type(obj)))

    @property
    def type(self) -> type:
        if not self.is_type:
            raise AttributeError(f"group '{self.name}' has no type information")

        return pickle.loads(self.attrs["__type"].tobytes())

    @property
    def is_type(self) -> bool:
        return "__type" in self.attrs

    # }}}

# }}}


# {{{ pickling context manager

class PickleFile(AbstractContextManager):
    """Handles an :class:`h5py.File` and other pickling options.

    .. attribute:: h5_file_options
    .. attribute:: h5_dset_options

    .. automethod:: __init__
    .. automethod:: __enter__
    .. automethod:: __exit__
    """

    def __init__(self,
            filename: os.PathLike, *,
            mode: str = "w",
            h5_file_options: Optional[Dict[str, Any]] = None,
            h5_dset_options: Optional[Dict[str, Any]] = None):
        if h5_file_options is None:
            h5_file_options = {}

        if h5_dset_options is None:
            h5_dset_options = {}

        self.filename = filename
        self.mode = mode

        self.h5 = None
        self.h5_file_options = h5_file_options
        self.h5_dset_options = h5_dset_options

    def __enter__(self) -> "h5pyckle.PickleGroup":
        """Open HDF5 file and enter context."""
        if self.h5 is not None:
            raise RuntimeError("cannot nest pickling contexts")

        self.h5 = h5py.File(self.filename,
                mode=self.mode,
                **self.h5_file_options)

        return PickleGroup(self, self.h5["/"])

    def __exit__(self, exc_type, exc_value, traceback):
        self.h5.close()
        self.h5 = None

# }}}


# {{{ type registering

@singledispatch
def dumper(obj: Any, parent: PickleGroup, *, name: Optional[str] = None):
    raise NotImplementedError(f"pickling of '{type(obj).__name__}'")


@singledispatch
def loader(group: PickleGroup) -> Any:
    raise NotImplementedError

# }}}


# {{{ io

def dump_to_group(obj: Any, parent: PickleGroup, *, name: Optional[str] = None):
    dumper(obj, parent, name=name)


def load_from_group(group: PickleGroup, *, exclude: Optional[List[str]] = None):
    """
    :param group: a group in an open :class:`h5py.File`.
    """
    if exclude is None:
        exclude = []

    groups = {}
    for name in pkl:
        if any(ex in name for ex in exclude):
            continue

        obj = pkl[name]
        if obj.is_type:
            groups[name] = load_from_type(obj)
        elif isinstance(obj, h5py.Dataset):
            groups[name] = obj[:]
        else:
            raise TypeError(f"cannot unpickle '{name}'")

    for name, value in pkl.attrs.items():
        if any(ex in name for ex in exclude):
            continue

        # NOTE: values pickled with `pickle` are stored as void
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

    return groups


def load_by_pattern(pkl: PickleGroup, *, pattern: str) -> Any:
    """
    :param pattern: the pattern is searched for using :meth:`h5py.Group.visit`
        and only the first match is returned. It searchs through groups,
        datasets, and their attributes.
    """
    def callback(name, obj):
        # TODO: should be easy to make this a regex
        if pattern in name:
            return name

        for key in obj.attrs:
            if pattern in key:
                return name

    name = pkl.visititems(callback)
    if name is None:
        raise ValueError(f"could not find any match for '{pattern}'")

    obj = pkl[name]
    if pattern in name:
        # found a group / dataset
        if obj.is_type:
            return load_from_type(h5grp)

        if isinstance(obj, h5py.Dataset):
            return obj[:]

        return load_from_group(h5grp)

    found = None
    for key, value in obj.attrs.items():
        if pattern in key:
            found = value
            break

    if found is None:
        # NOTE: this should really not fail since the patter was found somewhere
        raise RuntimeError(f"attribute matching '{pattern}' not found in '{name}'")

    try:
        value = pickle.loads(found)
    except pickle.PicklingError:
        value = found

    return value



def dump(obj: Any, filename: os.PathLike, *,
        h5_file_options: Optional[Dict[str, Any]] = None,
        h5_dset_options: Optional[Dict[str, Any]] = None):
    with PickleFile(filename, mode="w",
            h5_file_options=h5_file_options,
            h5_dset_options=h5_dset_options) as root:
        dump_to_group(obj, root, name=name)


def load(filename: os.PathLike):
    with PicklerFile(filename, mode="r") as root:
        return load_from_group(root)

# }}}


# {{{ helpers

def load_from_type(pkl: PickleGroup, *, obj_type=None):
    if obj_type is None:
        if "type" not in pkl.attrs:
            raise ValueError(f"cannot find type information in group '{pkl.name}'")

        obj_type = loader.dispatch(type)(pkl)

    return loader.dispatch(obj_type)(pkl)


def create_type(obj, pkl, *, name=None, only_with_name=False):
    if name is not None:
        pkl = pkl.create_group(name)
        dump(type(obj), pkl)
    else:
        if not only_with_name:
            dump(type(obj), pkl)

    return pkl

# }}}
