"""
.. currentmodule:: h5pyckle

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

.. class:: PickleGroup

    See :class:`h5pyckle.PickleGroup`.
"""

import os
try:
    import dill as pickle
except ImportError:
    import pickle

from functools import singledispatch
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np


# {{{ wrapper for h5py.Group

class PickleGroup(h5py.Group):
    """
    .. attribute:: type

        If the group has type information, this attribute will return the
        corresponding class.

    .. attribute:: has_type

        If *True*, the group has type information that can be used to
        reconstruct the object.

    .. attribute:: h5_dset_options

    .. automethod:: __init__
    .. automethod:: from_h5

    .. automethod:: create_group
    .. automethod:: create_dataset
    .. automethod:: __getitem__

    .. automethod:: create_type
    .. automethod:: append_type
    """

    def __init__(self,
            gid: h5py.h5g.GroupID, *,
            h5_dset_options: Optional[Dict[str, Any]] = None):
        if h5_dset_options is None:
            h5_dset_options = {}

        super().__init__(gid)
        self.h5_dset_options = h5_dset_options
        self._type = None

    @classmethod
    def from_h5(cls, h5) -> "PickleGroup":
        """Constructs a :class:`PickleGroup` from an ``h5py`` object."""
        if isinstance(h5, cls):
            return h5
        elif isinstance(h5, h5py.File):
            return cls(h5["/"].id)
        elif isinstance(h5, h5py.Group):
            return cls(h5.id)
        else:
            raise TypeError(f"unsupported parent type: {type(h5).__name__}")

    def replace(self, **kwargs):
        kwargs["gid"] = kwargs.get("gid", self.id)
        kwargs["h5_dset_options"] = kwargs.get(
                "h5_dset_options", self.h5_dset_options)

        return type(self)(**kwargs)

    # {{{ h5py.Group overwrites

    # pylint: disable=arguments-differ
    def create_group(self, name: str, *,
            track_order: Optional[bool] = True) -> "PickleGroup":
        """Thin wrapper around :meth:`h5py.Group.create_group`.

        :param name: name of the new group.
        :param track_order: If *True*, creation order in the group is preserved.
            This is the default to match the default :class:`dict` behavior.
        """
        grp = super().create_group(name, track_order=track_order)
        return self.replace(gid=grp.id)

    # pylint: disable=arguments-differ
    def create_dataset(self, name, *, shape=None, dtype=None, data=None):
        """Thin wrapper around :meth:`h5py.Group.create_dataset`. It uses
        the options from :attr:`h5_dset_options` to create the dataset.
        """

        return super().create_dataset(name,
                shape=shape, dtype=dtype, data=data,
                **self.h5_dset_options)

    def __getitem__(self, name: str) -> Any:
        """Retrieves an object in the group.

        :param name: a relative or absolute path in the group.
        """
        item = super().__getitem__(name)
        if isinstance(item, h5py.Group):
            return self.replace(gid=item.id)

        return item

    # }}}

    # {{{ type handling

    @property
    def reserved_names(self):
        return ["__type", "__type_name"]

    def create_type(self, name: str, obj: Any) -> "PickleGroup":
        """Creates a new group and adds appropriate type information.

        :param name: name of the new group. If *None*, :meth:`append_type` is
            called instead.
        :param obj: object that will be pickled in the new group.
        """
        if name is None:
            return self.append_type(obj)

        grp = self.create_group(name)
        return grp.append_type(obj)

    def append_type(self, obj: Any):
        """Append type information to the current group."""
        if self.has_type:
            raise RuntimeError(f"group '{self.name}' already has a type")

        self.attrs["__type"] = np.void(pickle.dumps(type(obj)))
        self.attrs["__type_name"] = np.array(type(obj).__name__.encode())
        return self

    @property
    def type(self) -> type:
        if not self.has_type:
            raise AttributeError(f"group '{self.name}' has no known type")

        if self._type is None:
            # pylint: disable=no-member
            cls = pickle.loads(self.attrs["__type"].tobytes())

            import importlib
            mod = importlib.import_module(cls.__module__)
            self._type = getattr(mod, cls.__name__)

        return self._type

    @property
    def has_type(self) -> bool:
        return "__type" in self.attrs

    # }}}

# }}}


# {{{ type registering

@singledispatch
def dumper(obj: Any, parent: PickleGroup, *, name: Optional[str] = None):
    raise NotImplementedError(f"pickling of '{type(obj).__name__}'")


@singledispatch
def loader(parent: PickleGroup) -> Any:
    raise NotImplementedError

# }}}


# {{{ io

def dump_to_group(
        obj: Any,
        parent: PickleGroup, *,
        name: Optional[str] = None):
    """
    :param parent: a group in an open :class:`h5py.File`.
    :param name: if provided, a new subgroup is created for this object.
    """
    parent = PickleGroup.from_h5(parent)
    dumper(obj, parent, name=name)


def load_from_group(
        parent: PickleGroup, *,
        exclude: Optional[List[str]] = None) -> Union[Any, Dict[str, Any]]:
    """
    :param parent: a group in an open :class:`h5py.File`.
    :param exclude: a list of patterns to exclude when loading data.
    """
    if exclude is None:
        exclude = []

    parent = PickleGroup.from_h5(parent)
    if parent.has_type:
        return load_from_type(parent)

    exclude += parent.reserved_names
    return load_group_as_dict(parent, exclude=exclude)


def load_by_pattern(parent: PickleGroup, *, pattern: str) -> Any:
    """
    :param parent: a group in an open :class:`h5py.File`.
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

        return None

    parent = PickleGroup.from_h5(parent)
    name = parent.visititems(callback)
    if name is None:
        raise ValueError(f"could not find any match for '{pattern}'")

    obj = parent[name]
    if pattern in name:
        # found a group / dataset
        if obj.has_type:
            return load_from_type(obj)
        if isinstance(obj, h5py.Dataset):
            return obj[:]

        return load_from_group(obj)

    found = None
    for key in obj.attrs:
        if pattern in key:
            found = key
            break

    if found is None:
        # NOTE: this should really not fail since the pattern was found somewhere
        raise RuntimeError(f"attribute matching '{pattern}' not found in '{name}'")

    return load_from_attribute(found, obj)


def dump(obj: Any, filename: os.PathLike, *,
        mode: str = "w",
        h5_file_options: Optional[Dict[str, Any]] = None,
        h5_dset_options: Optional[Dict[str, Any]] = None):
    if h5_file_options is None:
        h5_file_options = {}

    if h5_dset_options is None:
        h5_dset_options = {}

    with h5py.File(filename, mode=mode, **h5_file_options) as h5:
        root = PickleGroup(h5["/"].id, h5_dset_options=h5_dset_options)
        dump_to_group(obj, root)


def load(filename: os.PathLike) -> Dict[str, Any]:
    with h5py.File(filename, mode="r") as h5:
        return load_from_group(h5)

# }}}


# {{{ helpers

def load_from_type(group: PickleGroup, *, obj_type=None):
    if obj_type is None:
        if not group.has_type:
            raise ValueError(f"cannot find type information in group '{group.name}'")

        obj_type = group.type

    return loader.dispatch(obj_type)(group)


def load_from_attribute(name: str, group: PickleGroup):
    attr = group.attrs[name]
    if isinstance(attr, np.void):
        attr = attr.tobytes()

    if isinstance(attr, bytes):
        try:
            attr = pickle.loads(attr)
        except pickle.UnpicklingError:
            pass

    return attr


def load_group_as_dict(
        parent: PickleGroup,
        exclude: Optional[List[str]] = None) -> Dict[str, Any]:
    if exclude is None:
        exclude = parent.reserved_names

    groups = {}
    for name in parent:
        if any(ex in name for ex in exclude):
            continue

        obj = parent[name]
        if obj.has_type:
            groups[name] = load_from_type(obj)
        elif isinstance(obj, h5py.Dataset):
            groups[name] = obj[:]
        else:
            raise TypeError(f"cannot unpickle '{name}'")

    for name in parent.attrs:
        if any(ex in name for ex in exclude):
            continue

        groups[name] = load_from_attribute(name, parent)

    return groups

# }}}
