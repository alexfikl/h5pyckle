# SPDX-FileCopyrightText: 2020-2022 Alexandru Fikl <alexfikl@gmail.com>
#
# SPDX-License-Identifier: MIT

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
    Python versions. The object metadata and attributes are stored as a
    subgroup of the given parent group.

.. function:: loader(group: PickleGroup)

    Similar to :func:`dumper`, the loader is based on :func:`functools.singledispatch`.
    Unlike :func:`dumper`, this must register types explicitly with
    ``loader.register(MyFancyType)``.


.. autofunction:: dump_sequence_to_group
.. autofunction:: dump_to_attribute
.. autofunction:: load_from_type
.. autofunction:: load_from_attribute
.. autofunction:: load_group_as_dict
"""

try:
    import cloudpickle as pickle
except ImportError:
    try:
        import dill as pickle
    except ImportError:
        # https://github.com/python/mypy/issues/1153
        import pickle  # type: ignore[no-redef]

import io
import os
from functools import singledispatch
from contextlib import suppress
from pickle import UnpicklingError
from typing import Any, Dict, Optional, Sequence, Set, Tuple, Type, Union

import h5py
import numpy as np


# {{{ type aliases

# https://github.com/python/mypy/issues/5667
PathLike = Union[str, bytes, "os.PathLike[Any]", io.IOBase]

# }}}


# {{{ wrapper for h5py.Group

_H5PYCKLE_RESERVED_ATTRS = ["__type", "__type_name", "__pickle", "__version"]
_H5PYCKLE_VERSION = 1


class PickleGroup(h5py.Group):
    """Inherits from :class:`h5py.Group`.

    .. attribute:: pycls

        If the group has type information, this attribute will return the
        corresponding class.

    .. attribute:: has_type

        If *True*, the group has type information that can be used to
        reconstruct the object.

    .. attribute:: h5_dset_options

    .. automethod:: __init__
    .. automethod:: from_h5
    .. automethod:: replace

    .. automethod:: create_group
    .. automethod:: create_dataset
    .. automethod:: __getitem__

    .. automethod:: create_type
    .. automethod:: append_type
    """

    def __init__(
        self,
        gid: h5py.h5g.GroupID,
        *,
        h5_dset_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param gid: a :class:`~h5py.h5g.GroupID` used to initialize the group.
        :param h5_dset_options: a :class:`dict` of default options used
            when creating new datasets.
        """
        if h5_dset_options is None:
            h5_dset_options = {}

        super().__init__(gid)
        self.h5_dset_options = h5_dset_options
        self._type = None

    @classmethod
    def from_h5(cls, h5: Any) -> "PickleGroup":
        """Constructs a :class:`PickleGroup` from an ``h5py``-like object.

        :param h5: an existing ``h5py`` object.
        :returns: a new :class:`PickleGroup` constructed from the object.
        """
        if isinstance(h5, cls):
            return h5
        elif isinstance(h5, h5py.File):
            return cls(h5["/"].id)
        elif isinstance(h5, h5py.Group):
            return cls(h5.id)
        else:
            raise TypeError(f"unsupported parent type: {type(h5).__name__}")

    def replace(
        self,
        gid: Optional[h5py.h5g.GroupID] = None,
        h5_dset_options: Optional[Dict[str, Any]] = None,
    ) -> "PickleGroup":
        """Makes a copy of the current group.

        :param gid: a new :class:`~h5py.h5g.GroupID`.
        :param h5_dset_options: a new :class:`dict` of dataset creation options.
        """

        gid = self.id if gid is None else gid
        h5_dset_options = (
            self.h5_dset_options if h5_dset_options is None else h5_dset_options
        )

        return type(self)(gid, h5_dset_options=h5_dset_options)

    # {{{ h5py.Group overwrites

    # pylint: disable=arguments-differ
    def create_group(
        self, name: str, *, track_order: Optional[bool] = True
    ) -> "PickleGroup":
        """Thin wrapper around :meth:`h5py.Group.create_group`.

        :param name: name of the new group.
        :param track_order: If *True*, creation order in the group is preserved.
            This is the default to match the default :class:`dict` behavior.
        """
        grp = super().create_group(name, track_order=track_order)
        return self.replace(gid=grp.id)

    # pylint: disable=arguments-differ
    def create_dataset(
        self,
        name: str,
        *,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[Any] = None,
        data: Optional["np.ndarray"] = None,
    ) -> h5py.Dataset:
        """Thin wrapper around :meth:`h5py.Group.create_dataset`. It uses
        the options from :attr:`h5_dset_options` to create the dataset.

        :param name: name of the new dataset.
        :param shape: shape of the new dataset.
        :param dtype: :class:`numpy.dtype` of the new dataset.
        :param data: a :class:`numpy.ndarray` that contains the data for the
            dataset.
        """
        if "/" in name:
            raise ValueError(f"dataset names cannot contain a '/': '{name}'")

        return super().create_dataset(
            name, shape=shape, dtype=dtype, data=data, **self.h5_dset_options
        )

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

    def create_type(self, name: Optional[str], obj: Any) -> "PickleGroup":
        """Creates a new group and adds appropriate type information.

        :param name: name of the new group. If *None*, :meth:`append_type` is
            called instead.
        :param obj: object that will be pickled in the new group.
        """
        if name is None:
            return self.append_type(obj)

        grp = self.create_group(name)
        return grp.append_type(obj)

    def append_type(self, obj: Any, force_cls: Optional[type] = None) -> "PickleGroup":
        """Append type information to the current group.

        :param obj: object whose type information will be appended.
        :param force_cls: use this class as the object class instead of
            ``type(obj)``.
        """
        if self.has_type:
            raise RuntimeError(f"group '{self.name}' already has a type")

        if force_cls is None:
            cls = type(obj)
        else:
            assert isinstance(force_cls, type)
            cls = force_cls

        module = cls.__module__
        name = cls.__qualname__
        if not (module is None or module == str.__module__):
            name = f"{module}.{name}"

        self.attrs["__type"] = np.void(pickle.dumps(cls))
        self.attrs["__type_name"] = name.encode()
        self.attrs["__version"] = _H5PYCKLE_VERSION

        return self

    @property
    def pycls(self) -> Type[Any]:
        if not self.has_type:
            raise AttributeError(f"group '{self.name}' has no known type")

        if self._type is None:
            # pylint: disable=no-member
            cls = pickle.loads(self.attrs["__type"].tobytes())

            import importlib

            try:
                mod = importlib.import_module(cls.__module__)
                self._type = getattr(mod, cls.__name__)
            except AttributeError:
                self._type = cls

        return self._type  # type: ignore[return-value]

    @property
    def has_type(self) -> bool:
        return "__type" in self.attrs

    # }}}


# }}}


# {{{ type registering


@singledispatch
def dumper(obj: Any, parent: PickleGroup, *, name: Optional[str] = None) -> None:
    """
    :param obj: object to dump to the given group.
    :param parent: :class:`PicklerGroup` into which to dump the object.
    :param name: name of the newly created group. This is required for most
        cases, but can be omitted for some containers, e.g. a dictionary can
        be stored as separate groups based on its keys.
    """
    raise NotImplementedError(f"pickling of '{type(obj).__name__}'")


@singledispatch
def loader(parent: Any) -> Any:
    """
    :param parent: an instance of :class:`PickleGroup` from which to load an
        object.
    :returns: an object based on the type information stored in *parent*.
    """

    assert isinstance(parent, PickleGroup)

    if parent.has_type:
        raise NotImplementedError(
            f"group '{parent.name}' cannot be loaded: unsupported type"
            f"{parent.pycls.__name__}"
        )
    else:
        raise NotImplementedError(
            f"group '{parent.name}' cannot be loaded: has no type information"
        )


# }}}


# {{{ io


def dump_to_group(obj: Any, parent: PickleGroup, *, name: Optional[str] = None) -> None:
    """Stores pickled data in a specific HDF5 subgroup.

    :param parent: a group in an open :class:`h5py.File`.
    :param name: if provided, a new subgroup is created for this object.
    """
    parent = PickleGroup.from_h5(parent)
    dumper(obj, parent, name=name)


def load_from_group(
    parent: PickleGroup,
    *,
    exclude: Optional[Union[Set[str], Sequence[str]]] = None,
) -> Union[Any, Dict[str, Any]]:
    """
    :param parent: a group in an open :class:`h5py.File`.
    :param exclude: a list of patterns to exclude when loading data.
    """
    if exclude is None:
        exclude = []

    unique_exclude = set(list(exclude) + _H5PYCKLE_RESERVED_ATTRS)

    parent = PickleGroup.from_h5(parent)
    if parent.has_type:
        return load_from_type(parent)

    return load_group_as_dict(parent, exclude=unique_exclude)


def load_by_pattern(parent: PickleGroup, *, pattern: str) -> Any:
    """
    :param parent: a group in an open :class:`h5py.File`.
    :param pattern: the pattern is searched for using :meth:`h5py.Group.visit`
        and only the first match is returned. It searches through groups,
        datasets, and their attributes.
    """

    def callback(name: str, obj: Any) -> Optional[str]:
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


def dump(
    obj: Any,
    filename: PathLike,
    *,
    mode: str = "w",
    h5_file_options: Optional[Dict[str, Any]] = None,
    h5_dset_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    :param obj: object to pickle.
    :param filename: name of the file used for storage of pickled data.
    :param mode: see :attr:`h5py.File.mode` and the
        :ref:`h5py docs <h5py:file_open>`.
    :param h5_file_options: additional options passed directly to the
        :class:`h5py.File` constructor.
    :param h5_dset_options: additional options used when creating datasets.
        This is used when calling :meth:`PickleGroup.create_dataset`. See
        the :ref:`h5py docs <h5py:dataset>` for additional information
        about supported values.
    """
    if h5_file_options is None:
        h5_file_options = {}

    if h5_dset_options is None:
        h5_dset_options = {}

    with h5py.File(filename, mode=mode, **h5_file_options) as h5:
        root = PickleGroup(h5["/"].id, h5_dset_options=h5_dset_options)
        dump_to_group(obj, root)


def load(filename: PathLike) -> Union[Any, Dict[str, Any]]:
    """
    :param filename: file to load pickled data from.
    :returns: a :class:`dict` containing the full contents of the file. If
        only a subset of the file contains pickled data, use
        :func:`load_from_group` or :class:`load_by_pattern` instead.
    """
    with h5py.File(filename, mode="r") as h5:
        return load_from_group(h5)


# }}}


# {{{ dump helpers


def dump_sequence_to_group(
    obj: Union[Set[Any], Sequence[Any]],
    parent: PickleGroup,
    *,
    name: Optional[str] = None,
) -> None:
    """Dump the sequence *obj* into the group *parent* with the given *name*.

    If the sequence only contains :class:`numbers.Number`, then it is stored
    as a dataset instead of individual entries.

    :param obj: a class sastisfying the :class:`collections.abc.Sequence` protocol.
    """
    grp = parent.create_type(name, obj)
    obj = list(obj)

    from numbers import Number

    is_number = all(isinstance(el, Number) for el in obj)

    if is_number:
        grp.create_dataset("entry", data=np.array(obj))
    else:
        for i, el in enumerate(obj):
            dumper(el, grp, name=f"entry_{i}")


def dump_to_attribute(
    obj: Optional[Any], parent: PickleGroup, *, name: Optional[str] = None
) -> None:
    """Dumps the object into :attr:`h5py.Group.attrs`.

    If the object is not a basic type, such as a number or a string, then it
    is pickled first. See :func:`load_from_attribute` for the equivalent
    loading functionality.

    :param obj: an object to dump into the attributes of *parent*.
    """
    if obj is None:
        return

    from numbers import Number

    if isinstance(obj, (Number, str, bytes)):
        parent.attrs[name] = obj
    else:
        parent.attrs[name] = pickle.dumps(obj)


# }}}


# {{{ load helpers


def load_from_type(group: PickleGroup, *, cls: Optional[Type[Any]] = None) -> Any:
    """Load an object by using the dispatch of :func:`loader`.

    :param cls: if provided is used to determine the dispatch, otherwise
        :attr:`PickleGroup.pycls` is used.
    :returns: an instance of *cls* or :attr:`PickleGroup.pycls`, as appropriate.
    """

    if cls is None:
        if not group.has_type:
            raise ValueError(f"cannot find type information in group '{group.name}'")

        cls = group.pycls

    return loader.dispatch(cls)(group)


def load_from_attribute(name: str, group: PickleGroup) -> Optional[Any]:
    """Loads an attribute from the group *group*.

    This function also attempts to unpickle the attribute in case it was stored
    using :func:`pickle.dumps`.

    :param name: name of the attribute from *group*.
    """
    if name not in group.attrs:
        return None

    attr = group.attrs[name]
    if isinstance(attr, np.void):
        attr = attr.tobytes()

    if isinstance(attr, bytes):
        with suppress(UnpicklingError):
            attr = pickle.loads(attr)

    return attr


def load_group_as_dict(
    parent: PickleGroup,
    exclude: Optional[Union[Set[str], Sequence[str]]] = None,
) -> Dict[str, Any]:
    """Loads all the datasets and attributes of *parent* into a dictionary.

    :param exclude: a list of datasets or attributes to exclude when loading.
    """
    if exclude is None:
        exclude = []
    unique_exclude = set(list(exclude) + _H5PYCKLE_RESERVED_ATTRS)

    from h5py import Dataset

    groups = {}
    for name in parent:
        if any(ex in name for ex in unique_exclude):
            continue

        obj = parent[name]
        if obj.has_type:
            groups[name] = load_from_type(obj)
        elif isinstance(obj, Dataset):
            groups[name] = obj[:]
        else:
            raise TypeError(f"cannot unpickle '{name}'")

    for name in parent.attrs:
        if any(ex in name for ex in unique_exclude):
            continue

        groups[name] = load_from_attribute(name, parent)

    return groups


# }}}
