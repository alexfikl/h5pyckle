# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""
.. currentmodule:: h5pyckle

.. autofunction:: h5pyckable
"""

from __future__ import annotations

from dataclasses import Field, fields, is_dataclass
from typing import Any, ClassVar, Protocol, TypeVar

from h5pyckle.base import (
    PickleGroup,
    dump_to_attribute,
    dumper,
    load_from_attribute,
    load_from_type,
    loader,
)

_SCALAR_CLASSES = (int, float, str, bytes)


def _is_scalar_field(f: Field[Any]) -> bool:
    # FIXME: add support for delayed initialization
    if isinstance(f.type, str):
        return False

    # FIXME: add support for some typing, e.g. Optional, Union
    try:
        return issubclass(f.type, _SCALAR_CLASSES)
    except TypeError:
        return False


# {{{ dataclasses


# https://github.com/python/typeshed/blob/770724013de34af6f75fa444cdbb76d187b41875/stdlib/_typeshed/__init__.pyi#L329-L334
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


DataclassInstanceT = TypeVar("DataclassInstanceT", bound=DataclassInstance)


def _h5pyckle_dataclass(cls: type[DataclassInstanceT]) -> type[DataclassInstanceT]:
    # FIXME: just generate and exec the code to avoid the for loop?

    @dumper.register(cls)
    def _dump_dataclass(
        obj: Any, parent: PickleGroup, *, name: str | None = None
    ) -> None:
        group = parent.create_type(name, obj)

        for f in fields(cls):
            if not f.init:
                continue

            value = getattr(obj, f.name)
            if _is_scalar_field(f):
                dump_to_attribute(value, group, name=f.name)
            else:
                dumper(value, group, name=f.name)

    @loader.register(cls)
    def _load_dataclass(
        parent: PickleGroup,
    ) -> Any:
        kwargs = {}
        for f in fields(cls):
            if not f.init:
                continue

            value = load_from_attribute(f.name, parent)
            if f.name in parent:
                value = load_from_type(parent[f.name])

            kwargs[f.name] = value

        return parent.pycls(**kwargs)

    return cls


# }}}


# {{{ decorator


def h5pyckable(cls: type) -> type:
    """A class decorator to automatically make them compatible with :mod:`h5pyckle`.

    This will implement :func:`~h5pyckle.dumper` and :func:`~h5pyckle.loader`
    for the class *cls*. This may not always do the right thing and does not
    take care of missing arguments (e.g. during loading), so use with care.

    A standard usage is as follows

    .. code:: python

        @h5pyckable
        @dataclass(frozen=True)
        class MyClass:
            x: float
            y: int
            name: str
    """

    def wrapper(cls: type[object]) -> type[object]:
        if is_dataclass(cls):
            return _h5pyckle_dataclass(cls)

        raise TypeError(f"Cannot pickle type: '{cls.__name__}'")

    return wrapper(cls)


# }}}
