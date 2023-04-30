# SPDX-FileCopyrightText: 2020-2022 Alexandru Fikl <alexfikl@gmail.com>
#
# SPDX-License-Identifier: MIT

from numbers import Number
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import h5py

from h5pyckle.base import (
    PickleGroup,
    dumper,
    load_from_type,
    loader,
    pickle_from_group,
    pickle_to_group,
)

# {{{ object


@dumper.register(object)
def _dump_object(
    obj: object, parent: PickleGroup, *, name: Optional[str] = None
) -> None:
    # NOTE: if we got here, it means no other (more specific) dumping method
    # was found and we should fall back to the generic pickle
    group = parent.create_type(name, obj)

    group.attrs["__pickle"] = "pickle"
    pickle_to_group(obj, group, name="state")


@loader.register(object)
def _load_object(parent: PickleGroup) -> Any:
    return pickle_from_group("state", parent)


# }}}


# {{{ scalar


@dumper.register(Number)
def _dump_number(
    obj: Number, parent: PickleGroup, *, name: Optional[str] = None
) -> None:
    parent.attrs[name] = obj


@dumper.register(str)
@dumper.register(bytes)
def _dump_string(
    obj: Union[str, bytes], parent: PickleGroup, *, name: Optional[str] = None
) -> None:
    parent.attrs[name] = obj


@dumper.register(int)
@dumper.register(float)
def _dump_int(obj: Number, parent: PickleGroup, *, name: Optional[str] = None) -> None:
    # NOTE: managed to hit an arbitrary precision int
    grp = parent.create_type(name, obj)
    grp.attrs["value"] = repr(obj).encode()


@loader.register(int)
@loader.register(float)
def _load_int(parent: PickleGroup) -> int:
    from h5pyckle.base import load_from_attribute

    return parent.pycls(load_from_attribute("value", parent))


# }}}


# {{{ dict


@dumper.register(dict)
def _dump_dict(
    obj: Dict[str, Any], parent: PickleGroup, *, name: Optional[str] = None
) -> None:
    if name is None:
        group = parent.append_type(obj)
    else:
        group = parent.create_type(name, obj)

    for key, value in obj.items():
        dumper(value, group, name=key)


@loader.register(dict)
def _load_dict(parent: PickleGroup) -> Dict[str, Any]:
    from h5pyckle.base import load_group_as_dict

    return parent.pycls(load_group_as_dict(parent))


# }}}


# {{{ list / tuple / set


@dumper.register(set)
@dumper.register(list)
@dumper.register(tuple)
def _dump_sequence(
    obj: Union[List[Any], Set[Any], Tuple[Any, ...]],
    parent: PickleGroup,
    *,
    name: Optional[str] = None,
) -> None:
    from h5pyckle.base import dump_sequence_to_group

    dump_sequence_to_group(obj, parent, name=name)


@loader.register(list)
def _load_list(parent: PickleGroup) -> List[Any]:
    from h5pyckle.base import load_group_as_dict

    if "entry" in parent:
        assert isinstance(parent["entry"], h5py.Dataset)
        return list(parent["entry"][:])

    entries = load_group_as_dict(parent)

    # NOTE: entries are all named "entry_XXXX", so we sort them by the index
    keys = sorted(entries, key=lambda el: int(el[6:]))
    values = [entries[k] for k in keys]

    return parent.pycls(values)


@loader.register(tuple)
def _load_tuple(parent: PickleGroup) -> Tuple[Any, ...]:
    return parent.pycls(load_from_type(parent, cls=list))


@loader.register(set)
def _load_set(parent: PickleGroup) -> Set[Any]:
    return parent.pycls(load_from_type(parent, cls=list))


# }}}
