try:
    import dill as pickle
except ImportError:
    # https://github.com/python/mypy/issues/1153
    import pickle       # type: ignore[no-redef]

from numbers import Number
from pickle import UnpicklingError
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import h5py
import numpy as np

from h5pyckle.base import dumper, loader
from h5pyckle.base import PickleGroup, load_from_type

# https://docs.h5py.org/en/stable/high/attr.html#attributes
_MAX_ATTRIBUTE_SIZE = 2**13


# {{{ object

@dumper.register(object)
def _dump_object(
        obj: object, parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
    # NOTE: if we got here, it means no other (more specific) dumping method
    # was found and we should fall back to the generic pickle
    group = parent.create_type(name, obj)

    if hasattr(obj, "__getstate__"):
        group.attrs["__pickle"] = "getstate"

        state = obj.__getstate__()      # type: ignore[attr-defined]
        dumper(state, group, name="state")
    else:
        group.attrs["__pickle"] = "pickle"

        state = pickle.dumps(obj)
        if len(state) < _MAX_ATTRIBUTE_SIZE:
            group.attrs["state"] = np.void(state)
        else:
            group.create_dataset("state", data=np.array(state))


@loader.register(object)
def _load_object(parent: PickleGroup) -> Any:
    method = parent.attrs.get("__pickle")

    if method == "getstate":
        state = load_from_type(parent["state"])

        obj = parent.pycls()
        obj.__setstate__(state)
        return obj
    elif method == "pickle":
        if "state" in parent:
            return pickle.loads(parent["state"][()])
        else:
            return pickle.loads(parent.attrs["state"].tobytes())
    else:
        raise UnpicklingError

# }}}


# {{{ scalar

@dumper.register(Number)
def _dump_number(
        obj: Number, parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
    parent.attrs[name] = obj


@dumper.register(str)
@dumper.register(bytes)
def _dump_string(
        obj: Union[str, bytes], parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
    parent.attrs[name] = obj


@dumper.register(int)
@dumper.register(float)
def _dump_int(
        obj: Number, parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
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
        obj: Dict[str, Any], parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
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
        obj: Union[List[Any], Set[Any], Tuple[Any, ...]], parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
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
