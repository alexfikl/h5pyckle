try:
    import dill as pickle
except ImportError:
    import pickle
from pickle import UnpicklingError

from numbers import Number
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import h5py
import numpy as np

from h5pyckle.base import dumper, loader
from h5pyckle.base import PickleGroup, load_from_type

# https://docs.h5py.org/en/stable/high/attr.html#attributes
_MAX_ATTRIBUTE_SIZE = 2**13


# {{{ object

@dumper.register(object)
def _(obj: object, parent: PickleGroup, *, name: Optional[str] = None):
    # NOTE: if we got here, it means no other (more specific) dumping method
    # was found and we should fall back to the generic pickle
    group = parent.create_type(name, obj)

    if hasattr(obj, "__getstate__"):
        group.attrs["__pickle"] = "getstate"

        state = obj.__getstate__()
        dumper(state, group, name="state")
    else:
        group.attrs["__pickle"] = "pickle"

        state = pickle.dumps(obj)
        if len(state) < _MAX_ATTRIBUTE_SIZE:
            group.attrs["state"] = np.void(state)
        else:
            group.create_dataset("state", data=np.array(state))


@loader.register(object)
def _(parent: PickleGroup) -> object:
    method = parent.attrs.get("__pickle")

    if method == "getstate":
        state = load_from_type(parent["state"])

        obj = parent.type()
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
def _(obj: Number, parent: PickleGroup, *, name: Optional[str] = None):
    parent.attrs[name] = obj


@dumper.register(str)
@dumper.register(bytes)
def _(obj: Union[str, bytes], parent: PickleGroup, *, name: Optional[str] = None):
    parent.attrs[name] = obj


@dumper.register(int)
def _(obj: Number, parent: PickleGroup, *, name: Optional[str] = None):
    try:
        parent.attrs[name] = obj
    except TypeError:
        # NOTE: managed to hit an arbitrary precision int
        grp = parent.create_type(name, obj)
        grp.attrs["value"] = repr(obj).encode()


@loader.register(int)
def _(parent: PickleGroup) -> int:
    from h5pyckle.base import load_from_attribute
    return parent.type(load_from_attribute("value", parent))

# }}}


# {{{ dict

@dumper.register(dict)
def _(obj: Dict[str, Any], parent: PickleGroup, *, name: Optional[str] = None):
    if name is None:
        group = parent.append_type(obj)
    else:
        group = parent.create_type(name, obj)

    for key, value in obj.items():
        dumper(value, group, name=key)


@loader.register(dict)
def _(parent: PickleGroup) -> Dict[str, Any]:
    from h5pyckle.base import load_group_as_dict
    cls = parent.type
    return cls(load_group_as_dict(parent))

# }}}


# {{{ list / tuple / set

@dumper.register(set)
@dumper.register(list)
@dumper.register(tuple)
def _(obj: Union[List, Set, Tuple], parent: PickleGroup, *, name: Optional[str] = None):
    from h5pyckle.base import dump_iterable_to_group
    dump_iterable_to_group(obj, parent, name=name)


@loader.register(list)
def _(parent: PickleGroup) -> List:
    from h5pyckle.base import load_group_as_dict

    if "entry" in parent:
        assert isinstance(parent["entry"], h5py.Dataset)
        return list(parent["entry"][:])

    entries = load_group_as_dict(parent)

    # NOTE: entries are all named "entry_XXXX", so we sort them by the index
    keys = sorted(entries, key=lambda el: int(el[6:]))
    entries = [entries[k] for k in keys]

    cls = parent.type
    return cls(entries)


@loader.register(tuple)
def _(parent: PickleGroup) -> Tuple:
    cls = parent.type
    return cls(load_from_type(parent, obj_type=list))


@loader.register(set)
def _(parent: PickleGroup) -> Set:
    cls = parent.type
    return cls(load_from_type(parent, obj_type=list))

# }}}
