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
        state = obj.__getstate__()
        dumper(state, group, name="state")
    else:
        state = pickle.dumps(obj)
        if len(state) < _MAX_ATTRIBUTE_SIZE:
            group.attrs["pickle"] = np.void(state)
        else:
            group.create_dataset("pickle", data=state)


@loader.register(object)
def _(parent: PickleGroup) -> object:
    if "state" in parent:
        cls = parent.type
        state = load_from_type(parent["state"])
        return cls.__new__().__setstate__(state)
    elif "pickle" in parent:
        return pickle.loads(parent["pickle"][:])
    elif "pickle" in parent.attrs:
        return pickle.loads(parent.attrs["pickle"].tobytes())
    else:
        raise UnpicklingError

# }}}


# {{{ number

@dumper.register(Number)
def _(obj: Number, parent: PickleGroup, *, name: Optional[str] = None):
    parent.attrs[name] = obj


@dumper.register(str)
@dumper.register(bytes)
def _(obj: Union[str, bytes], parent: PickleGroup, *, name: Optional[str] = None):
    parent.attrs[name] = obj

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
    from h5pyckle.base import load_from_group
    cls = parent.type
    return cls(load_from_group(parent))

# }}}


# {{{ list / tuple / set

@dumper.register(set)
@dumper.register(list)
@dumper.register(tuple)
def _(obj: Union[List, Set, Tuple], parent: PickleGroup, *, name: Optional[str] = None):
    group = parent.create_type(name, obj)
    is_number = all(isinstance(el, Number) for el in obj)

    if is_number:
        group.create_dataset("entry", data=np.array(list(obj)))
    else:
        for i, el in enumerate(obj):
            dumper(el, group, name=f"entry_{i}")


@loader.register(list)
def _(parent: PickleGroup) -> List:
    from h5pyckle.base import load_from_group

    if "entry" in parent:
        assert isinstance(parent["entry"], h5py.Dataset)
        return list(parent["entry"][:])

    entries = load_from_group(parent)

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
