import pickle
from numbers import Number
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from h5pyckle.base import H5Group, dump, loader, load_from_type, create_type


def _dump_object(obj, h5, name):
    from h5pyckle.base import _MAX_ATTRIBUTE_SIZE
    try:
        dump(obj, h5, name=name)
    except NotImplementedError:
        obj = pickle.dumps(obj)

        if len(obj) < _MAX_ATTRIBUTE_SIZE:
            h5.attrs[name] = np.void(obj)
        else:
            h5.create_dataset(name, data=np.array(obj))


# {{{ numeric

@dump.register(Number)
def _(obj: Number, h5: H5Group, *, name: Optional[str] = None):
    h5.attrs[name] = obj

# }}}


# {{{ type

@dump.register(type)
def _(obj: type, h5: H5Group, *, name: Optional[str] = None):
    if name is None:
        name = "type"
    h5.attrs[name] = np.void(pickle.dumps(obj))


@loader.register(type)
def _(h5: H5Group) -> type:
    return pickle.loads(h5.attrs["type"].tobytes())

# }}}


# {{{ dict

@dump.register(dict)
def _(obj: Dict[str, Any], h5: H5Group, *, name: Optional[str] = None):
    h5 = create_type(obj, h5, name=name, only_with_name=True)
    for key, value in obj.items():
        _dump_object(value, h5, name=key)


@loader.register(dict)
def _(h5: H5Group) -> Dict[str, Any]:
    from h5pyckle.base import load
    return load(h5, exclude=["type"])

# }}}


# {{{ list / tuple

@dump.register(set)
@dump.register(list)
@dump.register(tuple)
def _(obj: Union[List, Set, Tuple], h5: H5Group, *, name: Optional[str] = None):
    h5 = create_type(obj, h5, name=name)
    is_number = all(isinstance(el, Number) for el in obj)

    if is_number:
        h5.create_dataset("entry", data=np.array(list(obj)))
    else:
        for i, el in enumerate(obj):
            _dump_object(el, h5, name=f"entry_{i}")


@loader.register(list)
def _(h5: H5Group) -> List:
    from h5pyckle.base import load

    if "entry" in h5:
        result = list(h5["entry"][:])
    else:
        result = list(load(h5, exclude=["type"]).values())

    return result


@loader.register(tuple)
def _(h5: H5Group) -> Tuple:
    return tuple(load_from_type(h5, obj_type=list))


@loader.register(set)
def _(h5: H5Group) -> Set:
    return set(load_from_type(h5, obj_type=list))

# }}}
