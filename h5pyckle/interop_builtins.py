import pickle
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from h5pyckle.base import H5Group, dump, loader, load_from_type, create_type


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
    h5.attrs[name] = np.array(pickle.dumps(obj))


@loader.register(type)
def _(h5: H5Group) -> type:
    return pickle.loads(h5.attrs["type"])

# }}}


# {{{ dict

@dump.register(dict)
def _(obj: Dict[str, Any], h5: H5Group, *, name: Optional[str] = None):
    from h5pyckle.base import _MAX_ATTRIBUTE_SIZE
    h5 = create_type(obj, h5, name=name, only_with_name=True)

    for key, value in obj.items():
        try:
            dump(value, h5, name=key)
        except (pickle.PicklingError, NotImplementedError):
            value = pickle.dumps(value)

            if len(value) < _MAX_ATTRIBUTE_SIZE:
                h5.attrs[key] = np.void(value)
            else:
                h5.create_dataset(key, data=np.array(value))


@loader.register(dict)
def _(h5: H5Group) -> Dict[str, Any]:
    from h5pyckle.base import load
    return load(h5, exclude=["type"])

# }}}


# {{{ list / tuple

@dump.register(list)
@dump.register(tuple)
def _(obj: Union[List, Tuple], h5: H5Group, *, name: Optional[str] = None):
    h5 = create_type(obj, h5, name=name)
    is_number = all(isinstance(el, Number) for el in obj)

    if is_number:
        h5.create_dataset("entry", data=np.array(obj))
    else:
        for i, el in enumerate(obj):
            dump(el, h5, name=f"entry_{i}")


@loader.register(list)
def _(h5: H5Group) -> List:
    from h5pyckle.base import load

    if "entry" in h5:
        result = list(h5["entry"][:])
    else:
        result = []
        for name in h5:
            result.extend(load(h5[name]).values())

    return result


@loader.register(tuple)
def _(h5: H5Group) -> Tuple:
    return tuple(load_from_type(h5, obj_type=list))

# }}}
