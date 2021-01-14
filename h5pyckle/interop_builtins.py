import pickle
from numbers import Number
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from h5pyckle.base import dumper, loader
from h5pyckle.base import PickleGroup, dump, load_from_type, create_type


def _dump_object(obj, pkl, name):
    from h5pyckle.base import _MAX_ATTRIBUTE_SIZE
    try:
        dumper(obj, pkl, name=name)
    except NotImplementedError:
        obj = pickle.dumps(obj)

        if len(obj) < _MAX_ATTRIBUTE_SIZE:
            pkl.attrs[name] = np.void(obj)
        else:
            pkl.create_dataset(name, data=np.array(obj))


# {{{ numeric

@dumper.register(Number)
def _(obj: Number, pkl: PickleGroup, *, name: Optional[str] = None):
    pkl.attrs[name] = obj

# }}}


# {{{ type

@dumper.register(type)
def _(obj: type, pkl: PickleGroup, *, name: Optional[str] = None):
    if name is None:
        name = "type"
    pkl.attrs[name] = np.void(pickle.dumps(obj))


@loader.register(type)
def _(pkl: PickleGroup) -> type:
    return pickle.loads(pkl.attrs["type"].tobytes())

# }}}


# {{{ dict

@dumper.register(dict)
def _(obj: Dict[str, Any], pkl: PickleGroup, *, name: Optional[str] = None):
    pkl = create_type(obj, pkl, name=name, only_with_name=True)
    for key, value in obj.items():
        _dump_object(value, pkl, name=key)


@loader.register(dict)
def _(pkl: PickleGroup) -> Dict[str, Any]:
    from h5pyckle.base import load
    return load(pkl, exclude=["type"])

# }}}


# {{{ list / tuple

@dumper.register(set)
@dumper.register(list)
@dumper.register(tuple)
def _(obj: Union[List, Set, Tuple], pkl: PickleGroup, *, name: Optional[str] = None):
    pkl = create_type(obj, pkl, name=name)
    is_number = all(isinstance(el, Number) for el in obj)

    if is_number:
        pkl.create_dataset("entry", data=np.array(list(obj)))
    else:
        for i, el in enumerate(obj):
            _dump_object(el, pkl, name=f"entry_{i}")


@loader.register(list)
def _(pkl: PickleGroup) -> List:
    from h5pyckle.base import load

    if "entry" in pkl:
        result = list(pkl["entry"][:])
    else:
        result = list(load(pkl, exclude=["type"]).values())

    return result


@loader.register(tuple)
def _(pkl: PickleGroup) -> Tuple:
    return tuple(load_from_type(pkl, obj_type=list))


@loader.register(set)
def _(pkl: PickleGroup) -> Set:
    return set(load_from_type(pkl, obj_type=list))

# }}}
