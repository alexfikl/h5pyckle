import pathlib
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

filename = pathlib.Path("pickled_nested.h5")

# {{{ nested dictionary

from h5pyckle import dump, load

arg_in = {
    "Author": {
        "FirstName": "Jane",
        "LastName": "Doe",
        "Affiliations": ["Some University", "Friendly Company"],
    }
}

dump(arg_in, filename)
arg_out = load(filename)
assert arg_in == arg_out

# }}}

filename = pathlib.Path("pickled_subgroup.h5")
if filename.exists():
    filename.unlink()

# {{{ subgroup

import h5py
from h5pyckle import dump_to_group, load_from_group, load_by_pattern

with h5py.File(filename, mode="a") as h5:
    subgroup = h5.require_group("pickling")
    dump_to_group(arg_in, subgroup)

with h5py.File(filename, mode="r") as h5:
    subgroup = h5["pickling"]
    arg_out = load_from_group(subgroup)
    first_name = load_by_pattern(subgroup, pattern="FirstName")

assert arg_in == arg_out
assert first_name == arg_in["Author"]["FirstName"]

# }}}

filename = pathlib.Path("pickled_custom.h5")

# {{{ custom type

from h5pyckle import PickleGroup, dumper, loader


@dataclass(frozen=True)
class CustomClass:
    name: str
    values: np.ndarray

    def __eq__(self, other: Any) -> bool:
        return self.name == other.name and bool(
            np.array_equal(self.values, other.values)
        )


@dumper.register(CustomClass)
def _dump_custom(
    obj: CustomClass, parent: PickleGroup, *, name: Optional[str] = None
) -> None:
    grp = parent.create_type(name, obj)
    grp.attrs["name"] = obj.name
    grp.create_dataset("values", data=obj.values)


@loader.register(CustomClass)
def _load_custom(parent: PickleGroup) -> CustomClass:
    name = parent.attrs["name"]
    values = parent["values"][:]
    return parent.pycls(name=name, values=values)


cc_in = CustomClass(name="George", values=np.ones(42))

dump(cc_in, filename)
cc_out = load(filename)

print(cc_in)
print(cc_out)
assert cc_in == cc_out

# }}}
