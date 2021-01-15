import os
from dataclasses import dataclass

import h5py
import numpy as np

from h5pyckle import PickleGroup, dumper, loader


# # {{{ nested dictionary

from h5pyckle import dump, load

arg_in = {
    "Author": {
        "FirstName": "Jane",
        "LastName": "Doe",
        "Affiliations": ["Some University", "Friendly Company"],
    }
}

dump(arg_in, "pickled_nested.h5")
arg_out = load("pickled_nested.h5")
assert arg_in == arg_out

# # }}}

if os.path.exists("pickled_subgroup.h5"):
    os.remove("pickled_subgroup.h5")

# # {{{ subgroup

from h5pyckle import dump_to_group, load_from_group, load_by_pattern

with h5py.File("pickled_subgroup.h5", mode="a") as h5:
    subgroup = h5.require_group("pickling")
    dump_to_group(arg_in, subgroup)

with h5py.File("pickled_subgroup.h5", mode="r") as h5:
    subgroup = h5["pickling"]
    arg_out = load_from_group(subgroup)
    first_name = load_by_pattern(subgroup, pattern="FirstName")

assert arg_in == arg_out
assert first_name == arg_in["Author"]["FirstName"]

# # }}}


# {{{ custom type

@dataclass
class CustomClass:
    name: str
    values: np.ndarray

    def __eq__(self, other):
        return self.name == other.name \
                and np.array_equal(self.values, other.values)


@dumper.register(CustomClass)
def _(obj: CustomClass, parent: PickleGroup, name: str = None):
    grp = parent.create_type(name, obj)
    grp.attrs["name"] = obj.name
    grp.create_dataset("values", data=obj.values)


@loader.register(CustomClass)
def _(parent: PickleGroup) -> CustomClass:
    name = parent.attrs["name"]
    values = parent["values"][:]
    return CustomClass(name=name, values=values)


cc_in = CustomClass(name="George", values=np.ones(42))

dump({"cc": cc_in}, "pickled_custom.h5")
cc_out = load("pickled_custom.h5")["cc"]
assert cc_in == cc_out

# }}}
