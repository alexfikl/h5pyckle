# SPDX-FileCopyrightText: 2020-2022 Alexandru Fikl <alexfikl@gmail.com>
#
# SPDX-License-Identifier: MIT

from h5pyckle.base import PickleGroup
from h5pyckle.base import dump, load, dump_to_group, load_from_group, load_by_pattern
from h5pyckle.base import dumper, loader
from h5pyckle.base import (
    dump_sequence_to_group,
    dump_to_attribute,
    load_from_type,
    load_from_attribute,
    load_group_as_dict,
)

# NOTE: importing to have the types registered
import h5pyckle.interop_builtins  # noqa: F401
import h5pyckle.interop_numpy  # noqa: F401

__all__ = (
    "dump",
    "load",
    "dump_to_group",
    "load_from_group",
    "load_by_pattern",
    "PickleGroup",
    "dumper",
    "loader",
    "dump_sequence_to_group",
    "dump_to_attribute",
    "load_from_type",
    "load_from_attribute",
    "load_group_as_dict",
)
