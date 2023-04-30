# SPDX-FileCopyrightText: 2020-2022 Alexandru Fikl <alexfikl@gmail.com>
#
# SPDX-License-Identifier: MIT

from importlib import metadata

# NOTE: importing to have the types registered
import h5pyckle.interop_builtins  # noqa: F401
import h5pyckle.interop_numpy  # noqa: F401
from h5pyckle.base import (
    PickleGroup,
    dump,
    dump_sequence_to_group,
    dump_to_attribute,
    dump_to_group,
    dumper,
    load,
    load_by_pattern,
    load_from_attribute,
    load_from_group,
    load_from_type,
    load_group_as_dict,
    loader,
)
from h5pyckle.decorator import h5pyckable

__version__ = metadata.version("h5pyckle")

__all__ = (
    "PickleGroup",
    "dump",
    "dump_sequence_to_group",
    "dump_to_attribute",
    "dump_to_group",
    "dumper",
    "h5pyckable",
    "load",
    "load_by_pattern",
    "load_from_attribute",
    "load_from_group",
    "load_from_type",
    "load_group_as_dict",
    "loader",
)
