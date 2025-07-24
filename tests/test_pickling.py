# SPDX-FileCopyrightText: 2020-2022 Alexandru Fikl <alexfikl@gmail.com>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.linalg as la
import pytest

from h5pyckle import dump, h5pyckable, load

logger = logging.getLogger(__name__)

dirname = pathlib.Path(__file__).parent / "pickles"
if not dirname.exists():
    dirname.mkdir()


def norm(x: np.ndarray[Any, np.dtype[Any]]) -> float:
    if x.dtype.char == "O":
        x = np.sqrt(x.dot(x))
    return float(la.norm(x))


def rnorm(
    x: np.ndarray[Any, np.dtype[Any]],
    y: np.ndarray[Any, np.dtype[Any]],
) -> float:
    norm_y = norm(y)
    if norm_y < 1.0e-15:
        norm_y = 1.0

    return norm(x - y) / norm_y


# {{{ test_pickling_dict


@pytest.mark.parametrize(
    "arg_in",
    [
        {},
        {"key": 1, "value": "dict"},
        {
            "author": {
                "name": "John",
                "family": "Smith",
                "affiliations": [
                    "University 1",
                    "Universtity 2",
                ],
            }
        },
        {
            "long_int": 123456789101112131415,
            "long_float": 3.14159265358979323846264338327950288419716939937510582097494,
        },
    ],
)
def test_pickling_dict(arg_in: dict[str, str]) -> None:
    filename = dirname / "pickle_dict.h5"

    dump(arg_in, filename)
    arg_out = load(filename)

    logger.info("expected: %s", arg_in)
    logger.info("got:      %s", arg_out)
    assert arg_in == arg_out


# }}}


# {{{ test_pickling_list_like


@pytest.mark.parametrize(
    "arg_in",
    [
        # [],
        # [1, 2, 3, 4, 5],
        [1, int, "string", 2.0],
    ],
)
def test_pickling_list_like(arg_in: list[Any]) -> None:
    filename = dirname / "pickle_list_like.h5"

    dump(
        {
            "list": arg_in,
            "tuple": tuple(arg_in),
            "set": set(arg_in),
        },
        filename,
    )
    out = load(filename)

    logger.info("expected: %s", arg_in)
    logger.info("got:      %s %s %s", out["list"], out["tuple"], out["set"])
    assert arg_in == out["list"]
    assert tuple(arg_in) == out["tuple"]
    assert set(arg_in) == out["set"]


# }}}


# {{{ test_pickling_numpy


@pytest.mark.parametrize("arg_in_type", ["scalar", "object"])
@pytest.mark.parametrize("dtype_in", [np.int32, np.float32, np.float64])
def test_pickling_numpy(arg_in_type: str, dtype_in: Any) -> None:
    filename = dirname / "pickle_numpy.h5"
    rng = np.random.default_rng(seed=42)

    if arg_in_type == "scalar":
        if dtype_in == np.int32:
            arg_in = np.arange(42, dtype=np.dtype(np.int32))
        else:
            arg_in = np.linspace(-1.0, 1.0, 42, dtype=np.dtype(dtype_in))

    elif arg_in_type == "object":
        from h5pyckle.interop_numpy import make_obj_array

        if dtype_in == np.int32:
            arg_in = make_obj_array([
                rng.integers(42, size=42, dtype=dtype_in) for _ in range(3)
            ])
        else:
            arg_in = make_obj_array([
                rng.random(size=42, dtype=dtype_in) for _ in range(3)
            ])
    else:
        raise ValueError(f"unknown type: '{arg_in_type}'")

    dump({"array": arg_in}, filename)
    arg_out = load(filename)["array"]

    error = rnorm(arg_out, arg_in)
    logger.info("error[%s, %s]: %.5e", arg_in_type, dtype_in, error)
    assert error < 1.0e-15

    assert arg_out.dtype == arg_in.dtype
    if arg_in.dtype.char == "O":
        assert arg_out[0].dtype == arg_in[0].dtype


# }}}


# {{{ test_pickling_numpy_subclass


def test_pickling_numpy_subclass() -> None:
    unyt = pytest.importorskip("unyt")
    x_in = unyt.unyt_array([1, 2, 3, 4, 5, 6], units=unyt.K)

    filename = dirname / "pickle_unyt.h5"
    dump(x_in, filename)
    x_out = load(filename)

    logger.info("expected: %s", x_in)
    logger.info("got:      %s", x_out)
    assert (x_in == x_out).all()


# }}}


# {{{ test_pickling_bytesio


def test_pickling_bytesio() -> None:
    import io

    bio = io.BytesIO()

    arg_in = {"name": "BytesIO", "values": (1, 2, 3), "nested": {"key": 42}}

    dump(arg_in, bio)
    arg_out = load(bio)

    logger.info("expected: %s", arg_in)
    logger.info("got:      %s", arg_out)
    assert arg_in == arg_out


# }}}


# {{{ test_pickling_numpy_scalar


def test_pickling_numpy_scalar() -> None:
    x = np.array(42)
    arg_in = {"x": x}

    filename = dirname / "pickle_np_scalar.h5"
    dump(arg_in, filename)
    arg_out = load(filename)

    logger.info("expected: %s", arg_in)
    logger.info("got:      %s", arg_out)
    assert arg_in == arg_out


# }}}


# {{{ test_pickling_scalar


def test_pickling_scalar() -> None:
    filename = dirname / "pickle_scalar.h5"

    arg_in = {"int": 1, "float": 3.14}
    dump(arg_in, filename)
    arg_out = load(filename)

    assert type(arg_in["int"]) == type(arg_out["int"])  # noqa: E721
    assert type(arg_in["float"]) == type(arg_out["float"])  # noqa: E721


# }}}


# {{{ test_pickling_dataclass


@h5pyckable
@dataclass(frozen=True)
class Employee:
    name: str
    position: str
    age: int
    date: tuple[int, int]
    paychecks: np.ndarray = field(compare=False)


def test_pickling_dataclass() -> None:
    filename = dirname / "pickle_dataclass.h5"

    from h5pyckle import dumper, loader

    assert dumper.dispatch(Employee).__name__ == "_dump_dataclass"
    assert loader.dispatch(Employee).__name__ == "_load_dataclass"

    arg_in = Employee(
        name="John Doe",
        position="Data Scientist",
        age=727,
        date=(4, 2022),
        paychecks=np.full((17,), 8700.1),
    )

    dump(arg_in, filename)
    arg_out = load(filename)

    assert arg_in == arg_out
    assert isinstance(arg_out, Employee)
    assert np.allclose(arg_in.paychecks, arg_out.paychecks, atol=3.0e-16)


# }}}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        _ = pytest.main([__file__])

# vim: fdm=marker
