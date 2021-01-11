import os

import pytest
import numpy as np
import numpy.linalg as la

from h5pyckle import dump_to_file, load_from_file


def norm(x):
    if x.dtype.char == "O":
        x = np.sqrt(x.dot(x))
    return la.norm(x)


def rnorm(x, y):
    norm_y = norm(y)
    if norm_y < 1.0e-15:
        norm_y = 1.0

    return norm(x - y) / norm_y


@pytest.mark.parametrize("arg_in", [
    {"key": 1, "value": "dict"},
    {"author": {"name": "John", "family": "Smith", "affiliations": [
        "University 1", "Universtity 2",
        ]}}
    ])
def test_pickling_dict(arg_in):
    filename = os.path.join(os.path.dirname(__file__), "pickle_dict.h5")

    dump_to_file(arg_in, filename)
    arg_out = load_from_file(filename)

    print("expected: ", arg_in)
    print("got:      ", arg_out)
    assert arg_in == arg_out


@pytest.mark.parametrize("arg_in", [
    [1, 2, 3, 4, 5],
    [1, int, "string", 1.0],
    ])
def test_pickling_list_like(arg_in):
    filename = os.path.join(os.path.dirname(__file__), "pickle_list_like.h5")

    dump_to_file({
        "list": arg_in,
        "tuple": tuple(arg_in),
        "set": set(arg_in),
        }, filename)
    out = load_from_file(filename)

    print("expected: ", arg_in)
    print("got:      ", out["list"], out["tuple"], out["set"])
    assert arg_in == out["list"]
    assert tuple(arg_in) == out["tuple"]
    assert set(arg_in) == out["set"]


@pytest.mark.parametrize("arg_in_type", ["scalar", "object"])
@pytest.mark.parametrize("dtype_in", [np.int32, np.float32, np.float64])
def test_pickling_numpy(arg_in_type, dtype_in):
    filename = os.path.join(os.path.dirname(__file__), "pickle_numpy.h5")

    if arg_in_type == "scalar":
        if dtype_in == np.int32:
            arg_in = np.arange(42, dtype=dtype_in)
        else:
            arg_in = np.linspace(-1.0, 1.0, 42, dtype=dtype_in)

    elif arg_in_type == "object":
        from h5pyckle.interop_numpy import make_obj_array
        if dtype_in == np.int32:
            arg_in = make_obj_array([
                np.random.randint(42, size=42, dtype=dtype_in) for _ in range(3)
                ])
        else:
            arg_in = make_obj_array([
                np.random.rand(42).astype(dtype_in) for _ in range(3)
                ])

    dump_to_file({"array": arg_in}, filename)
    arg_out = load_from_file(filename)["array"]

    error = rnorm(arg_out, arg_in)
    print("error[{}, {}]: {}".format(arg_in_type, dtype_in, error))
    assert error < 1.0e-15

    assert arg_out.dtype == arg_in.dtype
    if arg_in.dtype.char == "O":
        assert arg_out[0].dtype == arg_in[0].dtype


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
