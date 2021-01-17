import os

import pytest
import numpy as np
import numpy.linalg as la

from h5pyckle import dump, load


def norm(x):
    if x.dtype.char == "O":
        x = np.sqrt(x.dot(x))
    return la.norm(x)


def rnorm(x, y):
    norm_y = norm(y)
    if norm_y < 1.0e-15:
        norm_y = 1.0

    return norm(x - y) / norm_y


# {{{ test_pickling_dict

@pytest.mark.parametrize("arg_in", [
    {"key": 1, "value": "dict"},
    {"author": {"name": "John", "family": "Smith", "affiliations": [
        "University 1", "Universtity 2",
        ]}},
    {"long_int": 123456789101112131415,
        "long_float": 3.14159265358979323846264338327950288419716939937510582097494
        },
    ])
def test_pickling_dict(arg_in):
    filename = os.path.join(os.path.dirname(__file__), "pickle_dict.h5")

    dump(arg_in, filename)
    arg_out = load(filename)

    print("expected: ", arg_in)
    print("got:      ", arg_out)
    assert arg_in == arg_out

# }}}


# {{{ test_pickling_list_like

@pytest.mark.parametrize("arg_in", [
    [1, 2, 3, 4, 5],
    [1, int, "string", 2.0],
    ])
def test_pickling_list_like(arg_in):
    filename = os.path.join(os.path.dirname(__file__), "pickle_list_like.h5")

    dump({
        "list": arg_in,
        "tuple": tuple(arg_in),
        "set": set(arg_in),
        }, filename)
    out = load(filename)

    print("expected: ", arg_in)
    print("got:      ", out["list"], out["tuple"], out["set"])
    assert arg_in == out["list"]
    assert tuple(arg_in) == out["tuple"]
    assert set(arg_in) == out["set"]

# }}}


# {{{ test_pickling_numpy

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

    dump({"array": arg_in}, filename)
    arg_out = load(filename)["array"]

    error = rnorm(arg_out, arg_in)
    print("error[{}, {}]: {}".format(arg_in_type, dtype_in, error))
    assert error < 1.0e-15

    assert arg_out.dtype == arg_in.dtype
    if arg_in.dtype.char == "O":
        assert arg_out[0].dtype == arg_in[0].dtype

# }}}


# {{{ test_pickling_numpy_subclass

def test_pickling_numpy_subclass():
    unyt = pytest.importorskip("unyt")
    x_in = unyt.unyt_array([1, 2, 3, 4, 5, 6], units=unyt.K)

    filename = os.path.join(os.path.dirname(__file__), "pickle_unyt.h5")
    dump(x_in, filename)
    x_out = load(filename)

    assert (x_in == x_out).all()

# }}}


# {{{ test_pickling_bytesio

def test_pickling_bytesio():
    import io
    bio = io.BytesIO()

    arg_in = {
            "name": "BytesIO",
            "values": (1, 2, 3),
            "nested": {"key": 42}
            }

    dump(arg_in, bio)
    arg_out = load(bio)

    assert arg_in == arg_out

# }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
