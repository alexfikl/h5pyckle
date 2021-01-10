import pytest
import numpy as np
import numpy.linalg as la

from h5pyckle import Pickler, dump, load
import h5pyckle.interop_numpy   # noqa: F401


def test_pickling():
    x = np.linspace(0.0, 1.0, 128)
    d = {"key": 1}

    with Pickler("pickling.h5", mode="w") as h5:
        dump(d, h5, name="dict")
        dump(x, h5, name="x")

    with Pickler("pickling.h5", mode="r") as h5:
        data = load(h5)

    print(data["dict"])
    assert d == data["dict"]

    error = la.norm(x - data["x"])
    print("error: %.5e" % error)
    assert error < 1.0e-15


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
