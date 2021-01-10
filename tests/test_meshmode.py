from functools import partial

import pytest
import numpy as np

import pyopencl as cl
from pyopencl.tools import (  # noqa: F401
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

from h5pyckle import dump, load
from h5pyckle.interop_meshmode import ArrayContextPickler


def norm(actx, x):
    if isinstance(x, np.ndarray):
        x = actx.np.sqrt(x.dot(x))

    return actx.np.linalg.norm(x)


def rnorm(actx, x, y):
    norm_y = norm(actx, y)
    if norm_y < 1.0e-15:
        norm_y = 1.0

    return norm(actx, x - y) / norm_y


# {{{ test_pickling

@pytest.mark.parametrize("ambient_dim", [2, 3])
def test_pickling(ctx_factory, ambient_dim, visualize=False, target_order=3):
    pytest.importorskip("meshmode")

    from meshmode.array_context import PyOpenCLArrayContext
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    # {{{ geometry

    import meshmode.mesh.generation as mmg
    if ambient_dim == 2:
        nelements = 32
        mesh = mmg.make_curve_mesh(
                partial(mmg.ellipse, 1.0),
                np.linspace(0.0, 1.0, nelements + 1),
                order=target_order)
    elif ambient_dim == 3:
        mesh = mmg.generate_icosphere(
                r=1.0,
                order=target_order,
                uniform_refinement_rounds=1)
    else:
        raise ValueError(f"unsupported dimension: {ambient_dim}")

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory
    discr = Discretization(actx, mesh,
            PolynomialWarpAndBlendGroupFactory(target_order))
    fine_discr = Discretization(actx, mesh,
            PolynomialWarpAndBlendGroupFactory(target_order + 3))

    from meshmode.discretization.connection import make_same_mesh_connection
    conn = make_same_mesh_connection(actx, discr, fine_discr)

    # }}}

    # {{{ pickle

    from meshmode.dof_array import thaw
    nodes = thaw(actx, discr.nodes())

    with ArrayContextPickler(actx, "meshmode.h5", mode="w") as h5:
        dump(nodes[0], h5, name="Field")
        dump(nodes, h5, name="Nodes")
        dump(mesh, h5, name="Mesh")
        dump(discr, h5, name="Discretization")
        dump(conn, h5, name="Connection")

    with ArrayContextPickler(actx, "meshmode.h5", mode="r") as h5:
        data = load(h5)
        x_new = data["Field"]
        nodes_new = data["Nodes"]
        mesh_new = data["Mesh"]
        discr_new = data["Discretization"]
        conn_new = data["Connection"]

    # }}}

    # {{{ check

    # check stored field is the same
    error = rnorm(actx, x_new, nodes[0])
    print("error[scalar]: %.5e" % error)
    assert error < 1.0e-15

    # check stored mesh is the same
    assert mesh == mesh_new
    assert mesh == discr_new.mesh

    # check object array is the same
    error = rnorm(actx, nodes_new, nodes)
    print("error[vector]: %.5e" % error)
    assert error < 1.0e-15

    # check discretization nodes are the same
    nodes_new = thaw(actx, discr_new.nodes())
    error = rnorm(actx, nodes_new, nodes)
    print("error[discr]:  %.5e" % error)
    assert error < 1.0e-15

    # check connection is the same
    nodes = thaw(actx, conn.from_discr.nodes())
    nodes_new = thaw(actx, conn_new.from_discr.nodes())
    error = rnorm(actx, nodes_new, nodes)
    print("error[conns]:  %.5e" % error)
    assert error < 1.0e-15

    # }}}

# }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
