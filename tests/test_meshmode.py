from dataclasses import dataclass
from functools import partial
import pathlib
from typing import Any, cast

import pytest
import numpy as np

import logging

logger = logging.getLogger(__name__)
dirname = pathlib.Path(__file__).parent

try:
    from pytools import Record
except ImportError:
    # https://github.com/python/mypy/issues/1153
    class Record(dict):  # type: ignore[no-redef]
        pass


def norm(actx: Any, x: "np.ndarray[Any, Any]") -> float:
    if isinstance(x, np.ndarray):
        x = actx.np.sqrt(x @ x)

    from meshmode.dof_array import flat_norm

    return flat_norm(x)


def rnorm(actx: Any, x: "np.ndarray[Any, Any]", y: "np.ndarray[Any, Any]") -> float:
    norm_y = norm(actx, y)
    if norm_y < 1.0e-15:
        norm_y = 1.0

    return actx.to_numpy(norm(actx, x - y) / norm_y)


# {{{ test_discretization_pickling


@pytest.mark.parametrize("ambient_dim", [2, 3])
def test_discretization_pickling(
    ambient_dim: int, visualize: bool = False, target_order: int = 3
) -> None:
    """Tests that the interop_meshmode types can all be dumped/loaded correctly."""

    pytest.importorskip("meshmode")

    import pyopencl as cl
    from meshmode.array_context import PyOpenCLArrayContext

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

    # {{{ geometry

    import meshmode.mesh.generation as mmg

    if ambient_dim == 2:
        nelements = 32
        mesh = mmg.make_curve_mesh(
            partial(mmg.ellipse, 1.0),
            np.linspace(0.0, 1.0, nelements + 1),
            order=target_order,
        )
    elif ambient_dim == 3:
        mesh = mmg.generate_sphere(
            r=1.0, order=target_order, uniform_refinement_rounds=1
        )
    else:
        raise ValueError(f"unsupported dimension: {ambient_dim}")

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import default_simplex_group_factory

    discr = Discretization(
        actx, mesh, default_simplex_group_factory(ambient_dim, target_order)
    )
    fine_discr = Discretization(
        actx, mesh, default_simplex_group_factory(ambient_dim, target_order + 3)
    )

    from meshmode.discretization.connection import make_same_mesh_connection

    conn = make_same_mesh_connection(actx, discr, fine_discr)

    # }}}

    # {{{ pickle

    from h5pyckle import dump, load
    from h5pyckle.interop_meshmode import array_context_for_pickling

    from arraycontext import thaw

    nodes = cast(np.ndarray, thaw(discr.nodes(), actx))

    filename = dirname / "pickle_meshmode.h5"
    with array_context_for_pickling(actx):
        dump(
            {
                "Field": nodes[0],
                "Nodes": nodes,
                "Mesh": mesh,
                "Discretization": discr,
                "Connection": conn,
            },
            filename,
        )

    with array_context_for_pickling(actx):
        data = load(filename)

    x_new = data["Field"]
    nodes_new = data["Nodes"]
    mesh_new = data["Mesh"]
    discr_new = data["Discretization"]
    conn_new = data["Connection"]

    # }}}

    # {{{ check

    # check stored field is the same
    error = rnorm(actx, x_new, nodes[0])
    logger.info("error[scalar]: %.5e", error)
    assert error < 1.0e-15

    # check stored mesh is the same
    assert mesh == mesh_new
    assert mesh == discr_new.mesh

    # check object array is the same
    error = rnorm(actx, nodes_new, nodes)
    logger.info("error[vector]: %.5e", error)
    assert error < 1.0e-15

    # check discretization nodes are the same
    nodes_new = thaw(discr_new.nodes(), actx)
    error = rnorm(actx, nodes_new, nodes)
    logger.info("error[discr]:  %.5e", error)
    assert error < 1.0e-15

    # check connection is the same
    nodes = thaw(conn.from_discr.nodes(), actx)
    nodes_new = thaw(conn_new.from_discr.nodes(), actx)
    error = rnorm(actx, nodes_new, nodes)
    logger.info("error[conns]:  %.5e", error)
    assert error < 1.0e-15

    # }}}


# }}}


# {{{ test_record_pickling


@dataclass(frozen=True)
class TimingRecord:
    name: str
    mean: float
    std: float
    history: np.ndarray


def test_record_pickling() -> None:
    """Tests handling of __getstate__/__setstate__ with a Record."""

    pytest.importorskip("meshmode")

    cr_in = TimingRecord(
        name="run_12857",
        mean=1.0,
        std=0.2,
        history=np.random.rand(256),
    )

    from h5pyckle import dump, load

    filename = dirname / "pickle_record.h5"
    dump(cr_in, filename)
    cr_out: TimingRecord = load(filename)  # type: ignore[assignment]

    assert cr_in.name == cr_out.name
    assert np.array_equal(cr_in.history, cr_out.history)


# }}}


# {{{ test_pickling_cl_scalar


def test_pickling_cl_scalar() -> None:
    pytest.importorskip("meshmode")

    import pyopencl as cl
    from meshmode.array_context import PyOpenCLArrayContext

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

    # {{{

    x = cl.array.to_device(queue, np.array(42))
    arg_in = {"x": x}

    from h5pyckle import dump, load
    from h5pyckle.interop_meshmode import array_context_for_pickling

    filename = dirname / "pickle_cl_scalar.h5"
    with array_context_for_pickling(actx):
        dump(arg_in, filename)
        arg_out = load(filename)

    logger.info("expected: %s", arg_in)
    logger.info("got:      %s", arg_out)
    assert arg_in == arg_out

    # }}}


# }}}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
