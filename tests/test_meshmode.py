# SPDX-FileCopyrightText: 2020-2022 Alexandru Fikl <alexfikl@gmail.com>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from arraycontext import ArrayContext
    from meshmode.dof_array import DOFArray

logger = logging.getLogger(__name__)
dirname = pathlib.Path(__file__).parent


def norm(actx: ArrayContext, x: np.ndarray[Any, np.dtype[Any]] | DOFArray) -> float:
    from meshmode.dof_array import flat_norm

    if isinstance(x, np.ndarray):
        x = actx.np.sqrt(x @ x)

    return flat_norm(x)


def rnorm(
    actx: ArrayContext,
    x: np.ndarray[Any, np.dtype[Any]] | DOFArray,
    y: np.ndarray[Any, np.dtype[Any]] | DOFArray,
) -> float:
    norm_y = norm(actx, y)
    if norm_y < 1.0e-15:
        norm_y = 1.0

    return actx.to_numpy(norm(actx, x - y) / norm_y)


# {{{ test_discretization_pickling


@pytest.mark.meshmode
@pytest.mark.parametrize("ambient_dim", [2, 3])
@pytest.mark.parametrize("target_order", [3])
def test_discretization_pickling(ambient_dim: int, target_order: int) -> None:
    """Tests that the interop_meshmode types can all be dumped/loaded correctly."""

    pytest.importorskip("meshmode")

    import pyopencl as cl
    from meshmode.array_context import PyOpenCLArrayContext

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

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

    nodes = actx.thaw(discr.nodes())

    from arraycontext.impl.pyopencl.taggable_cl_array import TaggableCLArray

    ary = nodes[0][0]
    assert isinstance(ary, TaggableCLArray)

    from meshmode.transform_metadata import FirstAxisIsElementsTag

    ary = ary.tagged(FirstAxisIsElementsTag())

    filename = dirname / "pickle_meshmode.h5"
    with array_context_for_pickling(actx):
        dump(
            {
                "TaggableCLArray": ary,
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

    ary_new = data["TaggableCLArray"]
    x_new = data["Field"]
    nodes_new = data["Nodes"]
    mesh_new = data["Mesh"]
    discr_new = data["Discretization"]
    conn_new = data["Connection"]

    # }}}

    # {{{ check

    # check tagged array
    assert ary.tags == ary_new.tags
    assert ary.axes == ary_new.axes

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
    nodes_new = actx.thaw(discr_new.nodes())
    error = rnorm(actx, nodes_new, nodes)
    logger.info("error[discr]:  %.5e", error)
    assert error < 1.0e-15

    # check connection is the same
    nodes = actx.thaw(conn.from_discr.nodes())
    nodes_new = actx.thaw(conn_new.from_discr.nodes())
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


@pytest.mark.meshmode
def test_dataclass_pickling() -> None:
    """Tests handling of __getstate__/__setstate__ with a dataclass."""

    pytest.importorskip("meshmode")
    rng = np.random.default_rng(seed=42)

    cr_in = TimingRecord(
        name="run_12857",
        mean=1.0,
        std=0.2,
        history=rng.random(256),
    )

    from h5pyckle import dump, load

    filename = dirname / "pickle_record.h5"
    dump(cr_in, filename)
    cr_out: TimingRecord = load(filename)

    assert cr_in.name == cr_out.name
    assert np.array_equal(cr_in.history, cr_out.history)


# }}}


# {{{ test_pickling_cl_scalar


@pytest.mark.meshmode
def test_pickling_cl_scalar() -> None:
    pytest.importorskip("meshmode")

    import pyopencl as cl
    from meshmode.array_context import PyOpenCLArrayContext

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    # {{{

    import pyopencl.array as cla

    x = cla.to_device(queue, np.array(42))
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
        _ = pytest.main([__file__])

# vim: fdm=marker
