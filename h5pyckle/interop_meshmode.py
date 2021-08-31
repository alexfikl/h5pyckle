try:
    import dill as pickle
except ImportError:
    # https://github.com/python/mypy/issues/1153
    import pickle       # type: ignore[no-redef]

import sys
import threading
from contextlib import contextmanager
from typing import Iterator, List, Optional

import numpy as np

import pyopencl as cl
import pyopencl.array   # noqa: F401

from arraycontext import ArrayContext
from meshmode.dof_array import DOFArray
from meshmode.mesh import MeshElementGroup, Mesh
from meshmode.discretization import ElementGroupBase, Discretization
from meshmode.discretization.poly_element import PolynomialRecursiveNodesElementGroup
from meshmode.discretization.connection import \
        InterpolationBatch, \
        DiscretizationConnectionElementGroup, \
        DirectDiscretizationConnection

from h5pyckle.base import dumper, loader
from h5pyckle.base import PickleGroup, load_from_type
import h5pyckle.interop_numpy       # noqa: F401


__all__ = ["array_context_for_pickling"]

if getattr(sys, "H5PYCKLE_BUILDING_SPHINX_DOCS", False):
    # FIXME: without this mocking pyopencl or meshmode classes would result
    # in some sort of infinite loop trying to unwrap the registered class,
    # which had a `__wrapped__` attribute for some reason.

    def register(cls):
        def wrapper(func):
            return func

        return wrapper

    dumper.register = register      # type: ignore[assignment]
    loader.register = register      # type: ignore[assignment]


# {{{ context manager

_ARRAY_CONTEXT_FOR_PICKLING_TLS = threading.local()


@contextmanager
def array_context_for_pickling(actx: ArrayContext) -> Iterator[None]:
    """A context manager that can be used to provide an
    :class:`~arraycontext.ArrayContext` for pickling and unpickling
    :mod:`meshmode` objects.
    """
    try:
        existing_pickle_actx = _ARRAY_CONTEXT_FOR_PICKLING_TLS.actx
    except AttributeError:
        existing_pickle_actx = None

    if existing_pickle_actx is not None:
        raise RuntimeError("'array_context_for_pickling' can not be nested.")

    _ARRAY_CONTEXT_FOR_PICKLING_TLS.actx = actx
    try:
        yield
    finally:
        _ARRAY_CONTEXT_FOR_PICKLING_TLS.actx = None


def get_array_context() -> ArrayContext:
    try:
        actx: Optional[ArrayContext] = _ARRAY_CONTEXT_FOR_PICKLING_TLS.actx
    except AttributeError:
        actx = None

    if actx is None:
        raise RuntimeError("pickling or unpickling 'meshmode' objects "
                "requires an 'ArrayContext'. Use the 'array_context_for_pickling' "
                "context manager to provide one.")

    return actx


def to_numpy(x: Optional[cl.array.Array]) -> Optional[np.ndarray]:
    if x is None:
        return x

    actx = get_array_context()
    return actx.to_numpy(actx.thaw(x))      # type: ignore


def from_numpy(
        x: Optional[np.ndarray],
        freeze: bool = True) -> Optional[cl.array.Array]:
    if x is None:
        return x

    actx = get_array_context()
    x = actx.from_numpy(x)
    if freeze:
        x = actx.freeze(x)                  # type: ignore

    return x

# }}}


# {{{ pyopencl.Array

@dumper.register(cl.array.Array)
def _dump_cl_array(obj: cl.array.Array,
        parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
    group = parent.create_type(name, obj)

    group.attrs["frozen"] = obj.queue is None
    group.create_dataset("entry", data=to_numpy(obj))


@loader.register(cl.array.Array)
def _load_cl_array(parent: PickleGroup) -> cl.array.Array:
    from h5pyckle.interop_numpy import load_numpy_dataset
    return from_numpy(load_numpy_dataset(parent, "entry"), parent.attrs["frozen"])

# }}}


# {{{ dof arrays

@dumper.register(DOFArray)
def _dump_dof_array(
        obj: DOFArray, parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
    group = parent.create_type(name, obj)
    group.attrs["frozen"] = obj.array_context is None

    dumper([to_numpy(x) for x in obj], group, name="entries")


@loader.register(DOFArray)
def _load_dof_array(parent: PickleGroup) -> DOFArray:
    entries = load_from_type(parent["entries"])

    array_context = None if parent.attrs["frozen"] else get_array_context()
    return parent.pycls(array_context, tuple([  # pylint: disable=R1728
        from_numpy(x, parent.attrs["frozen"]) for x in entries
        ]))

# }}}


# {{{ mesh

@dumper.register(MeshElementGroup)
def _dump_mesh_element_grouo(
        obj: MeshElementGroup, parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
    subgrp = parent.create_type(name, obj)

    subgrp.attrs["order"] = obj.order
    subgrp.attrs["dim"] = obj.dim

    if obj.vertex_indices is not None:
        subgrp.create_dataset("vertex_indices", data=obj.vertex_indices)
    subgrp.create_dataset("nodes", data=obj.nodes)
    subgrp.create_dataset("unit_nodes", data=obj.unit_nodes)


@loader.register(MeshElementGroup)
def _load_mesh_element_group(parent: PickleGroup) -> MeshElementGroup:
    # NOTE: h5py extracts these as np.intp
    order = int(parent.attrs["order"])
    dim = int(parent.attrs["dim"])

    if "vertex_indices" in parent:
        vertex_indices = parent["vertex_indices"][:]
    else:
        vertex_indices = None
    nodes = parent["nodes"][:]
    unit_nodes = parent["unit_nodes"][:]

    return parent.pycls(order, vertex_indices, nodes,
            unit_nodes=unit_nodes,
            dim=dim)


@dumper.register(Mesh)
def _dump_mesh(
        obj: Mesh, parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
    parent = parent.create_type(name, obj)

    parent.attrs["boundary_tags"] = np.void(pickle.dumps(obj.boundary_tags))
    if obj.is_conforming is not None:
        parent.attrs["is_conforming"] = obj.is_conforming

    if obj.vertices is not None:
        parent.create_dataset("vertices", data=obj.vertices)

    dumper(obj.vertex_id_dtype, parent, name="vertex_id_dtype")
    dumper(obj.element_id_dtype, parent, name="element_id_dtype")
    dumper(obj.groups, parent, name="groups")

    # TODO
    # FacialAdjacencyGroup
    # NodalAdjacency


@loader.register(Mesh)
def _load_mesh(parent: PickleGroup) -> Mesh:
    is_conforming = parent.attrs.get("is_conforming", None)
    boundary_tags = pickle.loads(parent.attrs["boundary_tags"].tobytes())
    if "vertices" in parent:
        vertices = parent["vertices"][:]
    else:
        vertices = None

    vertex_id_dtype = load_from_type(parent["vertex_id_dtype"])
    element_id_dtype = load_from_type(parent["element_id_dtype"])
    groups = load_from_type(parent["groups"])

    # TODO
    # FacialAdjacencyGroup
    # NodalAdjacency

    return parent.pycls(vertices, groups,
            boundary_tags=boundary_tags,
            vertex_id_dtype=vertex_id_dtype,
            element_id_dtype=element_id_dtype,
            is_conforming=is_conforming,
            skip_tests=True)

# }}}


# {{{ discretization

class _SameElementGroupFactory:
    """Recreates the given groups for a new mesh.

    .. automethod:: __init__
    """

    def __init__(self, groups: List[ElementGroupBase]) -> None:
        """
        :arg groups: a :class:`list` of
            :class:`~meshmode.discretization.ElementGroupBase`.
        """

        self.groups = groups

    def __call__(self,
            mesh_el_group: MeshElementGroup,
            index: int) -> ElementGroupBase:
        if not 0 <= index < len(self.groups):
            raise ValueError("'group_index' outside known range of groups")

        grp = self.groups[index]

        if isinstance(grp, PolynomialRecursiveNodesElementGroup):
            return type(grp)(mesh_el_group, grp.order, grp.family, index)

        return type(grp)(mesh_el_group, grp.order, index)


@dumper.register(ElementGroupBase)
def _dump_element_group(
        obj: ElementGroupBase, parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
    # NOTE: these are dumped only for use in Discretization at the moment.
    # There we don't really need to dump mesh_el_group again
    group = parent.create_type(name, obj)
    group.attrs["order"] = obj.order
    group.attrs["index"] = obj.index
    group.attrs["dim"] = obj.dim


@loader.register(ElementGroupBase)
def _load_element_group(parent: PickleGroup) -> ElementGroupBase:
    # NOTE: the real mesh_el_group is set by the group factory
    from collections import namedtuple
    ElementGroup = namedtuple("ElementGroup", ["dim"])

    return parent.pycls(
            ElementGroup(dim=int(parent.attrs["dim"])),
            int(parent.attrs["order"]),
            int(parent.attrs["index"]))


@dumper.register(PolynomialRecursiveNodesElementGroup)
def _dump_recursivenodes_element_group(
        obj: PolynomialRecursiveNodesElementGroup, parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
    group = parent.create_type(name, obj)
    group.attrs["order"] = obj.order
    group.attrs["index"] = obj.index
    group.attrs["dim"] = obj.dim
    group.attrs["family"] = obj.family


@loader.register(PolynomialRecursiveNodesElementGroup)
def _load_recursivenodes_element_group(
        parent: PickleGroup) -> PolynomialRecursiveNodesElementGroup:
    # NOTE: the real mesh_el_group is set by the group factory
    from collections import namedtuple
    ElementGroup = namedtuple("ElementGroup", ["dim"])

    return parent.pycls(
            ElementGroup(dim=int(parent.attrs["dim"])),
            int(parent.attrs["order"]),
            str(parent.attrs["family"]),
            int(parent.attrs["index"]))


@dumper.register(Discretization)
def _dump_discretization(
        obj: Discretization, parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
    group = parent.create_type(name, obj)

    dumper(obj.mesh, group, name="mesh")
    dumper(obj.real_dtype, group, name="real_dtype")
    dumper(obj.groups, group, name="groups")


@loader.register(Discretization)
def _load_discretization(parent: PickleGroup) -> Discretization:
    mesh = load_from_type(parent["mesh"])
    real_dtype = load_from_type(parent["real_dtype"])
    groups = load_from_type(parent["groups"])

    actx = get_array_context()
    return parent.pycls(actx, mesh,
            group_factory=_SameElementGroupFactory(groups),
            real_dtype=real_dtype)

# }}}


# {{{ direct connection

@dumper.register(InterpolationBatch)
def _dump_interpolation_batch(
        obj: InterpolationBatch, parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
    grp = parent.create_type(name, obj)

    grp.attrs["from_group_index"] = obj.from_group_index
    if obj.to_element_face is not None:
        grp.attrs["to_element_face"] = obj.to_element_face

    grp.create_dataset("from_element_indices",
            data=to_numpy(obj.from_element_indices))
    grp.create_dataset("to_element_indices",
            data=to_numpy(obj.to_element_indices))
    grp.create_dataset("result_unit_nodes", data=obj.result_unit_nodes)


@loader.register(InterpolationBatch)
def _load_interpolation_batch(parent: PickleGroup) -> InterpolationBatch:
    from_group_index = parent.attrs["from_group_index"]
    to_element_face = parent.attrs.get("to_element_face", None)

    from_element_indices = parent["from_element_indices"][:]
    to_element_indices = parent["from_element_indices"][:]
    result_unit_nodes = parent["result_unit_nodes"][:]

    return parent.pycls(from_group_index,
            from_element_indices=from_numpy(from_element_indices),
            to_element_indices=from_numpy(to_element_indices),
            result_unit_nodes=result_unit_nodes,
            to_element_face=to_element_face,
            )


@dumper.register(DiscretizationConnectionElementGroup)
def _dump_connection_element_group(
        obj: DiscretizationConnectionElementGroup, parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
    group = parent.create_type(name, obj)
    dumper(obj.batches, group, name="batches")


@loader.register(DiscretizationConnectionElementGroup)
def _load_connection_element_group(
        parent: PickleGroup) -> DiscretizationConnectionElementGroup:
    batches = load_from_type(parent["batches"])

    return parent.pycls(batches)


@dumper.register(DirectDiscretizationConnection)
def _dump_direct_connection(
        obj: DirectDiscretizationConnection, parent: PickleGroup, *,
        name: Optional[str] = None) -> None:
    group = parent.create_type(name, obj)

    group.attrs["is_surjective"] = obj.is_surjective
    dumper(obj.from_discr, group, name="from_discr")
    dumper(obj.to_discr, group, name="to_discr")
    dumper(obj.groups, group, name="groups")


@loader.register(DirectDiscretizationConnection)
def _load_direct_connection(parent: PickleGroup) -> DirectDiscretizationConnection:
    is_surjective = parent.attrs["is_surjective"]
    from_discr = load_from_type(parent["from_discr"])
    to_discr = load_from_type(parent["to_discr"])
    groups = load_from_type(parent["groups"])

    return parent.pycls(from_discr, to_discr, groups, is_surjective=is_surjective)

# }}}
