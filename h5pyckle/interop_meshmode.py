import os
from typing import Any, Optional, Dict

try:
    import dill as pickle
except ImportError:
    import pickle

import h5py
import numpy as np

import pyopencl as cl
import pyopencl.array   # noqa: F401

from meshmode.dof_array import DOFArray
from meshmode.mesh import MeshElementGroup, Mesh
from meshmode.discretization import ElementGroupBase, Discretization
from meshmode.discretization.connection import \
        InterpolationBatch, \
        DiscretizationConnectionElementGroup, \
        DirectDiscretizationConnection

from h5pyckle.base import dumper, loader
from h5pyckle.base import PickleGroup, load_from_type
import h5pyckle.interop_numpy       # noqa: F401


__all__ = ["ArrayContextPickleGroup"]


# {{{ context manager

class ArrayContextPickleGroup(PickleGroup):
    """A :class:`~h5pyckle.PickleGroup` with access to an
    :class:`meshmode.array_context.ArrayContext`.

    .. attribute:: actx
    """

    def __init__(self, actx,
            gid: h5py.h5g.GroupID, *,
            h5_dset_options: Optional[Dict[str, Any]] = None):
        super().__init__(gid, h5_dset_options=h5_dset_options)
        self.actx = actx

    def replace(self, **kwargs):
        kwargs["actx"] = kwargs.get("actx", self.actx)
        kwargs["gid"] = kwargs.get("gid", self.id)
        kwargs["h5_dset_options"] = kwargs.get(
                "h5_dset_options", self.h5_dset_options)

        return type(self)(**kwargs)


def dump(actx, obj: Any, filename: os.PathLike, *,
        mode: str = "w",
        h5_file_options: Optional[Dict[str, Any]] = None,
        h5_dset_options: Optional[Dict[str, Any]] = None):
    """This function should be used when the object hierarchy contains
    :mod:`meshmode` objects that require an
    :class:`~meshmode.array_context.ArrayContext`.

    :param actx: :class:`~meshmode.array_context.ArrayContext` used for pickling.
    """
    if h5_file_options is None:
        h5_file_options = {}

    if h5_dset_options is None:
        h5_dset_options = {}

    from h5pyckle.base import dump_to_group
    with h5py.File(filename, mode=mode, **h5_file_options) as h5:
        root = ArrayContextPickleGroup(actx, h5["/"].id,
                h5_dset_options=h5_dset_options)
        dump_to_group(obj, root)


def load(actx, filename: os.PathLike):
    """This function should be used when the object hierarchy contains
    :mod:`meshmode` objects that require an
    :class:`~meshmode.array_context.ArrayContext`.

    :param actx: :class:`~meshmode.array_context.ArrayContext` used for unpickling.
    """

    from h5pyckle.base import load_from_group
    with h5py.File(filename, mode="r") as h5:
        return load_from_group(ArrayContextPickleGroup(actx, h5["/"].id))

# }}}


# {{{ pyopencl.Array

@dumper.register(cl.array.Array)
def _(obj: cl.array.Array,
        parent: ArrayContextPickleGroup, *,
        name: Optional[str] = None):
    group = parent.create_type(name, obj)

    group.attrs["frozen"] = obj.queue is None
    group.create_dataset("entry", data=_to_numpy(parent.actx, obj))


@loader.register(cl.array.Array)
def _(parent: ArrayContextPickleGroup) -> cl.array.Array:
    ary = _from_numpy(parent.actx, parent["entry"][:])
    if not parent.attrs["frozen"]:
        ary = ary.with_queue(parent.actx.queue)

    return ary

# }}}


# {{{ dof arrays

def _to_numpy(actx, x):
    return actx.to_numpy(actx.thaw(x))


def _from_numpy(actx, x):
    return actx.freeze(actx.from_numpy(x))


@dumper.register(DOFArray)
def _(obj: DOFArray,
        parent: ArrayContextPickleGroup, *,
        name: Optional[str] = None):
    group = parent.create_type(name, obj)
    group.attrs["frozen"] = obj.array_context is None

    dumper([
        _to_numpy(parent.actx, x) for x in obj
        ], group, name="entries")


@loader.register(DOFArray)
def _(parent: ArrayContextPickleGroup) -> DOFArray:
    entries = load_from_type(parent["entries"])

    if parent.attrs["frozen"]:
        from functools import partial
        array_context = None
        from_numpy = partial(_from_numpy, parent.actx)
    else:
        array_context = parent.actx
        from_numpy = parent.actx.from_numpy

    return parent.pycls(array_context, tuple(from_numpy(x) for x in entries))

# }}}


# {{{ mesh

@dumper.register(MeshElementGroup)
def _(obj: MeshElementGroup,
        parent: ArrayContextPickleGroup, *,
        name: Optional[str] = None):
    subgrp = parent.create_type(name, obj)

    subgrp.attrs["order"] = obj.order
    subgrp.attrs["dim"] = obj.dim

    if obj.vertex_indices is not None:
        subgrp.create_dataset("vertex_indices", data=obj.vertex_indices)
    subgrp.create_dataset("nodes", data=obj.nodes)
    subgrp.create_dataset("unit_nodes", data=obj.unit_nodes)


@loader.register(MeshElementGroup)
def _(parent: ArrayContextPickleGroup) -> MeshElementGroup:
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
def _(obj: Mesh,
        parent: ArrayContextPickleGroup, *,
        name: Optional[str] = None):
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
def _(parent: ArrayContextPickleGroup) -> Mesh:
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
            is_conforming=is_conforming)

# }}}


# {{{ discretization

class _SameElementGroupFactory:
    """Recreates the given groups for a new mesh.

    .. automethod:: __init__
    """

    def __init__(self, groups):
        """
        :arg groups: a :class:`list` of
            :class:`~meshmode.discretization.ElementGroupBase`.
        """

        self.groups = groups

    def __call__(self, mesh_el_group, index):
        if not 0 <= index < len(self.groups):
            raise ValueError("'group_index' outside known range of groups")

        grp = self.groups[index]

        from meshmode.discretization.poly_element import (
            PolynomialRecursiveNodesElementGroup,
        )

        if isinstance(grp, PolynomialRecursiveNodesElementGroup):
            return type(grp)(mesh_el_group, grp.order, grp.family, index)

        return type(grp)(mesh_el_group, grp.order, index)


@dumper.register(ElementGroupBase)
def _(obj: ElementGroupBase,
        parent: ArrayContextPickleGroup, *,
        name: Optional[str] = None):
    # NOTE: these are dumped only for use in Discretization at the moment.
    # There we don't really need to dump mesh_el_group again
    group = parent.create_type(name, obj)
    group.attrs["order"] = obj.order
    group.attrs["index"] = obj.index
    group.attrs["dim"] = obj.dim


@loader.register(ElementGroupBase)
def _(parent: ArrayContextPickleGroup) -> ElementGroupBase:
    # NOTE: the real mesh_el_group is set by the group factory
    from collections import namedtuple
    ElementGroup = namedtuple("ElementGroup", ["dim"])

    return parent.pycls(
            ElementGroup(dim=int(parent.attrs["dim"])),
            int(parent.attrs["order"]),
            int(parent.attrs["index"]))


@dumper.register(Discretization)
def _(obj: Discretization,
        parent: ArrayContextPickleGroup, *,
        name: Optional[str] = None):
    group = parent.create_type(name, obj)

    dumper(obj.mesh, group, name="mesh")
    dumper(obj.real_dtype, group, name="real_dtype")
    dumper(obj.groups, group, name="groups")


@loader.register(Discretization)
def _(parent: ArrayContextPickleGroup) -> Discretization:
    actx = parent.actx

    mesh = load_from_type(parent["mesh"])
    real_dtype = load_from_type(parent["real_dtype"])
    groups = load_from_type(parent["groups"])

    return parent.pycls(actx, mesh,
            group_factory=_SameElementGroupFactory(groups),
            real_dtype=real_dtype)

# }}}


# {{{ direct connection

@dumper.register(InterpolationBatch)
def _(obj: InterpolationBatch,
        parent: ArrayContextPickleGroup, *,
        name: Optional[str] = None):
    actx = parent.actx
    grp = parent.create_type(name, obj)

    grp.attrs["from_group_index"] = obj.from_group_index
    if obj.to_element_face is not None:
        grp.attrs["to_element_face"] = obj.to_element_face

    grp.create_dataset("from_element_indices",
            data=_to_numpy(actx, obj.from_element_indices))
    grp.create_dataset("to_element_indices",
            data=_to_numpy(actx, obj.to_element_indices))
    grp.create_dataset("result_unit_nodes", data=obj.result_unit_nodes)


@loader.register(InterpolationBatch)
def _(parent: ArrayContextPickleGroup) -> InterpolationBatch:
    actx = parent.actx

    from_group_index = parent.attrs["from_group_index"]
    to_element_face = parent.attrs.get("to_element_face", None)

    from_element_indices = parent["from_element_indices"][:]
    to_element_indices = parent["from_element_indices"][:]
    result_unit_nodes = parent["result_unit_nodes"][:]

    return parent.pycls(from_group_index,
            from_element_indices=_from_numpy(actx, from_element_indices),
            to_element_indices=_from_numpy(actx, to_element_indices),
            result_unit_nodes=result_unit_nodes,
            to_element_face=to_element_face,
            )


@dumper.register(DiscretizationConnectionElementGroup)
def _(obj: DiscretizationConnectionElementGroup,
        parent: ArrayContextPickleGroup, *,
        name: Optional[str] = None):
    group = parent.create_type(name, obj)
    dumper(obj.batches, group, name="batches")


@loader.register(DiscretizationConnectionElementGroup)
def _(parent: ArrayContextPickleGroup) -> DiscretizationConnectionElementGroup:
    batches = load_from_type(parent["batches"])

    return parent.pycls(batches)


@dumper.register(DirectDiscretizationConnection)
def _(obj: DirectDiscretizationConnection,
        parent: ArrayContextPickleGroup, *,
        name: Optional[str] = None):
    group = parent.create_type(name, obj)

    group.attrs["is_surjective"] = obj.is_surjective
    dumper(obj.from_discr, group, name="from_discr")
    dumper(obj.to_discr, group, name="to_discr")
    dumper(obj.groups, group, name="groups")


@loader.register(DirectDiscretizationConnection)
def _(parent: ArrayContextPickleGroup) -> DirectDiscretizationConnection:
    is_surjective = parent.attrs["is_surjective"]
    from_discr = load_from_type(parent["from_discr"])
    to_discr = load_from_type(parent["to_discr"])
    groups = load_from_type(parent["groups"])

    return parent.pycls(from_discr, to_discr, groups, is_surjective=is_surjective)

# }}}
