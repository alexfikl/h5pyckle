"""
:mod:`meshmode` is a library used to represent and work with high-order
unstructured meshes. It contains a lot of non-trivial types.

Currently, the following types are supported

* :class:`meshmode.dof_array.DOFArray` of any underlying type,
* :class:`meshmode.mesh.MeshElementGroup` and its subclasses,
* :class:`meshmode.mesh.Mesh`,
* :class:`meshmode.discretization.Discretization`,
* :class:`meshmode.discretization.connection.DirectDiscretizationConnection`.

The array type in :mod:`meshmode` is handled by an
:class:`~meshmode.array_context.ArrayContext` and cannot be stored directly
(as it could be on a GPU device). When pickling objects of the types above,
the :mod:`meshmode`-specific :func:`dump` and :func:`load` should be used.

.. autoclass:: ArrayContextPickleGroup
    :no-show-inheritance:

.. autofunction:: dump
.. autofunction:: load
"""

import os
from typing import Any, Optional, Dict

try:
    import dill as pickle
except ImportError:
    import pickle

import h5py
import numpy as np

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

    cls = parent.type
    return cls(array_context, tuple([from_numpy(x) for x in entries]))

# }}}


# {{{ mesh

@dumper.register(MeshElementGroup)
def _(obj: MeshElementGroup,
        parent: ArrayContextPickleGroup, *,
        name: Optional[str] = None):
    subgrp = parent.create_type(name, obj)

    subgrp.attrs["order"] = obj.order
    subgrp.attrs["dim"] = obj.dim

    subgrp.create_dataset("vertex_indices", data=obj.vertex_indices)
    subgrp.create_dataset("nodes", data=obj.nodes)
    subgrp.create_dataset("unit_nodes", data=obj.unit_nodes)


@loader.register(MeshElementGroup)
def _(parent: ArrayContextPickleGroup) -> MeshElementGroup:
    # NOTE: h5py extracts these as np.intp
    order = int(parent.attrs["order"])
    dim = int(parent.attrs["dim"])

    vertex_indices = parent["vertex_indices"][:]
    nodes = parent["nodes"][:]
    unit_nodes = parent["unit_nodes"][:]

    cls = parent.type
    return cls(order, vertex_indices, nodes,
            unit_nodes=unit_nodes,
            dim=dim)


@dumper.register(Mesh)
def _(obj: Mesh,
        parent: ArrayContextPickleGroup, *,
        name: Optional[str] = None):
    parent = parent.create_type(name, obj)

    parent.attrs["is_conforming"] = obj.is_conforming
    parent.attrs["boundary_tags"] = np.void(pickle.dumps(obj.boundary_tags))
    parent.create_dataset("vertices", data=obj.vertices)

    dumper(obj.vertex_id_dtype, parent, name="vertex_id_dtype")
    dumper(obj.element_id_dtype, parent, name="element_id_dtype")
    dumper(obj.groups, parent, name="groups")

    # TODO
    # FacialAdjacencyGroup
    # NodalAdjacency


@loader.register(Mesh)
def _(parent: ArrayContextPickleGroup) -> Mesh:
    is_conforming = parent.attrs["is_conforming"]
    boundary_tags = pickle.loads(parent.attrs["boundary_tags"].tobytes())
    vertices = parent["vertices"][:]

    vertex_id_dtype = load_from_type(parent["vertex_id_dtype"])
    element_id_dtype = load_from_type(parent["element_id_dtype"])
    groups = load_from_type(parent["groups"])

    # TODO
    # FacialAdjacencyGroup
    # NodalAdjacency

    cls = parent.type
    return cls(vertices, groups,
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


@loader.register(ElementGroupBase)
def _(parent: ArrayContextPickleGroup) -> ElementGroupBase:
    # NOTE: the mesh_el_group and index are set by the group factory
    cls = parent.type
    return cls(None,
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

    cls = parent.type
    return cls(actx, mesh,
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

    cls = parent.type
    return cls(from_group_index,
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

    cls = parent.type
    return cls(batches)


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

    cls = parent.type
    return cls(from_discr, to_discr, groups, is_surjective=is_surjective)

# }}}
