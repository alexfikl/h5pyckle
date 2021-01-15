import os
import pickle
from typing import Any, Optional, Dict

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
    """
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
    from h5pyckle.base import load_from_group
    with h5py.File(filename, mode="r") as h5:
        return load_from_group(ArrayContextPickleGroup(actx, h5["/"].id))

# }}}


# {{{ dof arrays

@dumper.register(DOFArray)
def _(obj: DOFArray, parent: PickleGroup, *, name: Optional[str] = None):
    actx = parent.actx

    from meshmode.dof_array import thaw, freeze
    if obj.array_context is None:
        obj = thaw(actx, obj)
    else:
        obj = thaw(actx, freeze(obj))

    # TODO: handle complex dtypes
    subgrp = parent.create_type(name, obj)
    for i, ary in enumerate(obj):
        subgrp.create_dataset(f"entry_{i}", data=actx.to_numpy(ary))


@loader.register(DOFArray)
def _(parent: PickleGroup) -> DOFArray:
    actx = parent.actx

    return DOFArray(actx, tuple([
        actx.from_numpy(parent[name][:]) for name in parent
        ]))

# }}}


# {{{ mesh

@dumper.register(MeshElementGroup)
def _(obj: MeshElementGroup, parent: PickleGroup, *, name: Optional[str] = None):
    subgrp = parent.create_type(name, obj)

    subgrp.attrs["order"] = obj.order
    subgrp.attrs["dim"] = obj.dim

    subgrp.create_dataset("vertex_indices", data=obj.vertex_indices)
    subgrp.create_dataset("nodes", data=obj.nodes)
    subgrp.create_dataset("unit_nodes", data=obj.unit_nodes)


@loader.register(MeshElementGroup)
def _(parent: PickleGroup) -> MeshElementGroup:
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
def _(obj: Mesh, parent: PickleGroup, *, name: Optional[str] = None):
    parent = parent.create_type(name, obj)

    dumper(obj.vertex_id_dtype, parent, name="vertex_id_dtype")
    dumper(obj.element_id_dtype, parent, name="element_id_dtype")

    parent.attrs["is_conforming"] = obj.is_conforming
    parent.attrs["boundary_tags"] = np.void(pickle.dumps(obj.boundary_tags))
    parent.create_dataset("vertices", data=obj.vertices)

    subgrp = parent.create_group("groups")
    for i, grp in enumerate(obj.groups):
        dumper(grp, subgrp, name=f"group_{i:05d}")

    # TODO
    # FacialAdjacencyGroup
    # NodalAdjacency


@loader.register(Mesh)
def _(parent: PickleGroup) -> Mesh:
    vertex_id_dtype = load_from_type(parent["vertex_id_dtype"])
    element_id_dtype = load_from_type(parent["element_id_dtype"])

    is_conforming = parent.attrs["is_conforming"]
    boundary_tags = pickle.loads(parent.attrs["boundary_tags"].tobytes())
    vertices = parent["vertices"][:]
    groups = [load_from_type(parent["groups"][name]) for name in parent["groups"]]

    # TODO
    # FacialAdjacencyGroup
    # NodalAdjacency

    return Mesh(vertices, groups,
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
def _(obj: ElementGroupBase, parent: PickleGroup, *, name: Optional[str] = None):
    subgrp = parent.create_type(name, obj)
    subgrp.attrs["order"] = obj.order


@loader.register(ElementGroupBase)
def _(parent: PickleGroup) -> ElementGroupBase:
    # NOTE: the mesh_el_group and index are set by the group factory
    cls = parent.type
    return cls(None, int(parent.attrs["order"]), -1)


@dumper.register(Discretization)
def _(obj: Discretization, parent: PickleGroup, *, name: Optional[str] = None):
    subgrp = parent.create_type(name, obj)
    dumper(obj.mesh, subgrp, name="mesh")
    dumper(obj.real_dtype, subgrp, name="real_dtype")

    subgrp = subgrp.create_group("groups")
    for i, grp in enumerate(obj.groups):
        dumper(grp, subgrp, name=f"group_{i}")


@loader.register(Discretization)
def _(parent: PickleGroup) -> Discretization:
    actx = parent.actx

    mesh = load_from_type(parent["mesh"])
    real_dtype = load_from_type(parent["real_dtype"])
    groups = [load_from_type(parent["groups"][name]) for name in parent["groups"]]

    return Discretization(actx, mesh,
            group_factory=_SameElementGroupFactory(groups),
            real_dtype=real_dtype)

# }}}


# {{{ direct connection

@dumper.register(InterpolationBatch)
def _(obj: InterpolationBatch, parent: PickleGroup, *, name: Optional[str] = None):
    actx = parent.actx
    grp = parent.create_type(name, obj)

    grp.attrs["from_group_index"] = obj.from_group_index
    if obj.to_element_face is not None:
        grp.attrs["to_element_face"] = obj.to_element_face

    grp.create_dataset("from_element_indices",
            data=actx.to_numpy(obj.from_element_indices))
    grp.create_dataset("to_element_indices",
            data=actx.to_numpy(obj.to_element_indices))
    grp.create_dataset("result_unit_nodes", data=obj.result_unit_nodes)


@loader.register(InterpolationBatch)
def _(parent: PickleGroup) -> InterpolationBatch:
    actx = parent.actx

    from_group_index = parent.attrs["from_group_index"]
    to_element_face = parent.attrs.get("to_element_face", None)

    from_element_indices = parent["from_element_indices"][:]
    to_element_indices = parent["from_element_indices"][:]
    result_unit_nodes = parent["result_unit_nodes"][:]

    return InterpolationBatch(from_group_index,
            from_element_indices=actx.freeze(actx.from_numpy(from_element_indices)),
            to_element_indices=actx.freeze(actx.from_numpy(to_element_indices)),
            result_unit_nodes=result_unit_nodes,
            to_element_face=to_element_face,
            )


@dumper.register(DiscretizationConnectionElementGroup)
def _(obj: DiscretizationConnectionElementGroup,
        parent: PickleGroup, *,
        name: Optional[str] = None):
    grp = parent.create_type(name, obj)
    for i, batch in enumerate(obj.batches):
        dumper(batch, grp, name=f"batch_{i}")


@loader.register(DiscretizationConnectionElementGroup)
def _(parent: PickleGroup) -> DiscretizationConnectionElementGroup:
    batches = [load_from_type(parent[name]) for name in parent]
    return DiscretizationConnectionElementGroup(batches)


@dumper.register(DirectDiscretizationConnection)
def _(obj: DirectDiscretizationConnection,
        parent: PickleGroup, *,
        name: Optional[str] = None):
    h5grp = parent.create_type(name, obj)

    dumper(obj.from_discr, h5grp, name="from_discr")
    dumper(obj.to_discr, h5grp, name="to_discr")
    h5grp.attrs["is_surjective"] = obj.is_surjective

    h5grp = h5grp.create_group("groups")
    for i, grp in enumerate(obj.groups):
        dumper(grp, h5grp, name=f"group_{i}")


@loader.register(DirectDiscretizationConnection)
def _(parent: PickleGroup) -> DirectDiscretizationConnection:
    is_surjective = parent.attrs["is_surjective"]

    from_discr = load_from_type(parent["from_discr"])
    to_discr = load_from_type(parent["to_discr"])

    groups = [load_from_type(parent["groups"][name]) for name in parent["groups"]]

    return DirectDiscretizationConnection(from_discr, to_discr, groups,
            is_surjective=is_surjective)

# }}}
