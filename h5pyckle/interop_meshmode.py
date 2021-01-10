import os
import pickle
from typing import Optional

import numpy as np

from meshmode.dof_array import DOFArray
from meshmode.mesh import MeshElementGroup, Mesh
from meshmode.discretization import ElementGroupBase, Discretization
from meshmode.discretization.connection import \
        InterpolationBatch, \
        DiscretizationConnectionElementGroup, \
        DirectDiscretizationConnection

from h5pyckle import H5Group, Pickler, dump, loader, load_from_type
import h5pyckle.interop_numpy       # noqa: F401

__all__ = ["ArrayContextPickler", "SameElementGroupFactory"]


# {{{ context manager

class ArrayContextPickler(Pickler):
    def __init__(self, actx,
            filename: os.PathLike, *,
            mode: str = "w",
            h5_file_options: Optional[dict] = None,
            h5_dset_options: Optional[dict] = None):
        super().__init__(filename, mode=mode,
                h5_file_options=h5_file_options,
                h5_dset_options=h5_dset_options)

        self.actx = actx

# }}}


# {{{ dof arrays

@dump.register(DOFArray)
def _(obj: DOFArray, h5: H5Group, *, name: str):
    actx = h5.pickler.actx

    from meshmode.dof_array import thaw, freeze
    if obj.array_context is None:
        obj = thaw(actx, obj)
    else:
        obj = thaw(actx, freeze(obj))

    # TODO: handle complex types

    subgrp = h5.create_group(name)
    dump(type(obj), subgrp)

    for i, ary in enumerate(obj):
        subgrp.create_dataset(f"group_{i:05d}", data=actx.to_numpy(ary))


@loader.register(DOFArray)
def _(h5: H5Group) -> DOFArray:
    actx = h5.pickler.actx

    return DOFArray(actx, tuple([
        actx.from_numpy(h5[name][:]) for name in h5
        ]))

# }}}


# {{{ mesh

@dump.register(MeshElementGroup)
def _(obj: MeshElementGroup, h5: H5Group, *, name: str):
    subgrp = h5.create_group(name)
    dump(type(obj), subgrp)

    subgrp.attrs["order"] = obj.order
    subgrp.attrs["dim"] = obj.dim

    subgrp.create_dataset("vertex_indices", data=obj.vertex_indices)
    subgrp.create_dataset("nodes", data=obj.nodes)
    subgrp.create_dataset("unit_nodes", data=obj.unit_nodes)


@loader.register(MeshElementGroup)
def _(h5: H5Group) -> MeshElementGroup:
    # NOTE: h5py extracts these as np.intp
    order = int(h5.attrs["order"])
    dim = int(h5.attrs["dim"])

    vertex_indices = h5["vertex_indices"][:]
    nodes = h5["nodes"][:]
    unit_nodes = h5["unit_nodes"][:]

    cls = load_from_type(h5, obj_type=type)
    return cls(order, vertex_indices, nodes,
            unit_nodes=unit_nodes,
            dim=dim)


@dump.register(Mesh)
def _(obj: Mesh, h5: H5Group, *, name: str):
    h5 = h5.create_group(name)
    dump(type(obj), h5)

    dump(obj.vertex_id_dtype, h5, name="vertex_id_dtype")
    dump(obj.element_id_dtype, h5, name="element_id_dtype")

    h5.attrs["is_conforming"] = obj.is_conforming
    h5.attrs["boundary_tags"] = np.array(pickle.dumps(obj.boundary_tags))
    h5.create_dataset("vertices", data=obj.vertices)

    subgrp = h5.create_group("groups")
    for i, grp in enumerate(obj.groups):
        dump(grp, subgrp, name=f"group_{i:05d}")

    # TODO
    # FacialAdjacencyGroup
    # NodalAdjacency


@loader.register(Mesh)
def _(h5: H5Group) -> Mesh:
    vertex_id_dtype = load_from_type(h5["vertex_id_dtype"])
    element_id_dtype = load_from_type(h5["element_id_dtype"])

    is_conforming = h5.attrs["is_conforming"]
    boundary_tags = pickle.loads(h5.attrs["boundary_tags"])
    vertices = h5["vertices"][:]
    groups = [load_from_type(h5["groups"][name]) for name in h5["groups"]]

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

class SameElementGroupFactory:
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


@dump.register(ElementGroupBase)
def _(obj: ElementGroupBase, h5: H5Group, *, name: str):
    subgrp = h5.create_group(name)
    dump(type(obj), subgrp)
    subgrp.attrs["order"] = obj.order


@loader.register(ElementGroupBase)
def _(h5: H5Group) -> ElementGroupBase:
    # NOTE: the mesh_el_group and index are set by the group factory
    cls = load_from_type(h5, obj_type=type)
    return cls(None, int(h5.attrs["order"]), -1)


@dump.register(Discretization)
def _(obj: Discretization, h5: H5Group, *, name: str):
    h5grp = h5.create_group(name)

    dump(type(obj), h5grp)
    dump(obj.mesh, h5grp, name="mesh")
    dump(obj.dtype, h5grp, name="real_dtype")

    h5grp = h5grp.create_group("groups")
    for i, grp in enumerate(obj.groups):
        dump(grp, h5grp, name=f"group_{i:05d}")


@loader.register(Discretization)
def _(h5: H5Group) -> Discretization:
    actx = h5.pickler.actx

    mesh = load_from_type(h5["mesh"])
    groups = [load_from_type(h5["groups"][name]) for name in h5["groups"]]
    real_dtype = load_from_type(h5["real_dtype"])

    return Discretization(actx, mesh,
            group_factory=SameElementGroupFactory(groups),
            real_dtype=real_dtype)

# }}}


# {{{ direct connection

@dump.register(InterpolationBatch)
def _(obj: InterpolationBatch, h5: H5Group, *, name: str):
    actx = h5.pickler.actx

    grp = h5.create_group(name)
    dump(type(obj), grp)
    grp.attrs["from_group_index"] = obj.from_group_index
    grp.attrs["to_element_face"] = obj.to_element_face

    grp.create_dataset("from_element_indices",
            data=actx.to_numpy(obj.from_element_indices))
    grp.create_dataset("to_element_indices",
            data=actx.to_numpy(obj.to_element_indices))
    grp.create_dataset("result_unit_nodes", data=obj.result_unit_nodes)


@loader.register(InterpolationBatch)
def _(h5: H5Group) -> InterpolationBatch:
    actx = h5.pickler.actx
    from_group_index = h5.attrs["from_group_index"]
    to_element_face = h5.attrs["to_element_face"]

    from_element_indices = h5["from_element_indices"][:]
    to_element_indices = h5["from_element_indices"][:]
    result_unit_nodes = h5["result_unit_nodes"][:]

    return InterpolationBatch(from_group_index,
            from_element_indices=actx.freeze(actx.from_numpy(from_element_indices)),
            to_element_indices=actx.freeze(actx.from_numpy(to_element_indices)),
            result_unit_nodes=result_unit_nodes,
            to_element_face=to_element_face,
            )


@dump.register(DiscretizationConnectionElementGroup)
def _(obj: DiscretizationConnectionElementGroup, h5: H5Group, *, name: str):
    grp = h5.create_group(name)
    dump(type(obj), grp)

    for i, batch in enumerate(obj.batches):
        dump(batch, grp, name=f"batch_{i:05d}")


@loader.register(DiscretizationConnectionElementGroup)
def _(h5: H5Group) -> DiscretizationConnectionElementGroup:
    batches = [load_from_type(h5[name]) for name in h5]
    return DiscretizationConnectionElementGroup(batches)


@dump.register(DirectDiscretizationConnection)
def _(obj: DirectDiscretizationConnection, h5: H5Group, *, name: str):
    h5grp = h5.create_group(name)
    h5grp.attrs["is_surjective"] = obj.is_surjective

    dump(obj.from_discr, h5grp, name="from_discr")
    dump(obj.to_discr, h5grp, name="to_discr")

    h5grp = h5grp.create_group("groups")
    for i, grp in enumerate(obj.groups):
        dump(grp, h5grp, name=f"group_{i:05d}")


@loader.register(DirectDiscretizationConnection)
def _(h5: H5Group) -> DirectDiscretizationConnection:
    is_surjective = h5.attrs["is_surjective"]

    from_discr = load_from_type(h5["from_discr"])
    to_discr = load_from_type(h5["to_discr"])

    groups = [load_from_type(h5["groups"][name]) for name in h5["groups"]]

    return DirectDiscretizationConnection(from_discr, to_discr, groups,
            is_surjective=is_surjective)

# }}}
