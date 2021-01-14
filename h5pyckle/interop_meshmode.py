import os
import pickle
from typing import Any, Optional

import numpy as np

from meshmode.dof_array import DOFArray
from meshmode.mesh import MeshElementGroup, Mesh
from meshmode.discretization import ElementGroupBase, Discretization
from meshmode.discretization.connection import \
        InterpolationBatch, \
        DiscretizationConnectionElementGroup, \
        DirectDiscretizationConnection

from h5pyckle.base import dumper, loader
from h5pyckle.base import PickleFile, PickleGroup, load_from_type, create_type
import h5pyckle.interop_numpy       # noqa: F401


__all__ = ["ArrayContextPickleFile"]


# {{{ context manager

class ArrayContextPickleFile(PickleFile):
    def __init__(self, actx,
            filename: os.PathLike, *,
            mode: str = "w",
            h5_file_options: Optional[dict] = None,
            h5_dset_options: Optional[dict] = None):
        super().__init__(filename, mode=mode,
                h5_file_options=h5_file_options,
                h5_dset_options=h5_dset_options)

        self.actx = actx


def dump(actx,
        obj: Any,
        filename: os.PathLike, *,
        name: Optional[str] = None):
    from h5pyckle.base import dump_to_group
    with ArrayContextPickleFile(actx, filename, mode="w") as root:
        dump_to_group(obj, root, name=name)


def load(actx, filename: os.PathLike):
    from h5pyckle.base import load_from_group
    with ArrayContextPickleFile(actx, filename, mode="r") as root:
        return load_from_group(root)

# }}}


# {{{ dof arrays

@dumper.register(DOFArray)
def _(obj: DOFArray, pkl: PickleGroup, *, name: Optional[str] = None):
    actx = pkl.context.actx

    from meshmode.dof_array import thaw, freeze
    if obj.array_context is None:
        obj = thaw(actx, obj)
    else:
        obj = thaw(actx, freeze(obj))

    # TODO: handle complex dtypes
    subgrp = create_type(obj, pkl, name=name)
    for i, ary in enumerate(obj):
        subgrp.create_dataset(f"entry_{i}", data=actx.to_numpy(ary))


@loader.register(DOFArray)
def _(pkl: PickleGroup) -> DOFArray:
    actx = pkl.context.actx

    return DOFArray(actx, tuple([
        actx.from_numpy(pkl[name][:]) for name in pkl
        ]))

# }}}


# {{{ mesh

@dumper.register(MeshElementGroup)
def _(obj: MeshElementGroup, pkl: PickleGroup, *, name: Optional[str] = None):
    subgrp = create_type(obj, pkl, name=name)

    subgrp.attrs["order"] = obj.order
    subgrp.attrs["dim"] = obj.dim

    subgrp.create_dataset("vertex_indices", data=obj.vertex_indices)
    subgrp.create_dataset("nodes", data=obj.nodes)
    subgrp.create_dataset("unit_nodes", data=obj.unit_nodes)


@loader.register(MeshElementGroup)
def _(pkl: PickleGroup) -> MeshElementGroup:
    # NOTE: h5py extracts these as np.intp
    order = int(pkl.attrs["order"])
    dim = int(pkl.attrs["dim"])

    vertex_indices = pkl["vertex_indices"][:]
    nodes = pkl["nodes"][:]
    unit_nodes = pkl["unit_nodes"][:]

    cls = load_from_type(pkl, obj_type=type)
    return cls(order, vertex_indices, nodes,
            unit_nodes=unit_nodes,
            dim=dim)


@dumper.register(Mesh)
def _(obj: Mesh, pkl: PickleGroup, *, name: str):
    pkl = pkl.create_group(name)
    dumper(type(obj), pkl)

    dumper(obj.vertex_id_dtype, pkl, name="vertex_id_dtype")
    dumper(obj.element_id_dtype, pkl, name="element_id_dtype")

    pkl.attrs["is_conforming"] = obj.is_conforming
    pkl.attrs["boundary_tags"] = np.void(pickle.dumps(obj.boundary_tags))
    pkl.create_dataset("vertices", data=obj.vertices)

    subgrp = pkl.create_group("groups")
    for i, grp in enumerate(obj.groups):
        dumper(grp, subgrp, name=f"group_{i:05d}")

    # TODO
    # FacialAdjacencyGroup
    # NodalAdjacency


@loader.register(Mesh)
def _(pkl: PickleGroup) -> Mesh:
    vertex_id_dtype = load_from_type(pkl["vertex_id_dtype"])
    element_id_dtype = load_from_type(pkl["element_id_dtype"])

    is_conforming = pkl.attrs["is_conforming"]
    boundary_tags = pickle.loads(pkl.attrs["boundary_tags"].tobytes())
    vertices = pkl["vertices"][:]
    groups = [load_from_type(pkl["groups"][name]) for name in pkl["groups"]]

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
def _(obj: ElementGroupBase, pkl: PickleGroup, *, name: Optional[str] = None):
    subgrp = create_type(obj, pkl, name=name)
    subgrp.attrs["order"] = obj.order


@loader.register(ElementGroupBase)
def _(pkl: PickleGroup) -> ElementGroupBase:
    # NOTE: the mesh_el_group and index are set by the group factory
    cls = load_from_type(pkl, obj_type=type)
    return cls(None, int(pkl.attrs["order"]), -1)


@dumper.register(Discretization)
def _(obj: Discretization, pkl: PickleGroup, *, name: Optional[str] = None):
    subgrp = create_type(obj, pkl, name=name)
    dumper(obj.mesh, subgrp, name="mesh")
    dumper(obj.real_dtype, subgrp, name="real_dtype")

    subgrp = subgrp.create_group("groups")
    for i, grp in enumerate(obj.groups):
        dumper(grp, subgrp, name=f"group_{i}")


@loader.register(Discretization)
def _(pkl: PickleGroup) -> Discretization:
    actx = pkl.context.actx

    mesh = load_from_type(pkl["mesh"])
    real_dtype = load_from_type(pkl["real_dtype"])
    groups = [load_from_type(pkl["groups"][name]) for name in pkl["groups"]]

    return Discretization(actx, mesh,
            group_factory=_SameElementGroupFactory(groups),
            real_dtype=real_dtype)

# }}}


# {{{ direct connection

@dumper.register(InterpolationBatch)
def _(obj: InterpolationBatch, pkl: PickleGroup, *, name: Optional[str] = None):
    actx = pkl.context.actx
    grp = create_type(obj, pkl, name=name)

    grp.attrs["from_group_index"] = obj.from_group_index
    if obj.to_element_face is not None:
        grp.attrs["to_element_face"] = obj.to_element_face

    grp.create_dataset("from_element_indices",
            data=actx.to_numpy(obj.from_element_indices))
    grp.create_dataset("to_element_indices",
            data=actx.to_numpy(obj.to_element_indices))
    grp.create_dataset("result_unit_nodes", data=obj.result_unit_nodes)


@loader.register(InterpolationBatch)
def _(pkl: PickleGroup) -> InterpolationBatch:
    actx = pkl.context.actx

    from_group_index = pkl.attrs["from_group_index"]
    to_element_face = pkl.attrs.get("to_element_face", None)

    from_element_indices = pkl["from_element_indices"][:]
    to_element_indices = pkl["from_element_indices"][:]
    result_unit_nodes = pkl["result_unit_nodes"][:]

    return InterpolationBatch(from_group_index,
            from_element_indices=actx.freeze(actx.from_numpy(from_element_indices)),
            to_element_indices=actx.freeze(actx.from_numpy(to_element_indices)),
            result_unit_nodes=result_unit_nodes,
            to_element_face=to_element_face,
            )


@dumper.register(DiscretizationConnectionElementGroup)
def _(obj: DiscretizationConnectionElementGroup,
        pkl: PickleGroup, *,
        name: Optional[str] = None):
    grp = create_type(obj, pkl, name=name)
    for i, batch in enumerate(obj.batches):
        dumper(batch, grp, name=f"batch_{i}")


@loader.register(DiscretizationConnectionElementGroup)
def _(pkl: PickleGroup) -> DiscretizationConnectionElementGroup:
    batches = [load_from_type(pkl[name]) for name in pkl]
    return DiscretizationConnectionElementGroup(batches)


@dumper.register(DirectDiscretizationConnection)
def _(obj: DirectDiscretizationConnection,
        pkl: PickleGroup, *,
        name: Optional[str] = None):
    h5grp = create_type(obj, pkl, name=name)

    dumper(obj.from_discr, h5grp, name="from_discr")
    dumper(obj.to_discr, h5grp, name="to_discr")
    h5grp.attrs["is_surjective"] = obj.is_surjective

    h5grp = h5grp.create_group("groups")
    for i, grp in enumerate(obj.groups):
        dumper(grp, h5grp, name=f"group_{i}")


@loader.register(DirectDiscretizationConnection)
def _(pkl: PickleGroup) -> DirectDiscretizationConnection:
    is_surjective = pkl.attrs["is_surjective"]

    from_discr = load_from_type(pkl["from_discr"])
    to_discr = load_from_type(pkl["to_discr"])

    groups = [load_from_type(pkl["groups"][name]) for name in pkl["groups"]]

    return DirectDiscretizationConnection(from_discr, to_discr, groups,
            is_surjective=is_surjective)

# }}}
