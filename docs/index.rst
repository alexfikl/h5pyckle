.. h5pyckle documentation master file, created by
   sphinx-quickstart on Sun Jan 10 11:30:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to h5pyckle's documentation!
====================================

.. automodule:: h5pyckle

Example
=======

The default interface to :mod:`h5pyckle` is the same as for the
:mod:`pickle` module from the standard library. Arbitrary objects can be
dumped and loaded from HDF5 files using

.. literalinclude:: ../examples/basic.py
    :lines: 9-21
    :language: python
    :linenos:

Data can also be stored and loaded directly from subgroups in HD5 files.

.. literalinclude:: ../examples/basic.py
    :lines: 30-43
    :language: python
    :linenos:

The library piggybacks on the :mod:`pickle` module and looks for the usual
:meth:`object.__getstate__`, :meth:`object.__setstate__`,
:meth:`object.__reduce__`, etc. When no alternative is found, it falls back
to :func:`pickle.dumps` / :func:`pickle.loads` and stores the result in a
group attribute or dataset (depending on the size).

To register custom methods for new types, use the :func:`functools.singledispatch`
method as

.. literalinclude:: ../examples/basic.py
    :lines: 49-83
    :language: python
    :linenos:

These examples can be found in :download:`examples/basic.py <../examples/basic.py>`.

Reference
=========

.. automodule:: h5pyckle.base

Numpy
-----

In general, ``h5py`` has very good interopability with :mod:`numpy`. This
module offers some basic handling of :mod:`numpy` types

* :class:`~numpy.dtype`
* :class:`~numpy.ndarray` of all the supported types. Object arrays are also
  supported and are stored largely the same as a Python :class:`list`.

This is very much a work in progress and additional support is encouraged.

.. automodule:: h5pyckle.interop_numpy

Meshmode
--------

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
the :mod:`meshmode`-specific :func:`~h5pyckle.interop_meshmode.dump` and
:func:`~h5pyckle.interop_meshmode.load` should be used.

.. currentmodule:: h5pyckle.interop_meshmode

.. class:: ArrayContextPickleGroup

    A :class:`~h5pyckle.PickleGroup` with access to an
    :class:`meshmode.array_context.ArrayContext`.

    .. attribute:: actx

    .. method:: __init__(actx, ...)

.. function:: dump(actx, ...)

    This function should be used instead of :func:`~h5pyckle.dump` when the
    object hierarchy contains :mod:`meshmode` objects that require an
    :class:`~meshmode.array_context.ArrayContext`.

.. function:: load(actx, ...)

    This function should be used instead of :func:`~h5pyckle.load` when
    the object hierarchy contains :mod:`meshmode` objects that require an
    :class:`~meshmode.array_context.ArrayContext`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
