Welcome to h5pyckle's documentation!
====================================

.. automodule:: h5pyckle

Example
=======

The default interface to :mod:`h5pyckle` is the same as for the
:mod:`pickle` module from the standard library. Arbitrary objects can be
dumped and loaded from HDF5 files using

.. literalinclude:: ../examples/basic.py
    :lines: 10-22
    :language: python
    :linenos:

Data can also be stored and loaded directly from subgroups in HD5 files.

.. literalinclude:: ../examples/basic.py
    :lines: 31-44
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
    :lines: 50-86
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
:class:`~arraycontext.ArrayContext` and cannot be stored directly
(as it could be on a GPU device). When pickling objects of the types above,
wrap the corresponding :func:`~h5pyckle.load` or :func:`~h5pyckle.dump`
calls with the context manager :func:`array_context_for_pickling`

.. currentmodule:: h5pyckle.interop_meshmode
.. autofunction:: array_context_for_pickling

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
