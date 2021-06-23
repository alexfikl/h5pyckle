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

.. automodule:: h5pyckle.interop_numpy

Meshmode
--------

.. automodule:: h5pyckle.interop_meshmode

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
