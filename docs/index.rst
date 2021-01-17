.. h5pyckle documentation master file, created by
   sphinx-quickstart on Sun Jan 10 11:30:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to h5pyckle's documentation!
====================================

.. automodule:: h5pyckle

Example
-------

The default interface to :mod:`h5pyckle` is the same as for the
:mod:`pickle` module from the standard library. Arbitrary objects can be
dumped and loaded from HDF5 files using

.. literalinclude:: ../examples/basic.py
    :lines: 12-24
    :language: python
    :linenos:

Data can also be stored and loaded directly from subgroups in HD5 files.

.. literalinclude:: ../examples/basic.py
    :lines: 33-45
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
    :lines: 52-80
    :language: python
    :linenos:

Reference
---------

.. automodule:: h5pyckle.base

Numpy
^^^^^
    return parent.type(name=name, values=values)
.. .. automodule:: h5pyckle.interop_numpy

Meshmode
^^^^^^^^

.. .. automodule:: h5pyckle.interop_meshmode

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
