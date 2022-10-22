.. image:: https://github.com/alexfikl/h5pyckle/workflows/CI/badge.svg
    :alt: Build Status
    :target: https://github.com/alexfikl/h5pyckle/actions?query=branch%3Amain+workflow%3ACI

.. image:: https://readthedocs.org/projects/h5pyckle/badge/?version=latest
    :alt: Documentation
    :target: https://h5pyckle.readthedocs.io/en/latest/?badge=latest

.. image:: https://api.reuse.software/badge/github.com/alexfikl/h5pyckle
    :alt: REUSE
    :target: https://api.reuse.software/info/github.com/alexfikl/h5pyckle

h5pyckle
========

An alternative to the venerable `pickle module <https://docs.python.org/3/library/pickle.html>`__
that uses `HDF5 <https://www.hdfgroup.org/solutions/hdf5>`__ as a storage
backend.

For a battle tested version of the same idea see the wonderful
`hickle library <https://github.com/telegraphic/hickle>`__. The main difference
between the two is that ``h5pyckle`` piggybacks on Python's
`singledispatch <https://docs.python.org/3/library/functools.html>`__ to
implement the pickling for various types. This has the benefit of

* making it very easy to add custom pickling routines for new types
* defering the dispatch issue for subclasses to a well-tested code

Links

* `Documentation <https://h5pyckle.readthedocs.io/en/latest/>`__
* `Code <https://github.com/alexfikl/h5pyckle>`__
