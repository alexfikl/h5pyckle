.. image:: https://github.com/alexfikl/h5pyckle/workflows/CI/badge.svg
    :alt: Build Status
    :target: https://github.com/alexfikl/h5pyckle/actions?query=branch%3Amain+workflow%3ACI

.. image:: https://readthedocs.org/projects/h5pyckle/badge/?version=latest
    :alt: Documentation
    :target: https://h5pyckle.readthedocs.io/en/latest/?badge=latest

h5pyckle
========

An alternative to the venerable `pickle module <https://docs.python.org/3/library/pickle.html>`__
that uses `HDF5 <https://www.hdfgroup.org/solutions/hdf5>`__ as a storage
backend.

For a battle tested version of the same idea see the wonderful
`hickle library <https://github.com/telegraphic/hickle>`__. The main difference
between the two is that ``h5pyckle`` piggybacks on Python's
`singledispatch <https://docs.python.org/3/library/functools.html>`__ to
implement the pickling for various types, which makes it quite flexible.

Links

* `Documentation <https://h5pyckle.readthedocs.io/en/latest/>`__
* `Code <https://github.com/alexfikl/h5pyckle>`__
