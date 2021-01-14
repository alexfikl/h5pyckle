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

.. code:: python

    from h5pyckle import dump, load

    arg_in = {
        "Author": {
            "FirstName": "Jane",
            "LastName": "Doe",
            "Affiliations": ["Some University", "Friendly Company"],
        }
    }
    dump(arg_in, "pickled.h5")
    arg_out = load("pickled.h5")
    assert arg_in == arg_out

Data can also be stored and loaded directly from subgroups in HD5 files.

.. code:: python

    from h5pyckle import dump_to_group, load_from_group

    with h5py.File("pickled.h5", mode="a") as h5:
        subgroup = h5.require_group("subgroup/for/pickle/data")
        dump_to_group(arg_in, subgroup)

    with h5py.File("pickled.h5", mode="r") as h5:
        subgroup = h5["subgroup/for/pickle/data"]
        arg_out = load_from_group(subgroup)
        first_name = load_by_pattern(subgroup, "FirstName")

    assert arg_in == arg_out
    assert first_name == arg_in["Author"]["FirstName"]

The library piggybacks on the :mod:`pickle` module and looks for the usual
:meth:`object.__getstate__`, :meth:`object.__setstate__`,
:meth:`object.__reduce__`, etc. When no alternative is found, it falls back
to :func:`pickle.dumps` / :func:`pickle.loads` and stores the result in a
group attribute or dataset (depending on the size).

To register custom methods for new types, use the :func:`functools.singledispatch`
method as

.. code:: python

    @dataclass
    class MyClass:
        name: str
        values: np.ndarray

    @dumper.register(MyClass)
    def _(obj: MyClass, parent: PickleGroup, name: Optional[str] = None):
        grp = parent.create_type(obj)
        grp.attrs["name"] = obj.name
        grp.create_dataset("values", data=obj.values)

    @loader.register(MyClass)
    def _(group: PickleGroup) -> MyClass:
        return MyClass(
            name=group.attrs["name"],
            values=group.values[:])

Reference
---------

.. automodule:: h5pyckle.base

Numpy
^^^^^

.. .. automodule:: h5pyckle.interop_numpy

Meshmode
^^^^^^^^

.. .. automodule:: h5pyckle.interop_meshmode

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
