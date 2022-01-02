Quickstart
==========
Installation
------------
To install the package, run:

.. code:: bash

    pip install irrCAC

Developers
----------
To run the tests, install `poetry <https://python-poetry.org/>`_ and run:

.. code:: bash

    poetry install

Next run:

.. code:: bash

    poetry run pytest

There is also a config file for `tox <https://tox.readthedocs.io/en/latest/>`_
so you can automatically run the tests for various python versions like this:

.. code:: bash

    tox
