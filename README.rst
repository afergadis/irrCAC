Chance-corrected Agreement Coefficients
=======================================

.. image:: https://readthedocs.org/projects/irrcac/badge/?version=latest
  :target: https://irrcac.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
  :target: https://github.com/pre-commit/pre-commit
  :alt: pre-commit

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black

The **irrCAC** is a Python package that provides several functions for
calculating various chance-corrected agreement coefficients. This package
closely follows the general framework of inter-rater reliability assessment
presented by Gwet (2014).

The functionality covers calculations for various chance-corrected agreement
coefficients (CAC) among 2 or more raters. Among the CAC coefficients covered
are Cohen's kappa, Conger's kappa, Fleiss' kappa, Brennan-Prediger coefficient,
Gwet's AC1/AC2 coefficients, and Krippendorff's alpha. Multiple sets of weights
are proposed for computing weighted analyses.

The functions included in this package can handle 2 types of input data. Those
types with the corresponding coefficients are in the following list:

1. Contingency Table

  1. Brennar-Prediger
  2. Cohen's kappa
  3. Gwet AC1/AC2
  4. Krippendorff's Alpha
  5. Percent Agreement
  6. Schott's Pi

2. Raw Data

  1. Fleiss' kappa
  2. Gwet AC1/AC2
  3. Krippendorff's Alpha
  4. Conger's kappa
  5. Brennar-Prediger

.. note::
   All of these statistical procedures are described in details in
   Gwet, K.L. (2014,ISBN:978-0970806284):
   "Handbook of Inter-Rater Reliability," 4th edition, Advanced Analytics, LLC.

   This package is a port *(with permission)* to Python of the
   `irrCAC <https://github.com/kgwet/irrCAC>`_ library for R by Gwet, K.L.

.. important::
   This is a **work in progress** and *does not* have (yet) the full
   functionality found in the R library.

Installation
------------
To install the package, run:

.. code:: bash

    pip install irrCAC

Developers
----------
To use the code for development it is recommended to install
`poetry <https://python-poetry.org/>`_ and run:

.. code:: bash

    poetry install

And add the `pre-commit` hook:

.. code:: bash

   pre-commit install

and update the hooks:

.. code:: bash

   pre-commit autoupdate

To update the project dependencies, run:

.. code:: bash

   poetry update

Next run the tests:

.. code:: bash

    poetry run pytest

There is also a config file for `tox <https://tox.readthedocs.io/en/latest/>`_
so you can automatically run the tests for various python versions like this:

.. code:: bash

    tox

Documentation
-------------
The documentation of the project is available at the following page:
http://irrcac.readthedocs.io/
