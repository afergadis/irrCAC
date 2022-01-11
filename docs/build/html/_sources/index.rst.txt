Chance-corrected Agreement Coefficients
=======================================

The **irrCAC** is an Python package that provides several functions for
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

1. `Contingency Table <irrCAC.html#module-irrCAC.table>`_:

  1. Brennar-Prediger
  2. Cohen's kappa
  3. Gwet AC1/AC2
  4. Krippendorff's Alpha
  5. Percent Agreement
  6. Schott's Pi

2. `Raw Data <irrCAC.html#module-irrCAC.raw>`_:

  1. Fleiss' kappa
  2. Gwet AC1/AC2
  3. Krippendorff's Alpha

The package also supports functionality for weighted analysis using a set of
predefined `weights <irrCAC.html#module-irrCAC.weights>`_ and interpreting the
level of agreement using `benchmarking <irrCAC.html#module-irrCAC.benchmark>`_.

Please refer to `usage <usage.html>`_ and the `api <py-modindex.html>`_ for
more.

.. note::
   All of these statistical procedures are described in details in
   Gwet, K.L. (2014,ISBN:978-0970806284):
   "Handbook of Inter-Rater Reliability," 4th edition, Advanced Analytics, LLC.

   This package is a port *(with permission)* to Python of the
   `irrCAC <https://github.com/kgwet/irrCAC>`_ library for R by Gwet, K.L.

.. important::
   This is a **work in progress** and *does not* have (yet) the full
   functionality found in the R library.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   usage

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
