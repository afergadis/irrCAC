# Chance-corrected Agreement Coefficients
The `irrCAC` is a Python package that provides several functions for
calculating various chance-corrected agreement coefficients. This package
closely follows the general framework of inter-rater reliability assessment
presented by Gwet (2014).

The functionality covers calculations for various chance-corrected agreement
coefficients (CAC) among 2 or more raters. Among the CAC coefficients covered
are `Cohen's kappa`, `Conger's kappa`, `Fleiss' kappa`, `Brennan-Prediger coefficient`,
`Gwet's AC1/AC2` coefficients, and `Krippendorff's alpha`. Multiple sets of weights
are proposed for computing weighted analyses.

The functions included in this package can handle 2 types of input data. Those
types with the corresponding coefficients are in the following list:

- **Contingency Table**
  - Brennar-Prediger
  - Cohen's kappa
  - Gwet AC1/AC2
  - Krippendorff's Alpha
  - Percent Agreement
  - Schott's Pi

- **Raw Data**
  - Fleiss' kappa
  - Gwet AC1/AC2
  - Krippendorff's Alpha
  - Conger's kappa
  - Brennar-Prediger

## Note
All of these statistical procedures are described in details in
Gwet, K.L. (2014,ISBN:978-0970806284):
"Handbook of Inter-Rater Reliability," 4th edition, Advanced Analytics, LLC.

This package is a port _(with permission)_ to Python of the
[irrCAC](https://github.com/kgwet/irrCAC)` library for R by Gwet, K.L.

This particular fork of the library fixes a few bugs in the `scipy students t interval`
and supports a editable install.

## Important
This is a **work in progress** and _does not_ have (yet) the full
functionality found in the R library.

## Installation
To install the package, run:

```bash
cd irrCAC
pip install --editable ./
```

## Documentation
The documentation of the project is available at the following page:
http://irrcac.readthedocs.io/
