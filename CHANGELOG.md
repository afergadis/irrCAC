## 0.4.4 (2024-11-05)

### Fix

- prevent divide by zero when computing p_value if stderr too small

## 0.4.3 (2024-08-26)

### Refactor

- **raw**: replace depricated function call

## 0.4.2 (2024-08-02)

### Fix

- replace 'round' function with 'np.round'

## v0.4.1 (2023-08-24)

### Refactor

- update to scipy 1.11.2
- fix tex warnings during pytest

## v0.4.0 (2022-06-28)

### Refactor

- update dev-dependencies and commitizen section

### Feat

- Brennan-Prediger coefficient for raw data

## v0.3.0 (2022-03-31)

### Feat

- Conger's kappa coefficient for raw data

### Fix

- custom weights were not supported

## 0.3.0 (2022-01-21)

## 0.2.0 (2022-01-21)

## v0.2.3 (2022-01-21)

### Fix

- custom weights were not supported (closes #1)
- 
## v0.2.3 (2022-01-18)

### Fix

- revise the calculation of p-values

### Refactor

- **docs**: remove sphinx's build directory
- add more pre-commit hooks
- migrate code style to Black

## v0.2.2 (2022-01-11)

### Feat

- calculate Krippendorff's Alpha.

### Fix

- calculations for weighted coefficients.

### Refactor

- reformat files.

## v0.2.1 (2022-01-05)

### Fix

- return copy of calculations.

## v0.2.0 (2022-01-04)

### Feat

- add Percent Agreement, Scott's Pi, and Krippendorff's Alpha coefficients.

## v0.1.0 (2022-01-03)

### Fix

- added sphinx-autodoc-typehints dependency.
- path to requirements file
- remove package that causes readthedocs to fail importing project.
