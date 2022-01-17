"""Chance-corrected Agreement Coefficient for "raw" ratings.

The functions in this module calculate chance-corrected agreement coefficients
using "raw" data, i.e., tables where each row represents a subject and each
column a rater. Each shell has the rating category.

Examples
--------
>>> from irrCAC.datasets import raw_4raters
>>> from irrCAC.raw import CAC
>>> data = raw_4raters()
>>> print(data)  # doctest: +NORMALIZE_WHITESPACE
       Rater1  Rater2  Rater3  Rater4
Units
1         1.0     1.0     NaN     1.0
2         2.0     2.0     3.0     2.0
3         3.0     3.0     3.0     3.0
4         3.0     3.0     3.0     3.0
5         2.0     2.0     2.0     2.0
6         1.0     2.0     3.0     4.0
7         4.0     4.0     4.0     4.0
8         1.0     1.0     2.0     1.0
9         2.0     2.0     2.0     2.0
10        NaN     5.0     5.0     5.0
11        NaN     NaN     1.0     1.0
12        NaN     NaN     3.0     NaN

Initialize a CAC object with the data frame of the ratings.

>>> cac4raters = CAC(data)
>>> print(cac4raters.gwet())  # doctest: +NORMALIZE_WHITESPACE
{'est': {'coefficient_value': 0.77544,
         'coefficient_name': 'AC1',
         'confidence_interval': (0.46081, 1),
         'p_value': 0.00021,
         'z': 5.42458,
         'se': 0.14295,
         'pa': 0.81818,
         'pe': 0.19032},
'weights': array([[1., 0., 0., 0., 0.],
                  [0., 1., 0., 0., 0.],
                  [0., 0., 1., 0., 0.],
                  [0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 1.]]),
'categories': [1.0, 2.0, 3.0, 4.0, 5.0]}

Get results for any available method in the class with the same data.

>>> print(cac4raters.fleiss())  # doctest: +NORMALIZE_WHITESPACE
{'est': {'coefficient_value': 0.76117,
         'coefficient_name': "Fleiss' kappa",
         'confidence_interval': (0.42438, 1),
         'p_value': 0.00042,
         'z': 4.97434,
         'se': 0.15302,
         'pa': 0.81818,
         'pe': 0.23872},
'weights': array([[1., 0., 0., 0., 0.],
                  [0., 1., 0., 0., 0.],
                  [0., 0., 1., 0., 0.],
                  [0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 1.]]),
'categories': [1.0, 2.0, 3.0, 4.0, 5.0]}

To use weights with the calculations, we pass the type of weights as argument.

>>> cac4raters_bipolar = CAC(data, weights="bipolar")
>>> print(cac4raters_bipolar.gwet())  # doctest: +NORMALIZE_WHITESPACE
{'est': {'coefficient_value': 0.90037,
         'coefficient_name': 'AC2',
         'confidence_interval': (0.66747, 1),
         'p_value': 0.0,
         'z': 8.50888,
         'se': 0.10582,
         'pa': 0.96836,
         'pe': 0.68244},
'weights': array([[1.        , 0.85714286, 0.66666667, 0.4       , 0.        ],
                  [0.85714286, 1.        , 0.93333333, 0.75      , 0.4       ],
                  [0.66666667, 0.93333333, 1.        , 0.93333333, 0.66666667],
                  [0.4       , 0.75      , 0.93333333, 1.        , 0.85714286],
                  [0.        , 0.4       , 0.66666667, 0.85714286, 1.        ]]),
'categories': [1.0, 2.0, 3.0, 4.0, 5.0]}
"""
from copy import deepcopy

import numpy as np
from scipy import stats

from irrCAC.weights import Weights


class CAC:
    """ Chance-corrected Agreement Coefficients (CAC)

    Calculates various chance-corrected agreement coefficients (CAC) among 2 or
    more raters are provided. Among the CAC coefficients covered are

    * Brennan-Prediger coefficient, (TODO)
    * Conger's kappa, (TODO)
    * Fleiss' kappa,
    * Gwet's AC1/AC2 coefficients, and
    * Krippendorff's Alpha.

    Multiple sets of weights are proposed for computing weighted analyses.
    All of these statistical procedures are described in details in Gwet
    :cite:p:`Gwe14`.

    Parameters
    ----------
    ratings : DataFrame
        A data frame of ratings where each column represents one rater and
        each row one subject.
    weights : array-like, ndarray, or str, {"identity", "quadratic", "ordinal",\
    "linear", "radical", "ratio", "circular", "bipolar"}
        A mandatory parameter that is either a string variable or a matrix.
        The string describes one of the predefined weights. If this
        parameter is a matrix then it must be a square matrix qxq where q
        is the number of possible categories where a subject can be
        classified. If some of the q possible categories are not used,
        then it is strongly advised to specify the complete list of
        possible categories as a vector in parameter ``categories``.
        Otherwise, the program may not work.
    categories : list or None, default None
        An optional vector parameter containing the list of all possible
        ratings. It may be useful in case some possible ratings are not
        used by any rater, they will still be used when calculating
        agreement coefficients. The default value is None. In this case,
        only categories reported by the raters are used in the calculations.
    confidence_level : float, default 0.95
        An optional parameter representing the confidence level associated
        with the confidence interval. Its default value is 0.95.
    N : int, default infinity
        An optional parameter representing the population size (if any).
        It may be used to perform the final population correction to the
        variance. Its default value is infinity.
    digits : int, default 5
        The number of digits to round the results.
    """

    def __init__(
        self,
        ratings,
        weights="identity",
        categories=None,
        confidence_level=0.95,
        N=np.inf,
        digits=5,
    ):
        weights_choices = (
            "identity",
            "quadratic",
            "ordinal",
            "linear",
            "radical",
            "ratio",
            "circular",
            "bipolar",
        )
        if weights not in weights_choices:
            raise ValueError(f"weights values can be any of {weights_choices}")
        if not 0.9 <= confidence_level <= 0.99:
            raise ValueError("Please provide a value in range [0.90, 0.99].")
        self.confidence_level = confidence_level

        # Drop subjects with no ratings.
        self.ratings = ratings.dropna(how="all")
        self.ratings.replace(to_replace="", value=np.nan, inplace=True)
        self.n, self.r = self.ratings.shape  # subjects, raters
        self.f = self.n / N
        if categories is None:
            self.categories = sorted(self.ratings.stack().unique().tolist())
        else:
            self.categories = categories
        self.q = len(self.categories)
        if isinstance(weights, str):
            self.weights_name = weights
            weights_functions = Weights(self.categories)
            self.weights_mat = weights_functions[self.weights_name]
        else:
            self.weights_name = "Custom Weights"
            self.weights_mat = np.asarray(weights)
        self.digits = digits
        self.coefficient_value = 0
        self.coefficient_name = None
        self.confidence_interval = (0, 0)
        self.p_value = 0
        self.z = 0
        self.se = 0
        self.pa = 0
        self.pe = 0
        self.agreement = {
            "est": dict(
                coefficient_value=0,
                coefficient_name=None,
                confidence_interval=(0, 0),
                p_value=0,
                z=0,
                se=0,
                pa=0,
                pe=0,
            ),
            "weights": self.weights_mat,
            "categories": self.categories,
        }

    def __str__(self):
        class_path = f"{CAC.__module__}.{CAC.__name__}"
        subjects = f"Subjects: {self.n}"
        raters = f"Raters: {self.r}"
        categories = f"Categories: {self.categories}"
        weights_name = f'Weights: "{self.weights_name}"'
        _str = f"{class_path} {subjects}, {raters}, {categories}, {weights_name}"
        return f"<{_str}>"

    def __repr__(self):
        return self.__str__()

    def gwet(self):
        """Gwet's AC1/AC2 coefficient.

        The AC1 coefficient was suggested by Gwet :cite:p:`Gwe08` as a
        paradox-resistant alternative to Cohen’s Kappa. The percent chance agreement it
        is defined as the propensity for raters to agree on hard-to-score subjects and
        is calculated by multiplying the probability to agree when the rating is random
        by the probability to select a hard-to-score subject.

        The Gwet's AC2 coefficient is the one when using weights for the
        calculation.
        """
        agree_mat = np.zeros(shape=(self.n, self.q))
        for k in range(self.q):
            agree_mat[:, k] = self.ratings[self.ratings == self.categories[k]].count(
                axis=1
            )
        agree_mat_w = np.transpose(np.matmul(self.weights_mat, agree_mat.T))
        ri_vec = agree_mat.sum(axis=1)
        sum_q = (agree_mat * (agree_mat_w - 1)).sum(axis=1)
        n2more = sum(ri_vec >= 2)
        pa = sum(sum_q[ri_vec >= 2] / (ri_vec * (ri_vec - 1))[ri_vec >= 2]) / n2more
        pi_vec = (
            np.repeat(1 / self.n, self.n).reshape(self.n, 1)
            * (agree_mat / np.repeat(ri_vec, self.q).reshape(self.n, self.q))
        ).T.sum(axis=1)
        weights_mat_sum = sum(sum(self.weights_mat))
        if self.q >= 2:
            pe = weights_mat_sum * sum(pi_vec * (1 - pi_vec)) / (self.q * (self.q - 1))
        else:
            pe = 1 - 1e-15
        ac1 = (pa - pe) / (1 - pe)
        den_ivec = ri_vec * (ri_vec - 1)
        den_ivec = den_ivec - (den_ivec == 0)
        pa_ivec = sum_q / den_ivec
        pe_r2 = pe * (ri_vec >= 2)
        ac1_ivec = (self.n / n2more) * (pa_ivec - pe_r2) / (1 - pe)
        pe_ivec = (
            (weights_mat_sum / (self.q * (self.q - 1)))
            * np.matmul(agree_mat, (1 - pi_vec))
            / ri_vec
        )
        ac1_ivec_x = ac1_ivec - 2 * (1 - ac1) * (pe_ivec - pe) / (1 - pe)
        var_ac1 = (1 - self.f) / (self.n * (self.n - 1)) * sum((ac1_ivec_x - ac1) ** 2)
        stderr = np.sqrt(var_ac1)
        p_value = 2 * (1 - stats.t.cdf(abs(ac1 / stderr), self.n - 1))
        lcb, ucb = stats.t.interval(
            alpha=self.confidence_level, df=self.n - 1, scale=stderr, loc=ac1
        )
        ucb = min(1, ucb)

        if weights_mat_sum == self.q:
            coeff_name = "AC1"
        else:
            coeff_name = "AC2"

        self.coefficient_value = np.round(ac1, self.digits)
        self.coefficient_name = coeff_name
        self.confidence_interval = (round(lcb, self.digits), round(ucb, self.digits))
        self.p_value = p_value
        self.z = round(ac1 / stderr, self.digits)
        self.se = round(stderr, self.digits)
        self.pa = round(pa, self.digits)
        self.pe = round(pe, self.digits)
        self.agreement["est"].update(
            dict(
                coefficient_name=coeff_name,
                pa=self.pa,
                pe=self.pe,
                se=self.se,
                z=self.z,
                coefficient_value=self.coefficient_value,
                confidence_interval=self.confidence_interval,
                p_value=self.p_value,
            )
        )
        return deepcopy(self.agreement)

    def fleiss(self):
        """Fleiss' generalized kappa coefficient.

        Fleiss :cite:p:`Fle71` defined the percent chance agreement the
        probability that any pair of raters classify a subject into the same category.

        Notes
        -----
        The calculation of the kappa coefficient here takes into account any
        missing values.
        """
        agree_mat = np.zeros(shape=(self.n, self.q))
        for c in range(self.q):
            agree_mat[:, c] = self.ratings[self.ratings == self.categories[c]].count(
                axis=1
            )
        agree_mat_w = np.transpose(np.matmul(self.weights_mat, agree_mat.T))
        ri_vec = agree_mat.sum(axis=1)
        sum_q = (agree_mat * (agree_mat_w - 1)).sum(axis=1)
        n2more = sum(ri_vec >= 2)
        pa = float(
            sum(sum_q[ri_vec >= 2] / (ri_vec * (ri_vec - 1))[ri_vec >= 2]) / n2more
        )
        pi_vec = (
            np.repeat(1 / self.n, self.n).reshape(self.n, 1)
            * (agree_mat / np.repeat(ri_vec, self.q).reshape(self.n, self.q))
        ).T.sum(axis=1)
        pe = float(
            np.sum(
                self.weights_mat
                * (pi_vec.reshape(self.q, 1) * pi_vec.reshape(1, self.q))
            )
        )
        fleiss_kappa = (pa - pe) / (1 - pe)
        den_ivec = ri_vec * (ri_vec - 1)
        den_ivec = den_ivec - (den_ivec == 0)
        pa_ivec = sum_q / den_ivec
        pe_r2 = pe * (ri_vec >= 2)
        kappa_ivec = (self.n / n2more) * (pa_ivec - pe_r2) / (1 - pe)
        pi_vec_wk_ = np.matmul(self.weights_mat, pi_vec)
        pi_vec_w_k = np.matmul(self.weights_mat.T, pi_vec)
        pi_vec_w = (pi_vec_wk_ + pi_vec_w_k) / 2
        pe_ivec = np.matmul(agree_mat, pi_vec_w) / ri_vec
        kappa_ivec_x = kappa_ivec - 2 * (1 - fleiss_kappa) * (pe_ivec - pe) / (1 - pe)
        var_fleiss = (
            (1 - self.f)
            / (self.n * (self.n - 1))
            * sum((kappa_ivec_x - fleiss_kappa) ** 2)
        )
        stderr = np.sqrt(var_fleiss)
        p_value = float(2 * (1 - stats.t.cdf(abs(fleiss_kappa / stderr), self.n - 1)))
        lcb, ucb = stats.t.interval(
            alpha=self.confidence_level, df=self.n - 1, scale=stderr, loc=fleiss_kappa
        )
        ucb = min(1, ucb)

        self.coefficient_value = round(fleiss_kappa, self.digits)
        self.coefficient_name = "Fleiss' kappa"
        self.confidence_interval = (round(lcb, self.digits), round(ucb, self.digits))
        self.p_value = p_value
        self.z = round(fleiss_kappa / stderr, self.digits)
        self.se = round(stderr, self.digits)
        self.pa = round(pa, self.digits)
        self.pe = round(pe, self.digits)
        self.agreement["est"].update(
            dict(
                coefficient_name=self.coefficient_name,
                pa=self.pa,
                pe=self.pe,
                se=self.se,
                z=self.z,
                coefficient_value=self.coefficient_value,
                confidence_interval=self.confidence_interval,
                p_value=self.p_value,
            )
        )
        return deepcopy(self.agreement)

    def krippendorff(self):
        """Krippendorff’s alpha coefficient for an arbitrary number of raters.

        Krippendorff’s alpha :cite:p:`Kri70,Kri80` coefficient for an arbitrary number
        of raters (2, 3, +) when the input data represent the raw ratings reported for
        each subject and each rater.
        """
        agree_mat = np.zeros(shape=(self.n, self.q))
        for k in range(self.q):
            agree_mat[:, k] = self.ratings[self.ratings == self.categories[k]].count(
                axis=1
            )
        agree_mat_w = np.transpose(np.matmul(self.weights_mat, agree_mat.T))
        ri_vec = agree_mat.sum(axis=1)
        agree_mat = agree_mat[ri_vec >= 2]
        agree_mat_w = agree_mat_w[ri_vec >= 2]
        ri_vec = ri_vec[ri_vec >= 2]
        ri_mean = np.mean(ri_vec)
        n = agree_mat.shape[0]
        epsi = 1 / np.sum(ri_vec)
        sum_q = (agree_mat * (agree_mat_w - 1)).sum(axis=1)
        paprime = np.sum(sum_q / (ri_mean * (ri_vec - 1))) / n
        pa = float((1 - epsi) * paprime + epsi)
        pi_vec = np.matmul(np.repeat(1 / n, n).reshape(1, n), agree_mat / ri_mean).T
        pe = float(np.sum(self.weights_mat * np.matmul(pi_vec, pi_vec.T)))
        krippen_alpha = (pa - pe) / (1 - pe)
        krippen_alpha_est = np.round(krippen_alpha, self.digits)
        krippen_alpha_prime = (paprime - pe) / (1 - pe)
        pa_ivec = (
            sum_q / (ri_mean * (ri_vec - 1)) - pa * (ri_vec - ri_mean) / ri_mean
        ).reshape(-1, 1)
        krippen_ivec = (pa_ivec - pe) / (1 - pe)
        pi_vec_wk_ = np.matmul(self.weights_mat, pi_vec)
        pi_vec_w_k = np.matmul(self.weights_mat.T, pi_vec)
        pi_vec_w = (pi_vec_wk_ + pi_vec_w_k) / 2
        pe_ivec = np.matmul(agree_mat, pi_vec_w) / ri_mean - (
            pe * (ri_vec - ri_mean) / ri_mean
        ).reshape(-1, 1)
        krippen_ivec_x = krippen_ivec - 2 * (1 - krippen_alpha_prime) * (
            pe_ivec - pe
        ) / (1 - pe)
        var_krippen = (
            (1 - self.f)
            / (n * (n - 1))
            * sum((krippen_ivec_x - krippen_alpha_prime) ** 2)
        )
        stderr = np.sqrt(float(var_krippen))
        p_value = 2 * (1 - stats.t.cdf(abs(krippen_alpha / stderr), n - 1))
        lcb, ucb = stats.t.interval(
            alpha=self.confidence_level, df=n - 1, scale=stderr, loc=krippen_alpha
        )
        ucb = min(1, ucb)
        self.coefficient_value = round(krippen_alpha_est, self.digits)
        self.coefficient_name = "Krippendorff's Alpha"
        self.confidence_interval = (round(lcb, self.digits), round(ucb, self.digits))
        self.p_value = p_value
        self.z = round(krippen_alpha_est / stderr, self.digits)
        self.se = round(stderr, self.digits)
        self.pa = round(pa, self.digits)
        self.pe = round(pe, self.digits)
        self.agreement["est"].update(
            dict(
                coefficient_name=self.coefficient_name,
                pa=self.pa,
                pe=self.pe,
                se=self.se,
                z=self.z,
                coefficient_value=self.coefficient_value,
                confidence_interval=self.confidence_interval,
                p_value=self.p_value,
            )
        )
        return deepcopy(self.agreement)
