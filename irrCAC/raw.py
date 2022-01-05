""" Chance-corrected Agreement Coefficient for "raw" ratings.

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

>>> cac4raters_bipolar = CAC(data, weights='bipolar')
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
import numpy as np
from copy import deepcopy
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
    * Krippendorff's alpha. (TODO)

    Multiple sets of weights are proposed for computing weighted analyses.
    All of these statistical procedures are described in details in [1].
    
    [1] Gwet, K.L. (2014, ISBN:978-0970806284): *"Handbook of Inter-Rater
    Reliability"*, 4th edition, Advanced Analytics, LLC.

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
            weights='identity',
            categories=None,
            confidence_level=0.95,
            N=np.inf,
            digits=5):
        weights_choices = (
            "identity", "quadratic", "ordinal", "linear", "radical", "ratio",
            "circular", "bipolar")
        if weights not in weights_choices:
            raise ValueError(f'weights values can be any of {weights_choices}')
        if not 0.9 <= confidence_level <= 0.99:
            raise ValueError('Please provide a value in range [0.90, 0.99].')
        self.confidence_level = confidence_level

        # Drop subjects with no ratings.
        self.ratings = ratings.dropna(how='all')
        self.ratings.replace(to_replace='', value=np.nan, inplace=True)
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
            self.weights_name = 'Custom Weights'
            self.weights_mat = np.asarray(weights)
        self.agree_mat = np.zeros(shape=(self.n, self.q))
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
            'est': {
                'coefficient_value': 0,
                'coefficient_name': None,
                'confidence_interval': (0, 0),
                'p_value': 0,
                'z': 0,
                'se': 0,
                'pa': 0,
                'pe': 0,
            },
            'weights': self.weights_mat,
            'categories': self.categories
        }

    def __str__(self):
        class_path = f'{CAC.__module__}.{CAC.__name__}'
        subjects = f'Subjects: {self.n}'
        raters = f'Raters: {self.r}'
        categories = f'Categories: {self.categories}'
        weights_name = f'Weights: "{self.weights_name}"'
        _str = f'{class_path} {subjects}, {raters}, {categories}, {weights_name}'
        return f'<{_str}>'

    def __repr__(self):
        return self.__str__()

    def gwet(self):
        """ Gwet's AC1/AC2 coefficient.
        
        The AC1 coefficient was suggested by Gwet (2008a) as a paradox-resistant
        alternative to Cohen’s Kappa. The percent chance agreement it is
        defined as the propensity for raters to agree on hard-to-score
        subjects and is calculated by multiplying the probability to agree
        when the rating is random by the probability to select a
        hard-to-score subject.

        The Gwet's AC2 coefficient is the one when using weight for the
        calculation.
        """
        for k in range(self.q):
            self.agree_mat[:, k] = self.ratings[
                self.ratings == self.categories[k]].count(axis=1)
        agree_mat_w = np.transpose(
            np.matmul(self.weights_mat, self.agree_mat.T))
        ri_vec = self.agree_mat.sum(axis=1)
        sum_q = (self.agree_mat * (agree_mat_w - 1)).sum(axis=1)
        n2more = sum(ri_vec >= 2)
        pa = sum(
            sum_q[ri_vec >= 2] / (ri_vec * (ri_vec - 1))[ri_vec >= 2]) / n2more
        pi_vec = (
            np.repeat(1 / self.n, self.n).reshape(self.n, 1) * (
                self.agree_mat /
                np.repeat(ri_vec, self.q).reshape(self.n, self.q))).T.sum(
                    axis=1)
        weights_mat_sum = sum(sum(self.weights_mat))
        if self.q > 1:
            pe = weights_mat_sum * sum(pi_vec * (1 - pi_vec)) / (
                self.q * (self.q - 1))
        else:
            pe = 1
        ac1 = (pa - pe) / (1 - pe)
        den_ivec = ri_vec * (ri_vec - 1)
        den_ivec = den_ivec - (den_ivec == 0)
        pa_ivec = sum_q / den_ivec
        pe_r2 = pe * (ri_vec >= 2)
        ac1_ivec = (self.n / n2more) * (pa_ivec - pe_r2) / (1 - pe)
        pe_ivec = (weights_mat_sum / (self.q * (self.q - 1))) * np.matmul(
            self.agree_mat, (1 - pi_vec)) / ri_vec
        ac1_ivec_x = ac1_ivec - 2 * (1 - ac1) * (pe_ivec - pe) / (1 - pe)
        var_ac1 = (1 - self.f) / (self.n * (self.n - 1)) * sum(
            (ac1_ivec_x - ac1)**2)
        stderr = np.sqrt(var_ac1)
        p_value = 2 * (1 - stats.t.cdf(abs(ac1 / stderr), self.n - 1))
        lcb, ucb = stats.t.interval(
            alpha=self.confidence_level, df=self.n - 1, scale=stderr, loc=ac1)
        ucb = min(1, ucb)

        if weights_mat_sum == self.q:
            coeff_name = 'AC1'
        else:
            coeff_name = 'AC2'

        self.coefficient_value = np.round(ac1, self.digits)
        self.coefficient_name = coeff_name
        self.confidence_interval = (
            round(lcb, self.digits), round(ucb, self.digits))
        self.p_value = round(p_value, self.digits)
        self.z = round(ac1 / stderr, self.digits)
        self.se = round(stderr, self.digits)
        self.pa = round(pa, self.digits)
        self.pe = round(pe, self.digits)
        self.agreement['est'].update(
            {
                'coefficient_name': coeff_name,
                'pa': self.pa,
                'pe': self.pe,
                'se': self.se,
                'z': self.z,
                'coefficient_value': self.coefficient_value,
                'confidence_interval': self.confidence_interval,
                'p_value': self.p_value
            })
        return deepcopy(self.agreement)

    def fleiss(self):
        """ Fleiss' generalized kappa coefficient.

        Fleiss' defined the percent chance agreement the probability that any
        pair of raters classify a subject into the same category.

        The calculation of the kappa coefficient here takes into account any
        missing values.
        """
        for c in range(self.q):
            self.agree_mat[:, c] = self.ratings[
                self.ratings == self.categories[c]].count(axis=1)
        ri_vec = self.agree_mat.sum(axis=1)
        sum_q = (self.agree_mat * (self.agree_mat - 1)).sum(axis=1)
        n2more = sum(ri_vec >= 2)
        pa = sum(
            sum_q[ri_vec >= 2] / (ri_vec * (ri_vec - 1))[ri_vec >= 2]) / n2more
        pi_vec = (
            np.repeat(1 / self.n, self.n).reshape(self.n, 1) * (
                self.agree_mat /
                np.repeat(ri_vec, self.q).reshape(self.n, self.q))).T.sum(
                    axis=1)
        pe = sum(pi_vec * pi_vec.T)
        kappa = (pa - pe) / (1 - pe)
        den_ivec = ri_vec * (ri_vec - 1)
        den_ivec = den_ivec - (den_ivec == 0)
        pa_ivec = sum_q / den_ivec
        pe_r2 = pe * (ri_vec >= 2)
        kappa_ivec = (self.n / n2more) * (pa_ivec - pe_r2) / (1 - pe)
        pe_ivec = np.matmul(self.agree_mat, pi_vec) / ri_vec
        kappa_ivec_x = kappa_ivec - 2 * (1 - kappa) * (pe_ivec - pe) / (1 - pe)
        var_kappa = (1 - self.f) / (self.n * (self.n - 1)) * sum(
            (kappa_ivec_x - kappa)**2)
        stderr = np.sqrt(var_kappa)
        p_value = 2 * (1 - stats.t.cdf(abs(kappa / stderr), self.n - 1))
        lcb, ucb = stats.t.interval(
            alpha=self.confidence_level,
            df=self.n - 1,
            scale=stderr,
            loc=kappa)
        ucb = min(1, ucb)

        self.coefficient_value = round(kappa, self.digits)
        self.coefficient_name = "Fleiss' kappa"
        self.confidence_interval = (
            round(lcb, self.digits), round(ucb, self.digits))
        self.p_value = round(p_value, self.digits)
        self.z = round(kappa / stderr, self.digits)
        self.se = round(stderr, self.digits)
        self.pa = round(pa, self.digits)
        self.pe = round(pe, self.digits)
        self.agreement['est'].update(
            {
                'coefficient_name': self.coefficient_name,
                'pa': self.pa,
                'pe': self.pe,
                'se': self.se,
                'z': self.z,
                'coefficient_value': self.coefficient_value,
                'confidence_interval': self.confidence_interval,
                'p_value': self.p_value
            })
        return deepcopy(self.agreement)

    # def conger(self):
    #     for k in range(self.q):
    #         self.agree_mat[:, k] = self.ratings[
    #             self.ratings == self.categories[k]].count(axis=1)
    #     agree_mat_w = np.transpose(
    #         np.matmul(self.weights_mat, self.agree_mat.T))
    #     classif_mat = np.zeros((self.r, self.q))
    #     for k in range(self.q):
    #         with_mis = self.ratings == self.categories[k]
    #         without_mis = with_mis.T.fillna(False)
    #         classif_mat[:, k] = without_mis.sum(axis=1)
    #     ri_vec = self.agree_mat.sum(axis=1)
    #     sum_q = (self.agree_mat * (self.agree_mat - 1)).sum(axis=1)
    #     n2more = sum(ri_vec >= 2)
    #     pa = sum(
    #         sum_q[ri_vec >= 2] / (ri_vec * (ri_vec - 1))[ri_vec >= 2]) / n2more
    #     ng_vec = classif_mat.sum(axis=1).reshape(-1, 1)
    #     pgk_mat = classif_mat / np.broadcast_to(ng_vec, (self.r, self.q))
    #     p_mean_k = pgk_mat.T.sum(axis=1) / self.r
    #     p_mean_k = p_mean_k.reshape(-1, 1)
    #     s2kl_mat = (
    #         np.matmul(pgk_mat.T, pgk_mat) - self.r *
    #         (p_mean_k * p_mean_k.T)) / (
    #             self.r - 1)
    #     pe = np.sum(
    #         self.weights_mat * (p_mean_k * p_mean_k.T - s2kl_mat / self.r))
    #     conger_kappa = (pa - pe) / (1 - pe)
    #     bkl_mat = (self.weights_mat + self.weights_mat.T) / 2
    #     pe_ivec1 = self.r * (
    #         self.agree_mat * (p_mean_k.T * bkl_mat).sum(axis=1)).sum(axis=1)
    #     pe_ivec2 = np.zeros((1, self.n))
    #     lambda_ig_mat = np.zeros((self.n, self.r))
    #     is_numeric_ratings = self.ratings.applymap(
    #         lambda x: isinstance(x, (int, float))).all(1).sum() == self.n
    #     if is_numeric_ratings:
    #         epsi_ig_mat = 1 - self.ratings.isna()
    #     else:
    #         epsi_ig_mat = 1 - self.ratings.applymap(
    #             lambda x: isinstance(x, str))
    #     for k in range(self.q):
    #         lambda_ig_kmat = np.zeros((self.n, self.r))
    #         for l in range(self.q):
    #             delta_ig_mat = self.ratings == self.categories[l]
    #     pass
