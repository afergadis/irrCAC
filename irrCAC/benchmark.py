""" The "Cumulative Probability" approach to Benchmarking.
"""
from scipy.stats import norm


class Benchmark:
    r"""Compute benchmark scale membership probabilities.

    An elaborate approach to interpret a kappa value is based on the notion of
    cumulative interval membership probability (CIMP). The interval probability
    represents the Normality-based probability that the "true" agreement
    coefficient kappa belongs to the interval in question and is calculated
    based on :math:`\hat{\kappa}` and an arbitrary interval :math:`(a, b)` as
    follows:

    .. math::
        P(a \le \kappa_1 \le b)
            &= P[(\hat{\kappa}_1 - b)/se(\hat{\kappa}_1)
               \le Z \le
               (\hat{\kappa}_1 - a)/se(\hat{\kappa}_1)]
            &= \Phi[(\hat{\kappa}_1 - b)/se(\hat{\kappa}_1)] -
               \Phi[(\hat{\kappa}_1 - a)/se(\hat{\kappa}_1)]

    where :math:`\Phi` is the cumulative distribution function of the standard
    Normal distribution. The general rule consists of retaining the highest
    interval whose CIMP equals or exceeds the threshold of 0.95. For more
    details see `Inter-rater reliability among multiple raters when subjects
    are rated by different pairs of subjects.
    <https://inter-rater-reliability.blogspot.com/2018/02/>`_

    Parameters
    ----------
    coeff : float
        A floating number representing the estimated value of an agreement
        coefficient.
    se : float
        The coefficient standard error.

    Examples
    --------
    Using the following example, for kappa 0.67 and standard error 0.15,
    it is recommended to consider the agreement as moderate on the Altman scale.

    >>> from irrCAC.benchmark import Benchmark
    >>> benchmark = Benchmark(coeff=0.67, se=0.15)
    >>> print(benchmark.altman())  # doctest: +NORMALIZE_WHITESPACE
    {'scale': [(0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (-1.0, 0.2)],
    'Altman': ['Very Good', 'Good', 'Moderate', 'Fair', 'Poor'],
    'CumProb': [0.18168, 0.67511, 0.96356, 0.99912, 1.0]}
    >>> my_scale = dict(
    ... lb=[0.6, 0.3, 0.0],
    ... ub=[1.0, 0.6, 0.3],
    ... interp=['Excellent', 'Acceptable', 'Poor'],
    ... scale_name='My Scale')
    >>> print(benchmark.interpret(my_scale))  # doctest: +NORMALIZE_WHITESPACE
    {'scale': [(0.6, 1.0), (0.3, 0.6), (0.0, 0.3)],
    'My Scale': ['Excellent', 'Acceptable', 'Poor'],
    'CumProb': [0.67511, 0.99308, 1.0]}
    """

    def __init__(self, coeff: float, se: float):
        assert coeff <= 1, ValueError("`coeff` value cannot exceed 1.")
        self.coeff = coeff
        self.se = se

    def __str__(self):
        return (
            f"<Benchmark scales Coefficient value: {self.coeff}, "
            f"Standard Error: {self.se}>"
        )

    def __repr__(self):
        return self.__str__()

    def interpret(self, bench):
        """Interpret the agreement coefficient on a benchmark scale.

        To interpret the agreement coefficient we see in which range the
        cumulative probability exceeds 0.95. E.g., if we have a coefficient
        value of 0.67 with standard error 0.15, we get the following
        results.

        =========== ========= ========
        Scale       Altman    CumProb
        =========== ========= ========
        (0.8, 0.1)  Very Good 0.18168
        (0.6, 0.8)  Good      0.67511
        (0.4, 0.6)  Moderate  0.96356
        (0.2, 0.4)  Fair      0.99912
        (-1.0, 0.2) Poor      1.0
        =========== ========= ========

        It is safer to say that we have a *Moderate* agreement (the first scale
        that is >= 0.95), than to say that we have a *Good* agreement (because
        0.6 <= 0.67 <= 0.8). The reason for that is that we have a large
        standard error.

        Parameters
        ----------
        bench : dict
            A dictionary with the lower and upper bounds of the scale, the
            interpretation of each scale and a scale name.

        Returns
        -------
        dict
            A dict with three keys: Kappa intervals, Benchmark scale
            interpretation, and Cumulative probability. For example:

            ``{'scale': [
            (0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (-1.0, 0.2)],
            'Altman': ['Very Good', 'Good', 'Moderate', 'Fair', 'Poor'],
            'CumProb': [0.18168, 0.67511, 0.96356, 0.99912, 1.0]}``

        """
        for key in ("lb", "ub", "interp", "scale_name"):
            if key not in bench:
                raise ValueError(
                    "Please provide a dictionary like this: "
                    '`{"lb": list, "ub": list, "interp": list, '
                    '"scale_name": str}`.'
                )
        n = len(bench["lb"])
        cmprob = []
        trancate_fact = norm.cdf((self.coeff + 1) / self.se) - norm.cdf(
            (self.coeff - 1) / self.se
        )
        for i in range(n):
            value = (
                norm.cdf((self.coeff - bench["lb"][i]) / self.se)
                - norm.cdf((self.coeff - bench["ub"][i]) / self.se)
            ) / trancate_fact
            if i == 0:
                cmprob.append(value)
            else:
                cmprob.append(cmprob[i - 1] + value)
        cmprob = [round(prob, 5) for prob in cmprob]

        return {
            "scale": list(zip(bench["lb"], bench["ub"])),
            f'{bench["scale_name"]}': bench["interp"],
            "CumProb": cmprob,
        }

    def altman(self):
        """Interpret the level of agreement using the Altman :cite:p:`Alt90` \
        benchmark scale.

        +----------------+------------+
        | Interpretation | Scale      |
        +================+============+
        | Very Good      | 0.8 - 1.0  |
        +----------------+------------+
        | Good           | 0.6 - 0.8  |
        +----------------+------------+
        | Moderate       | 0.4 - 0.6  |
        +----------------+------------+
        | Fair           | 0.2 - 0.4  |
        +----------------+------------+
        | Poor           | -1.0 - 0.2 |
        +----------------+------------+
        """
        scale = dict(
            lb=[0.8, 0.6, 0.4, 0.2, -1.0],
            ub=[1.0, 0.8, 0.6, 0.4, 0.2],
            interp=["Very Good", "Good", "Moderate", "Fair", "Poor"],
            scale_name="Altman",
        )
        return self.interpret(scale)

    def cicchetti_sparrow(self):
        """ Interpret the level of agreement using the Cicchetti and Sparrow \
        :cite:p:`CS81` benchmark scale.

        +----------------+-------------+
        | Interpretation | Scale       |
        +================+=============+
        | Excellent      | 0.75 - 1.0  |
        +----------------+-------------+
        | Good           | 0.6  - 0.75 |
        +----------------+-------------+
        | Fair           | 0.4  - 0.6  |
        +----------------+-------------+
        | Poor           | 0.0  - 0.4  |
        +----------------+-------------+
        """
        scale = dict(
            lb=[0.75, 0.6, 0.4, 0.0],
            ub=[1.0, 0.75, 0.6, 0.4],
            interp=["Excellent", "Good", "Fair", "Poor"],
            scale_name="Cicchetti",
        )
        return self.interpret(scale)

    def fleiss(self):
        """Interpret the level of agreement using the Fleiss :cite:p:`Fle71` \
        benchmark scale.

        +----------------+-------------+
        | Interpretation | Scale       |
        +================+=============+
        | Excellent      | 0.75 - 1.0  |
        +----------------+-------------+
        | Fair to Good   | 0.4  - 0.75 |
        +----------------+-------------+
        | Poor           | 0.0  - 0.4  |
        +----------------+-------------+
        """
        scale = dict(
            lb=[0.75, 0.4, -1.0],
            ub=[1.0, 0.75, 0.4],
            interp=["Excellent", "Intermediate to Good", "Poor"],
            scale_name="Fleiss",
        )
        return self.interpret(scale)

    def landis_koch(self):
        """Interpret the level of agreement using the Landis and Koch :cite:p:`LK77` \
        scale.

        +----------------+------------+
        | Interpretation | Scale      |
        +================+============+
        | Almost Perfect | 0.8 - 1.0  |
        +----------------+------------+
        | Substantial    | 0.6 - 0.8  |
        +----------------+------------+
        | Moderate       | 0.4 - 0.6  |
        +----------------+------------+
        | Fair           | 0.2 - 0.4  |
        +----------------+------------+
        | Slight         | 0.0 - 0.2  |
        +----------------+------------+
        | Poor           | -1.0 - 0.0 |
        +----------------+------------+
        """
        scale = dict(
            lb=[0.8, 0.6, 0.4, 0.2, 0.0, -1.0],
            ub=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
            interp=[
                "Almost Perfect",
                "Substantial",
                "Moderate",
                "Fair",
                "Slight",
                "Poor",
            ],
            scale_name="Landis-Koch",
        )
        return self.interpret(scale)

    def regier(self):
        """Interpret the level of agreement using the Regier et al. :cite:p:`RNC+13` \
        benchmark scale.

        +----------------+------------+
        | Interpretation | Scale      |
        +================+============+
        | Excellent      | 0.8 - 1.0  |
        +----------------+------------+
        | Very Good      | 0.6 - 0.8  |
        +----------------+------------+
        | Good           | 0.4 - 0.6  |
        +----------------+------------+
        | Questionable   | 0.2 - 0.4  |
        +----------------+------------+
        | Unacceptable   | 0.0 - 0.2  |
        +----------------+------------+
        """
        scale = dict(
            lb=[0.8, 0.6, 0.4, 0.2, 0.0],
            ub=[1.0, 0.8, 0.6, 0.4, 0.2],
            interp=[
                "Excellent",
                "Very Good",
                "Good",
                "Questionable",
                "Unacceptable",
            ],
            scale_name="Regier",
        )
        return self.interpret(scale)
