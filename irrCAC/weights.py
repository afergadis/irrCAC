""" A set of predefined weight schemes that can be used with the agreement \
coefficients.
"""

import numpy as np
import pandas as pd


class Weights:
    """Methods for computing weights for a set of categories.

    The class can compute weights, using the methods:

    * bipolar
    * circular
    * identity
    * linear
    * ordinal
    * quadratic
    * radial
    * ratio
    """

    def __init__(self, categories):
        """Initialize a `Weights` class based on the rating categories.

        Parameters
        ----------
        categories : list, array-like or data frame
            A mandatory parameter representing the vector of all possible
            ratings. If the `categories` has numerical values, the vector is
            sorted. If the `categories` are categorical labels, the vector is
            a sequence of numbers in range `1…len(categories)`.

        Raises
        ------
        ValueError

            * The provided value for `categories` is not one of a list, a numpy
              array or a pandas dataframe.
            * Giving an unknown name of weights type.
        """
        if isinstance(categories, list):
            self.q = len(categories)
        elif isinstance(categories, (np.ndarray, pd.DataFrame)):
            self.q = categories.shape[-1]
        else:
            raise ValueError(
                "Valid input for `categories` is one of "
                "list, numpy array, or pandas data frame."
            )
        if all(isinstance(n, (int, float)) for n in categories):
            self.categ_vec = sorted(categories)
        else:
            self.categ_vec = list(range(1, len(categories) + 1))
        self.xmin, self.xmax = min(self.categ_vec), max(self.categ_vec)

    def __getitem__(self, item):
        if item == "bipolar":
            return self.bipolar()
        elif item == "circular":
            return self.circular()
        elif item == "identity":
            return self.identity()
        elif item == "linear":
            return self.linear()
        elif item == "ordinal":
            return self.ordinal()
        elif item == "quadratic":
            return self.quadratic()
        elif item == "radical":
            return self.radical()
        elif item == "ratio":
            return self.ratio()
        else:
            raise ValueError(f'"{item} is an unknown type of weights.')

    def __str__(self):
        return f"Weights for {self.q} categories."

    def bipolar(self):
        r"""Function for computing the Bipolar Weights

        Bipolar weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where :math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = \frac{(k - l)^2}{
                ((k + l - 2 \cdot \min(c))(2 \cdot \max(c) - k - l))}

        where :math:`c \in \mathbb{R}^q` is the vector of the `categories`.

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of bipolar weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                if k != el:
                    weights[k][el] = pow(self.categ_vec[k] - self.categ_vec[el], 2) / (
                        (self.categ_vec[k] + self.categ_vec[el] - 2 * self.xmin)
                        * (2 * self.xmax - self.categ_vec[k] - self.categ_vec[el])
                    )
                else:
                    weights[k][el] = 0
        weights = 1 - weights / np.max(weights)
        return weights

    def circular(self):
        r"""Function for computing the Circular Weights

        Circular weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where :math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = \sin{(\frac{\pi (k - l)}{(\max(c) - \min(c) + 1)})^2}

        where :math:`c \in \mathbb{R}^q` is the vector of the `categories`.

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of circular weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        U = self.xmax - self.xmin + 1
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = pow(
                    np.sin(np.pi * (self.categ_vec[k] - self.categ_vec[el]) / U), 2
                )
        weights = 1 - weights / np.max(weights)
        return weights

    def identity(self):
        r"""Function for computing the Identity Weights.

        The identity weighted matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where :math:`q` is the number of `categories`, is the same as to calculate the
        coefficients without weights (unweighted). The weights are defined as
        :math:`w_{kk}=1, (k=1, \ldots, q)` and
        :math:`w_{kl}=0, (k,l=1, \ldots, q)` if :math:`k \neq l`.

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of identity weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        return weights

    def linear(self):
        r"""Function for computing the Linear Weights.

        Linear weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = 1 - \frac{|k - l|}{q - 1}

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of linear weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = 1 - abs(self.categ_vec[k] - self.categ_vec[el]) / abs(
                    self.xmax - self.xmin
                )
        return weights

    def ordinal(self):
        r"""Function for computing the Ordinal Weights

        Ordinal weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where :math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = \frac{(\max(k, l) - \min(k, l) + 1) \cdot
                     (\max(k, l) - \min(k, l))}{2}

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of ordinal weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                nkl = max(k, el) - min(k, el) + 1
                weights[k][el] = nkl * (nkl - 1) / 2
        weights = 1 - weights / np.max(weights)
        return weights

    def quadratic(self):
        r"""Function for computing the Quadratic Weights

        Quadratic weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where :math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = 1 - \frac{(k - l)^2}{(q - 1)^2}

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of quadratic weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        diff = self.xmax - self.xmin
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = 1 - pow(
                    (self.categ_vec[k] - self.categ_vec[el]) / diff, 2
                )
        return weights

    def radical(self):
        r"""Function for computing the Radical Weights

        Radical weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`,
        where :math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = 1 - \frac{|k - l|^{1/2}}{(\max(c) - \min(c))^{1/2}}

        where :math:`c \in \mathbb{R}^q` is the vector of the `categories`.

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of radical weights to be used for calculating the
            weighted coefficients.
        """
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = 1 - np.sqrt(
                    abs(self.categ_vec[k] - self.categ_vec[el])
                ) / np.sqrt(abs(self.xmax - self.xmin))
        return weights

    def ratio(self):
        r"""Function for computing the Ratio Weights

        Ratio weights of a matrix :math:`\mathbf{W} \in \mathbb{R}^{q \times q}`, where
        :math:`q` is the number of `categories`, are defined for each cell
        :math:`w_{kl}, (k,l=1, \ldots, q)` by

        .. math::
            w_{kl} = 1 - (\frac{k - l}{k + l})^2/(
                \frac{\max(c) - \min(c)}{\max(c) - \min(c)})^2

        where :math:`c \in \mathbb{R}^q` is the vector of the `categories`.

        Returns
        -------
        :math:`\mathbb{R}^{q \times q}` matrix
            A square matrix of ratio weights to be used for calculating the
            weighted coefficients.

        Raises
        ------
        ValueError
            In cases we have the 0 as a category. This will produce a division
            by zero error.
        """
        if 0 in self.categ_vec:
            raise ValueError(
                "You have 0 as a category. Please do not use"
                " 0 as a category because it produce a"
                " division by 0."
            )
        weights = np.eye(self.q)
        for k in range(self.q):
            for el in range(self.q):
                weights[k][el] = 1 - (
                    pow(
                        (self.categ_vec[k] - self.categ_vec[el])
                        / (self.categ_vec[k] + self.categ_vec[el]),
                        2,
                    )
                ) / pow((self.xmax - self.xmin) / (self.xmax + self.xmin), 2)
        return weights
