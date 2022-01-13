from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from irrCAC.weights import Weights


class Test(TestCase):
    def setUp(self) -> None:
        # Test Weights using three categories.
        self.categories = [1, 2, 3]
        self.weights = Weights(self.categories)

    @pytest.fixture(autouse=True)
    def _pass_fixture(self, capsys):
        self.capsys = capsys

    def test_init_with_ndarray(self):
        w = Weights(np.array([1, 2, 3]))
        self.assertEqual(3, len(w.categ_vec))

    def test_init_with_df(self):
        data = {"r1": [1, 1, 2], "r2": [1, 2, 2], "r3": [1, 1, 1]}
        df = pd.DataFrame(data)
        w = Weights(df)
        self.assertEqual(3, len(w.categ_vec))

    def test_init_exception(self):
        with self.assertRaises(ValueError):
            _ = Weights("abc")

    def test_init_with_categorical(self):
        w = Weights(["a", "b", "c"])
        self.assertEqual(3, len(w.categ_vec))

    # def test_str(self):
    #     print(f'Weights for {len(self.categories)} categories.')
    #     expected_stdout = self.capsys.readouterr()
    #     print(self.weights)
    #     result_stdout = self.capsys.readouterr()
    #     self.assertEqual(expected_stdout, result_stdout)

    def test_identity(self):
        expected_weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        identity_weights = np.round(self.weights["identity"], 3)
        np.testing.assert_array_equal(expected_weights, identity_weights)

    def test_bipolar(self):
        expected_weights = np.array([[1, 0.667, 0], [0.667, 1, 0.667], [0, 0.667, 1]])
        bipolar_weights = np.round(self.weights["bipolar"], 3)
        np.testing.assert_array_equal(expected_weights, bipolar_weights)

    def test_circular(self):
        expected_weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        circular_weights = np.round(self.weights["circular"], 3)
        np.testing.assert_array_equal(expected_weights, circular_weights)

    def test_linear(self):
        expected_weights = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]])
        linear_weights = np.round(self.weights["linear"], 3)
        np.testing.assert_array_equal(expected_weights, linear_weights)

    def test_ordinal(self):
        expected_weights = np.array([[1, 0.667, 0], [0.667, 1, 0.667], [0, 0.667, 1]])
        ordinal_weights = np.round(self.weights["ordinal"], 3)
        np.testing.assert_array_equal(expected_weights, ordinal_weights)

    def test_quadratic(self):
        expected_weights = np.array([[1, 0.75, 0], [0.75, 1, 0.75], [0, 0.75, 1]])
        quadratic_weights = np.round(self.weights["quadratic"], 3)
        np.testing.assert_array_equal(expected_weights, quadratic_weights)

    def test_radical(self):
        expected_weights = np.array(
            [[1.0, 0.293, 0.0], [0.293, 1.0, 0.293], [0.0, 0.293, 1.0]]
        )
        radical_weights = np.round(self.weights["radical"], 3)
        np.testing.assert_array_equal(expected_weights, radical_weights)

    def test_ratio(self):
        expected_weights = np.array(
            [[1.0, 0.556, 0.0], [0.556, 1.0, 0.84], [0.0, 0.84, 1.0]]
        )
        ratio_weights = np.round(self.weights["ratio"], 3)
        np.testing.assert_array_equal(expected_weights, ratio_weights)

    def test_ratio_exception(self):
        w = Weights([0, 1, 2])
        with self.assertRaises(ValueError):
            _ = w["ratio"]
