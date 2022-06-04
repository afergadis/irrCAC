from unittest import TestCase

import numpy as np

from irrCAC.datasets import (
    raw_4raters,
    raw_5observers,
    raw_ben_gerry,
    raw_g1g2,
    raw_gender,
)
from irrCAC.raw import CAC


class TestBP(TestCase):
    def test_bp_kappa_raw4raters(self):
        data = raw_4raters()
        cac = CAC(data, digits=3)
        est = cac.bp()["est"]
        self.assertEqual(est["pa"], 0.818, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.200, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.773, "Wrong " "coefficient value.")
        self.assertEqual(est["se"], 0.145, "Wrong standard error.")
        self.assertEqual(
            est["confidence_interval"][0],
            0.454,
            "Wrong lower value for confidence interval.",
        )
        self.assertEqual(
            est["confidence_interval"][1],
            1,
            "Wrong upper value for confidence interval.",
        )
        self.assertEqual(round(est["p_value"], 5), 0.00012, "Wrong p value.")

    def test_bp_kappa_raw4raters_bipolar_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="bipolar", digits=3)
        est = cac.bp()["est"]
        self.assertEqual(est["pa"], 0.968, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.717, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.888)
        self.assertEqual(est["se"], 0.112, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.641, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0, "Wrong p-value.")

    def test_bp_kappa_raw4raters_circular_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="circular", digits=3)
        est = cac.bp()["est"]
        self.assertEqual(est["pa"], 0.902, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.447, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.824)
        self.assertEqual(est["se"], 0.137, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.522, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 4e-05, "Wrong p-value.")

    def test_bp_kappa_raw4raters_linear_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="linear", digits=3)
        est = cac.bp()["est"]
        self.assertEqual(est["pa"], 0.939, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.600, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.848)
        self.assertEqual(est["se"], 0.123, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.577, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 1e-05, "Wrong p-value.")

    def test_bp_kappa_raw4raters_ordinal_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="ordinal", digits=3)
        est = cac.bp()["est"]
        self.assertEqual(est["pa"], 0.968, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.720, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.886)
        self.assertEqual(est["se"], 0.114, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.636, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0, "Wrong p-value.")

    def test_bp_kappa_raw4raters_quadratic_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="quadratic", digits=3)
        est = cac.bp()["est"]
        self.assertEqual(est["pa"], 0.975, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.750, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.902)
        self.assertEqual(est["se"], 0.111, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.657, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0, "Wrong p-value.")

    def test_bp_kappa_raw4raters_radical_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="radical", digits=3)
        est = cac.bp()["est"]
        self.assertEqual(est["pa"], 0.897, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.452, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.813)
        self.assertEqual(est["se"], 0.133, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.520, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 4e-05, "Wrong p-value.")

    def test_bp_kappa_raw4raters_ratio_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="ratio", digits=3)
        est = cac.bp()["est"]
        self.assertEqual(est["pa"], 0.954, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.713, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.840)
        self.assertEqual(est["se"], 0.132, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.549, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 3e-05, "Wrong p-value.")

    def test_bp_kappa_raw4raters_custom_weights(self):
        data = raw_4raters()
        weights = np.array(
            [
                [1.00, 0.75, 0.50, 0.25, 0.00],
                [0.00, 1.00, 0.75, 0.50, 0.25],
                [0.25, 0.00, 1.00, 0.75, 0.50],
                [0.50, 0.25, 0.00, 1.00, 0.75],
                [0.75, 0.50, 0.25, 0.00, 1.00],
            ]
        )
        cac = CAC(data, weights=weights, digits=3)
        est = cac.bp()["est"]
        self.assertEqual(est["pa"], 0.886, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.500, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.773)
        self.assertEqual(est["se"], 0.145, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.454, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 1.2e-04, "Wrong p-value.")

    def test_bp_kappa_raw5observers(self):
        data = raw_5observers()
        cac = CAC(data, digits=3)
        est = cac.bp()["est"]
        self.assertEqual(est["pa"], 0.616, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.250, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.487, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.121, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.228, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.747, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00062, "Wrong p value.")

    def test_bp_kappa_raw_g1g2(self):
        data = raw_g1g2()
        cac = CAC(data, digits=3)
        est = cac.bp()["est"]
        self.assertEqual(est["pa"], 0.744, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.200, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.679, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.136, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.385, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.974, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00012, "Wrong p value.")

    def test_bp_kappa_raw_gender(self):
        data = raw_gender()
        cac = CAC(data, digits=3)
        est = cac.bp()["est"]
        self.assertEqual(est["pa"], 0.456, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.333, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.183, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.102, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], -0.035, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.402, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.04688, "Wrong p value.")

    def test_bp_kappa_raw_ben_gerry(self):
        data = raw_ben_gerry()
        cac = CAC(data, digits=3)
        est = cac.bp()["est"]
        self.assertEqual(est["pa"], 0.727, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.200, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.659, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.185, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.251, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00225, "Wrong p value.")
