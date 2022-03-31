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


class TestConger(TestCase):
    def test_conger_kappa_raw4raters(self):
        data = raw_4raters()
        cac = CAC(data, digits=3)
        est = cac.conger()["est"]
        self.assertEqual(est["pa"], 0.818, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.233, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.763, "Wrong " "coefficient value.")
        self.assertEqual(
            est["confidence_interval"][0],
            0.435,
            "Wrong lower value for confidence interval.",
        )
        self.assertEqual(
            est["confidence_interval"][1],
            1,
            "Wrong upper value for confidence interval.",
        )
        self.assertEqual(round(est["p_value"], 5), 0.00034, "Wrong p value.")

    def test_conger_kappa_raw4raters_bipolar_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="bipolar", digits=3)
        est = cac.conger()["est"]
        self.assertEqual(est["pa"], 0.968, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.796, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.845)
        self.assertEqual(est["se"], 0.142, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.532, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 1e-04, "Wrong p-value.")

    def test_conger_kappa_raw4raters_circular_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="circular", digits=3)
        est = cac.conger()["est"]
        self.assertEqual(est["pa"], 0.902, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.499, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.805)
        self.assertEqual(est["se"], 0.147, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.481, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.0002, "Wrong p-value.")

    def test_conger_kappa_raw4raters_linear_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="linear", digits=3)
        est = cac.conger()["est"]
        self.assertEqual(est["pa"], 0.939, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.675, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.814)
        self.assertEqual(est["se"], 0.145, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.494, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00016, "Wrong p-value.")

    def test_conger_kappa_raw4raters_ordinal_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="ordinal", digits=3)
        est = cac.conger()["est"]
        self.assertEqual(est["pa"], 0.968, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.796, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.844)
        self.assertEqual(est["se"], 0.144, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.526, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00011, "Wrong p-value.")

    def test_conger_kappa_raw4raters_quadratic_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="quadratic", digits=3)
        est = cac.conger()["est"]
        self.assertEqual(est["pa"], 0.975, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.827, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.858)
        self.assertEqual(est["se"], 0.144, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.541, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 9e-05, "Wrong p-value.")

    def test_conger_kappa_raw4raters_radical_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="radical", digits=3)
        est = cac.conger()["est"]
        self.assertEqual(est["pa"], 0.897, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.515, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.788)
        self.assertEqual(est["se"], 0.146, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.466, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 2.2e-04, "Wrong p-value.")

    def test_conger_kappa_raw4raters_ratio_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="ratio", digits=3)
        est = cac.conger()["est"]
        self.assertEqual(est["pa"], 0.954, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.756, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.812)
        self.assertEqual(est["se"], 0.149, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.485, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 2e-04, "Wrong p-value.")

    def test_conger_kappa_raw4raters_custom_weights(self):
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
        est = cac.conger()["est"]
        self.assertEqual(est["pa"], 0.886, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.521, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.763)
        self.assertEqual(est["se"], 0.149, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.435, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 3.4e-04, "Wrong p-value.")

    def test_conger_kappa_raw5observers(self):
        data = raw_5observers()
        cac = CAC(data, digits=3)
        est = cac.conger()["est"]
        self.assertEqual(est["pa"], 0.616, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.276, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.469, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.117, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.219, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.719, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00126, "Wrong p value.")

    def test_conger_kappa_raw_g1g2(self):
        data = raw_g1g2()
        cac = CAC(data, digits=3)
        est = cac.conger()["est"]
        self.assertEqual(est["pa"], 0.744, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.249, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.659, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.146, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.343, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.974, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00059, "Wrong p value.")

    def test_conger_kappa_raw_gender(self):
        data = raw_gender()
        cac = CAC(data, digits=3)
        est = cac.conger()["est"]
        self.assertEqual(est["pa"], 0.456, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.347, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.166, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.121, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], -0.094, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.426, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.19309, "Wrong p value.")

    def test_conger_kappa_raw_ben_gerry(self):
        data = raw_ben_gerry()
        cac = CAC(data, digits=3)
        est = cac.conger()["est"]
        self.assertEqual(est["pa"], 0.727, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.205, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.657, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.179, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.262, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00374, "Wrong p value.")
