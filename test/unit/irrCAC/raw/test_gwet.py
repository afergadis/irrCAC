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


class TestGwet(TestCase):
    def test_gwet_raw4raters(self):
        data = raw_4raters()
        cac = CAC(data, digits=3)
        est = cac.gwet()["est"]
        self.assertEqual(est["pa"], 0.818, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.190, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.775, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.143, "Wrong standard error value.")
        self.assertEqual(est["confidence_interval"][0], 0.461, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00021, "Wrong p-value.")

    def test_gwet_raw4raters_bipolar_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="bipolar", digits=3)
        est = cac.gwet()["est"]
        self.assertEqual(est["pa"], 0.968, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.682, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.900, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.106, "Wrong standard error value.")
        self.assertEqual(est["confidence_interval"][0], 0.667, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value")
        self.assertEqual(round(est["p_value"], 6), 4e-6, "Wrong p value.")

    def test_gwet_raw4raters_circular_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="circular", digits=3)
        est = cac.gwet()["est"]
        self.assertEqual(est["pa"], 0.902, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.426, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.830, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.133, "Wrong standard error value.")
        self.assertEqual(est["confidence_interval"][0], 0.538, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value")
        self.assertEqual(round(est["p_value"], 5), 6e-5, "Wrong p value.")

    def test_gwet_raw4raters_linear_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="linear", digits=3)
        est = cac.gwet()["est"]
        self.assertEqual(est["pa"], 0.939, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.571, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.859, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.117, "Wrong standard error value.")
        self.assertEqual(est["confidence_interval"][0], 0.600, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value")
        self.assertEqual(round(est["p_value"], 5), 2e-5, "Wrong p value.")

    def test_gwet_raw4raters_ordinal_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="ordinal", digits=3)
        est = cac.gwet()["est"]
        self.assertEqual(est["pa"], 0.968, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.685, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.899, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.107, "Wrong standard error value.")
        self.assertEqual(est["confidence_interval"][0], 0.664, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value")
        self.assertEqual(round(est["p_value"], 6), 4e-6, "Wrong p value.")

    def test_gwet_raw4raters_quadratic_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="quadratic", digits=3)
        est = cac.gwet()["est"]
        self.assertEqual(est["pa"], 0.975, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.714, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.914, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.104, "Wrong standard error value.")
        self.assertEqual(est["confidence_interval"][0], 0.685, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 6), 3e-06, "Wrong p value.")

    def test_gwet_raw4raters_radical_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="radical", digits=3)
        est = cac.gwet()["est"]
        self.assertEqual(est["pa"], 0.897, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.430, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.820, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.128, "Wrong standard error value.")
        self.assertEqual(est["confidence_interval"][0], 0.537, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 5e-05, "Wrong p value.")

    def test_gwet_raw4raters_ratio_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="ratio", digits=3)
        est = cac.gwet()["est"]
        self.assertEqual(est["pa"], 0.954, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.678, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.857, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.122, "Wrong standard error value.")
        self.assertEqual(est["confidence_interval"][0], 0.589, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 2e-05, "Wrong p value.")

    def test_gwet_raw4raters_custom_weights(self):
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
        est = cac.gwet()["est"]
        self.assertEqual(est["pa"], 0.886, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.476, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.783, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.138, "Wrong standard error value.")
        self.assertEqual(est["confidence_interval"][0], 0.479, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 1.5e-04, "Wrong p value.")

    def test_gwet_raw5observers(self):
        data = raw_5observers()
        cac = CAC(data, digits=3)
        est = cac.gwet()["est"]
        self.assertEqual(est["pa"], 0.616, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.236, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.497, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.122, "Wrong standard error value.")
        self.assertEqual(est["confidence_interval"][0], 0.235, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.758, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00114, "Wrong p value.")

    def test_gwet_raw_g1g2(self):
        data = raw_g1g2()
        cac = CAC(data, digits=3)
        est = cac.gwet()["est"]
        self.assertEqual(est["pa"], 0.744, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.187, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.685, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.134, "Wrong standard error value.")
        self.assertEqual(est["confidence_interval"][0], 0.396, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.973, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00019, "Wrong p value.")

    def test_gwet_raw_gender(self):
        data = raw_gender()
        cac = CAC(data, digits=3)
        est = cac.gwet()["est"]
        self.assertEqual(est["pa"], 0.456, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.312, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.208, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.091, "Wrong standard error value.")
        self.assertEqual(est["confidence_interval"][0], 0.014, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.403, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.03776, "Wrong p value.")

    def test_gwet_raw_ben_gerry(self):
        data = raw_ben_gerry()
        cac = CAC(data, digits=3)
        est = cac.gwet()["est"]
        self.assertEqual(est["pa"], 0.727, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.194, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.661, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.184, "Wrong standard error value.")
        self.assertEqual(est["confidence_interval"][0], 0.257, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00420, "Wrong p value.")
