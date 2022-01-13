from unittest import TestCase

from irrCAC.datasets import (
    raw_4raters,
    raw_5observers,
    raw_ben_gerry,
    raw_g1g2,
    raw_gender,
)
from irrCAC.raw import CAC


class TestKrippendorff(TestCase):
    def test_krippendorff_kappa_raw4raters(self):
        data = raw_4raters()
        cac = CAC(data, digits=3)
        est = cac.krippendorff()["est"]
        self.assertEqual(est["pa"], 0.805, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.240, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.743, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.146, "Wrong standard error.")
        self.assertEqual(
            est["confidence_interval"][0],
            0.419,
            "Wrong lower value for confidence interval.",
        )
        self.assertEqual(
            est["confidence_interval"][1],
            1,
            "Wrong upper value for confidence interval.",
        )
        self.assertEqual(round(est["p_value"], 5), 0.00046, "Wrong p value.")

    def test_krippendorff_kappa_raw4raters_bipolar_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="bipolar", digits=3)
        est = cac.krippendorff()["est"]
        self.assertEqual(est["pa"], 0.966, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.794, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.835)
        self.assertEqual(est["se"], 0.128, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.549, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 7e-05, "Wrong p-value.")

    def test_krippendorff_kappa_raw4raters_circular_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="circular", digits=3)
        est = cac.krippendorff()["est"]
        self.assertEqual(est["pa"], 0.895, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.502, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.790)
        self.assertEqual(est["se"], 0.141, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.476, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00023, "Wrong p-value.")

    def test_krippendorff_kappa_raw4raters_linear_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="linear", digits=3)
        est = cac.krippendorff()["est"]
        self.assertEqual(est["pa"], 0.935, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.674, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.800)
        self.assertEqual(est["se"], 0.135, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.499, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 0.00015, "Wrong p-value.")

    def test_krippendorff_kappa_raw4raters_ordinal_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="ordinal", digits=3)
        est = cac.krippendorff()["est"]
        self.assertEqual(est["pa"], 0.966, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.795, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.834)
        self.assertEqual(est["se"], 0.131, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.542, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 8e-5, "Wrong p-value.")

    def test_krippendorff_kappa_raw4raters_quadratic_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="quadratic", digits=3)
        est = cac.krippendorff()["est"]
        self.assertEqual(est["pa"], 0.974, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.825, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.849)
        self.assertEqual(est["se"], 0.129, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.561, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 6e-5, "Wrong p-value.")

    def test_krippendorff_kappa_raw4raters_radical_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="radical", digits=3)
        est = cac.krippendorff()["est"]
        self.assertEqual(est["pa"], 0.890, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.517, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.772)
        self.assertEqual(est["se"], 0.140, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.461, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 25e-5, "Wrong p-value.")

    def test_krippendorff_kappa_raw4raters_ratio_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights="ratio", digits=3)
        est = cac.krippendorff()["est"]
        self.assertEqual(est["pa"], 0.951, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.757, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.797)
        self.assertEqual(est["se"], 0.140, "Wrong standard error.")
        self.assertEqual(est["confidence_interval"][0], 0.484, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 1, "Wrong CI upper value.")
        self.assertEqual(round(est["p_value"], 5), 2e-4, "Wrong p-value.")

    def test_krippendorff_kappa_raw5observers(self):
        data = raw_5observers()
        cac = CAC(data, digits=3)
        est = cac.krippendorff()["est"]
        self.assertEqual(est["pa"], 0.627, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.282, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.481, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.114, "Wrong standard error.")
        self.assertEqual(
            est["confidence_interval"][0],
            0.237,
            "Wrong lower value for confidence interval.",
        )
        self.assertEqual(
            est["confidence_interval"][1],
            0.724,
            "Wrong upper value for confidence interval.",
        )
        self.assertEqual(round(est["p_value"], 5), 0.00083, "Wrong p value.")

    def test_krippendorff_kappa_raw_g1g2(self):
        data = raw_g1g2()
        cac = CAC(data, digits=3)
        est = cac.krippendorff()["est"]
        self.assertEqual(est["pa"], 0.728, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.261, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.632, "Wrong " "coefficient value.")
        self.assertEqual(est["se"], 0.150, "Wrong standard error.")
        self.assertEqual(
            est["confidence_interval"][0],
            0.306,
            "Wrong lower value for confidence interval.",
        )
        self.assertEqual(
            est["confidence_interval"][1],
            0.958,
            "Wrong upper value for confidence interval.",
        )
        self.assertEqual(round(est["p_value"], 5), 0.00119, "Wrong p value.")

    def test_krippendorff_kappa_raw_gender(self):
        data = raw_gender()
        cac = CAC(data, digits=3)
        est = cac.krippendorff()["est"]
        self.assertEqual(est["pa"], 0.465, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.375, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.143, "Wrong " "coefficient value.")
        self.assertEqual(est["se"], 0.141, "Wrong standard error.")
        self.assertEqual(
            est["confidence_interval"][0],
            -0.158,
            "Wrong lower value for confidence interval.",
        )
        self.assertEqual(
            round(est["confidence_interval"][1], 3),
            0.445,
            "Wrong upper value for confidence interval.",
        )
        self.assertEqual(round(est["p_value"], 5), 0.32472, "Wrong p value.")

    def test_krippendorff_kappa_raw_ben_gerry(self):
        data = raw_ben_gerry()
        cac = CAC(data, digits=3)
        est = cac.krippendorff()["est"]
        self.assertEqual(est["pa"], 0.740, "Wrong pa value.")
        self.assertEqual(est["pe"], 0.223, "Wrong pe value.")
        self.assertEqual(est["coefficient_value"], 0.665, "Wrong coefficient value.")
        self.assertEqual(est["se"], 0.180, "Wrong standard error.")
        self.assertEqual(
            est["confidence_interval"][0],
            0.264,
            "Wrong lower value for confidence interval.",
        )
        self.assertEqual(
            est["confidence_interval"][1],
            1,
            "Wrong upper value for confidence interval.",
        )
        self.assertEqual(round(est["p_value"], 5), 0.00416, "Wrong p value.")
