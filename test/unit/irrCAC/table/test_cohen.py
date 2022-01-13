from unittest import TestCase

from irrCAC.datasets import table_cont3x3abstractors, table_cont4x4diagnosis
from irrCAC.table import CAC


class TestCohen(TestCase):
    def setUp(self) -> None:
        self.cont3x3abstractors = table_cont3x3abstractors()
        self.cont4x4diagnosis = table_cont4x4diagnosis()

    def test_cohen_cont3x3abstractors(self):
        cac = CAC(self.cont3x3abstractors, digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.796, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.68, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.913, "Wrong CI upper value.")

    def test_cohen_cont3x3abstractors_bipolar_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="bipolar", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.872, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.791, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.953, "Wrong CI upper value.")

    def test_cohen_cont3x3abstractors_circular_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="circular", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.796, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.68, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.913, "Wrong CI upper value.")

    def test_cohen_cont3x3abstractors_linear_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="linear", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.843, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.748, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.938, "Wrong CI upper value.")

    def test_cohen_cont3x3abstractors_ordinal_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="ordinal", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.872, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.791, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.953, "Wrong CI upper value.")

    def test_cohen_cont3x3abstractors_quadratic_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="quadratic", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.892, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.822, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.962, "Wrong CI upper value.")

    def test_cohen_cont3x3abstractors_radical_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="radical", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.819, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.712, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.925, "Wrong CI upper value.")

    def test_cohen_cont3x3abstractors_ratio_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="ratio", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.927, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.876, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.977, "Wrong CI upper value.")

    def test_cohen_cont4x4diagnosis(self):
        cac = CAC(self.cont4x4diagnosis, digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.432, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.046, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.341, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.522, "Wrong CI upper value.")

    def test_cohen_cont4x4diagnosis_bipolar_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="bipolar", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.387, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.061, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.267, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.507, "Wrong CI upper value.")

    def test_cohen_cont4x4diagnosis_circular_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="circular", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.446, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.049, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.350, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.542, "Wrong CI upper value.")

    def test_cohen_cont4x4diagnosis_linear_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="linear", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.407, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.053, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.302, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.512, "Wrong CI upper value.")

    def test_cohen_cont4x4diagnosis_ordinal_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="ordinal", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.390, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.062, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.269, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.512, "Wrong CI upper value.")

    def test_cohen_cont4x4diagnosis_quadratic_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="quadratic", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.383, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.066, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.254, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.513, "Wrong CI upper value.")

    def test_cohen_cont4x4diagnosis_radical_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="radical", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.419, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.048, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.324, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.515, "Wrong CI upper value.")

    def test_cohen_cont4x4diagnosis_ratio_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="ratio", digits=3)
        results = cac.cohen()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.413, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.063, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.288, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.538, "Wrong CI upper value.")
