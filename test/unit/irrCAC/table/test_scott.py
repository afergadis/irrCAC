from unittest import TestCase

from irrCAC.datasets import table_cont3x3abstractors, table_cont4x4diagnosis
from irrCAC.table import CAC


class TestScott(TestCase):
    def setUp(self) -> None:
        self.cont3x3abstractors = table_cont3x3abstractors()
        self.cont4x4diagnosis = table_cont4x4diagnosis()

    def test_scott_cont3x3abstractors(self):
        cac = CAC(self.cont3x3abstractors, digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.796, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.059, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.679, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.913, "Wrong CI upper value.")

    def test_scott_cont3x3abstractors_bipolar_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="bipolar", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.872, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.041, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.791, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.953, "Wrong CI upper value.")

    def test_scott_cont3x3abstractors_circular_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="circular", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.796, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.059, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.679, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.913, "Wrong CI upper value.")

    def test_scott_cont3x3abstractors_linear_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="linear", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.843, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.048, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.748, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.938, "Wrong CI upper value.")

    def test_scott_cont3x3abstractors_ordinal_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="ordinal", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.872, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.041, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.791, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.953, "Wrong CI upper value.")

    def test_scott_cont3x3abstractors_quadratic_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="quadratic", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.892, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.035, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.822, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.962, "Wrong CI upper value.")

    def test_scott_cont3x3abstractors_radical_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="radical", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.819, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.054, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.712, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.925, "Wrong CI upper value.")

    def test_scott_cont3x3abstractors_ratio_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="ratio", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.927, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.025, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.876, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.977, "Wrong CI upper value.")

    def test_scott_cont4x4diagnosis(self):
        cac = CAC(self.cont4x4diagnosis, digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.430, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.046, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.339, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.521, "Wrong CI upper value.")

    def test_scott_cont4x4diagnosis_bipolar_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="bipolar", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.386, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.061, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.265, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.506, "Wrong CI upper value.")

    def test_scott_cont4x4diagnosis_circular_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="circular", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.444, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.049, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.347, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.541, "Wrong CI upper value.")

    def test_scott_cont4x4diagnosis_linear_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="linear", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.406, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.054, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.300, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.511, "Wrong CI upper value.")

    def test_scott_cont4x4diagnosis_ordinal_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="ordinal", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.389, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.062, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.267, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.511, "Wrong CI upper value.")

    def test_scott_cont4x4diagnosis_quadratic_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="quadratic", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.382, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.066, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.252, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.512, "Wrong CI upper value.")

    def test_scott_cont4x4diagnosis_radical_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="radical", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.418, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.049, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.322, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.514, "Wrong CI upper value.")

    def test_scott_cont4x4diagnosis_ratio_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="ratio", digits=3)
        results = cac.scott()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.412, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.063, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.287, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.538, "Wrong CI upper value.")
