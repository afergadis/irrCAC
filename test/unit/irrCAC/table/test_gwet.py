from unittest import TestCase
from irrCAC.datasets import table_cont3x3abstractors, table_cont4x4diagnosis
from irrCAC.table import CAC


class TestGwet(TestCase):
    def setUp(self) -> None:
        self.cont3x3abstractors = table_cont3x3abstractors()
        self.cont4x4diagnosis = table_cont4x4diagnosis()

    def test_gwet_cont3x3abstractors(self):
        cac = CAC(self.cont3x3abstractors, digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.849, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.764, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.935, "Wrong CI upper value.")

    def test_gwet_cont3x3abstractors_bipolar_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="bipolar", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.925, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.881, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.969, "Wrong CI upper value.")

    def test_gwet_cont3x3abstractors_circular_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="circular", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.849, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.764, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.935, "Wrong CI upper value.")

    def test_gwet_cont3x3abstractors_linear_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="linear", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.900, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.842, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.958, "Wrong CI upper value.")

    def test_gwet_cont3x3abstractors_ordinal_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="ordinal", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.925, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.881, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.969, "Wrong CI upper value.")

    def test_gwet_cont3x3abstractors_quadratic_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="quadratic", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.940, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.905, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.976, "Wrong CI upper value.")

    def test_gwet_cont3x3abstractors_radical_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="radical", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.875, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.804, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.947, "Wrong CI upper value.")

    def test_gwet_cont3x3abstractors_ratio_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="ratio", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.963, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.942, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.985, "Wrong CI upper value.")

    def test_gwet_cont4x4diagnosis(self):
        cac = CAC(self.cont4x4diagnosis, digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.456, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.043, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.371, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.542, "Wrong CI upper value.")

    def test_gwet_cont4x4diagnosis_bipolar_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="bipolar", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.315, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.068, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.181, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.450, "Wrong CI upper value.")

    def test_gwet_cont4x4diagnosis_circular_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="circular", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.488, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.045, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.399, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.578, "Wrong CI upper value.")

    def test_gwet_cont4x4diagnosis_linear_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="linear", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.377, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.056, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.267, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.488, "Wrong CI upper value.")

    def test_gwet_cont4x4diagnosis_ordinal_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="ordinal", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.324, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.069, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.189, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.459, "Wrong CI upper value.")

    def test_gwet_cont4x4diagnosis_quadratic_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="quadratic", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.299, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.075, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.150, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.447, "Wrong CI upper value.")

    def test_gwet_cont4x4diagnosis_radical_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="radical", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.418, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.048, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.323, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.513, "Wrong CI upper value.")

    def test_gwet_cont4x4diagnosis_ratio_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="ratio", digits=3)
        results = cac.gwet()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.359, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.070, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.221, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.496, "Wrong CI upper value.")
