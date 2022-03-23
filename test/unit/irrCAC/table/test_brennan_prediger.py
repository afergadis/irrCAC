from unittest import TestCase

import numpy as np

from irrCAC.datasets import table_cont3x3abstractors, table_cont4x4diagnosis
from irrCAC.table import CAC


class TestBP(TestCase):
    def setUp(self) -> None:
        self.cont3x3abstractors = table_cont3x3abstractors()
        self.cont4x4diagnosis = table_cont4x4diagnosis()

    def test_bp_cont3x3abstractors(self):
        cac = CAC(self.cont3x3abstractors, digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.835, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.742, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.928, "Wrong CI upper value.")

    def test_bp_cont3x3abstractors_bipolar_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="bipolar", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.901, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.845, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.957, "Wrong CI upper value.")

    def test_bp_cont3x3abstractors_circular_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="circular", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.835, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.742, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.928, "Wrong CI upper value.")

    def test_bp_cont3x3abstractors_linear_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="linear", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.876, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.806, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.946, "Wrong CI upper value.")

    def test_bp_cont3x3abstractors_ordinal_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="ordinal", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.901, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.845, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.957, "Wrong CI upper value.")

    def test_bp_cont3x3abstractors_quadratic_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="quadratic", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.918, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.871, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.964, "Wrong CI upper value.")

    def test_bp_cont3x3abstractors_radical_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="radical", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.855, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.773, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.937, "Wrong CI upper value.")

    def test_bp_cont3x3abstractors_ratio_weights(self):
        cac = CAC(self.cont3x3abstractors, weights="ratio", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.951, "Wrong coeff value.")
        self.assertEqual(est["confidence_interval"][0], 0.923, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.978, "Wrong CI upper value.")

    def test_bp_cont4x4diagnosis(self):
        cac = CAC(self.cont4x4diagnosis, digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.450, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.044, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.363, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.537, "Wrong CI upper value.")

    def test_bp_cont4x4diagnosis_bipolar_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="bipolar", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.264, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.075, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.115, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.412, "Wrong CI upper value.")

    def test_bp_cont4x4diagnosis_circular_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="circular", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.471, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.047, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.379, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.563, "Wrong CI upper value.")

    def test_bp_cont4x4diagnosis_linear_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="linear", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.347, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.060, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.229, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.465, "Wrong CI upper value.")

    def test_bp_cont4x4diagnosis_ordinal_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="ordinal", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.273, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.076, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.123, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.422, "Wrong CI upper value.")

    def test_bp_cont4x4diagnosis_quadratic_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="quadratic", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.236, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.085, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.069, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.403, "Wrong CI upper value.")

    def test_bp_cont4x4diagnosis_radical_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="radical", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.401, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.050, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.303, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.500, "Wrong CI upper value.")

    def test_bp_cont4x4diagnosis_ratio_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights="ratio", digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.310, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.077, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.158, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.462, "Wrong CI upper value.")

    def test_bp_cont4x4diagnosis_custom_weights(self):
        weights = np.array(
            [
                [1.00, 0.67, 0.33, 0.00],
                [0.00, 1.00, 0.67, 0.33],
                [0.00, 0.00, 1.00, 0.67],
                [0.33, 0.00, 0.00, 1.00],
            ]
        )
        cac = CAC(self.cont4x4diagnosis, weights=weights, digits=3)
        results = cac.bp()
        est = results["est"]
        self.assertEqual(est["coefficient_value"], 0.447, "Wrong coeff value.")
        self.assertEqual(est["se"], 0.048, "Wrong stderr.")
        self.assertEqual(est["confidence_interval"][0], 0.352, "Wrong CI lower value.")
        self.assertEqual(est["confidence_interval"][1], 0.542, "Wrong CI upper value.")
