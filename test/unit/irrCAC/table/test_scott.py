from unittest import TestCase
from irrCAC.datasets import table_cont3x3abstractors, table_cont4x4diagnosis
from irrCAC.table import CAC


class TestScott(TestCase):
    def setUp(self) -> None:
        self.cont3x3abstractors = table_cont3x3abstractors()
        self.cont4x4diagnosis = table_cont4x4diagnosis()

    def test_scott_cont3x3abstractors(self):
        cac = CAC(self.cont3x3abstractors)
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.796, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.059, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.679,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.913,
            'Wrong CI upper value.')

    def test_scott_cont3x3abstractors_bipolar_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='bipolar')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.872, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.041, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.791,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.953,
            'Wrong CI upper value.')

    def test_scott_cont3x3abstractors_circular_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='circular')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.796, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.059, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.679,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.913,
            'Wrong CI upper value.')

    def test_scott_cont3x3abstractors_linear_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='linear')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.843, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.048, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.748,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.938,
            'Wrong CI upper value.')

    def test_scott_cont3x3abstractors_ordinal_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ordinal')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.872, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.041, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.791,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.953,
            'Wrong CI upper value.')

    def test_scott_cont3x3abstractors_quadratic_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='quadratic')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.892, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.035, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.822,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.962,
            'Wrong CI upper value.')

    def test_scott_cont3x3abstractors_radical_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='radical')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.819, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.054, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.712,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.925,
            'Wrong CI upper value.')

    def test_scott_cont3x3abstractors_ratio_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ratio')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.927, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.025, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.876,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.977,
            'Wrong CI upper value.')

    def test_scott_cont4x4diagnosis(self):
        cac = CAC(self.cont4x4diagnosis)
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.430, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.046, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.339,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.521,
            'Wrong CI upper value.')

    def test_scott_cont4x4diagnosis_bipolar_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='bipolar')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.386, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.061, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.265,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.506,
            'Wrong CI upper value.')

    def test_scott_cont4x4diagnosis_circular_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='circular')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.444, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.049, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.347,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.541,
            'Wrong CI upper value.')

    def test_scott_cont4x4diagnosis_linear_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='linear')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.406, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.054, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.300,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.511,
            'Wrong CI upper value.')

    def test_scott_cont4x4diagnosis_ordinal_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='ordinal')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.389, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.062, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.267,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.511,
            'Wrong CI upper value.')

    def test_scott_cont4x4diagnosis_quadratic_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='quadratic')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.382, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.066, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.252,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.512,
            'Wrong CI upper value.')

    def test_scott_cont4x4diagnosis_radical_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='radical')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.418, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.049, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.322,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.514,
            'Wrong CI upper value.')

    def test_scott_cont4x4diagnosis_ratio_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='ratio')
        results = cac.scott()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.412, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.063, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.287,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.538,
            'Wrong CI upper value.')

