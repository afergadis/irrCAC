from unittest import TestCase
from irrCAC.datasets import table_cont3x3abstractors, table_cont4x4diagnosis
from irrCAC.table import CAC


class TestCohen(TestCase):
    def setUp(self) -> None:
        self.cont3x3abstractors = table_cont3x3abstractors()
        self.cont4x4diagnosis = table_cont4x4diagnosis()

    def test_cohen_cont3x3abstractors(self):
        cac = CAC(self.cont3x3abstractors)
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.796, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 2), 0.68,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.913,
            'Wrong CI upper value.')

    def test_cohen_cont3x3abstractors_bipolar_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='bipolar')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.872, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.792,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.953,
            'Wrong CI upper value.')

    def test_cohen_cont3x3abstractors_circular_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='circular')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.796, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 2), 0.68,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.913,
            'Wrong CI upper value.')

    def test_cohen_cont3x3abstractors_linear_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='linear')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.843, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.748,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.938,
            'Wrong CI upper value.')

    def test_cohen_cont3x3abstractors_ordinal_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ordinal')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.872, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.792,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.953,
            'Wrong CI upper value.')

    def test_cohen_cont3x3abstractors_quadratic_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='quadratic')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.892, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.822,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.962,
            'Wrong CI upper value.')

    def test_cohen_cont3x3abstractors_radical_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='radical')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.819, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.712,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.925,
            'Wrong CI upper value.')

    def test_cohen_cont3x3abstractors_ratio_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ratio')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.927, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.876,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.977,
            'Wrong CI upper value.')

    def test_cohen_cont4x4diagnosis(self):
        cac = CAC(self.cont4x4diagnosis)
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.432, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.046, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.341,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.522,
            'Wrong CI upper value.')

    def test_cohen_cont4x4diagnosis_bipolar_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='bipolar')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.387, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.061, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.267,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.507,
            'Wrong CI upper value.')

    def test_cohen_cont4x4diagnosis_circular_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='circular')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.446, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.049, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.350,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.542,
            'Wrong CI upper value.')

    def test_cohen_cont4x4diagnosis_linear_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='linear')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.407, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.053, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.302,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.512,
            'Wrong CI upper value.')

    def test_cohen_cont4x4diagnosis_ordinal_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='ordinal')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.390, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.062, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.269,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.512,
            'Wrong CI upper value.')

    def test_cohen_cont4x4diagnosis_quadratic_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='quadratic')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.383, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.066, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.254,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.513,
            'Wrong CI upper value.')

    def test_cohen_cont4x4diagnosis_radical_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='radical')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.419, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.048, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.324,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.515,
            'Wrong CI upper value.')

    def test_cohen_cont4x4diagnosis_ratio_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='ratio')
        results = cac.cohen()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.413, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.063, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.288,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.538,
            'Wrong CI upper value.')

