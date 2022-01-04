from unittest import TestCase
from irrCAC.datasets import table_cont3x3abstractors, table_cont4x4diagnosis
from irrCAC.table import CAC


class TestGwet(TestCase):
    def setUp(self) -> None:
        self.cont3x3abstractors = table_cont3x3abstractors()
        self.cont4x4diagnosis = table_cont4x4diagnosis()

    def test_gwet_cont3x3abstractors(self):
        cac = CAC(self.cont3x3abstractors)
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.849, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.764,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.935,
            'Wrong CI upper value.')

    def test_gwet_cont3x3abstractors_bipolar_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='bipolar')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.925, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.881,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.969,
            'Wrong CI upper value.')

    def test_gwet_cont3x3abstractors_circular_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='circular')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.849, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.764,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.935,
            'Wrong CI upper value.')

    def test_gwet_cont3x3abstractors_linear_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='linear')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.900, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.842,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.958,
            'Wrong CI upper value.')

    def test_gwet_cont3x3abstractors_ordinal_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ordinal')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.925, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.881,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.969,
            'Wrong CI upper value.')

    def test_gwet_cont3x3abstractors_quadratic_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='quadratic')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.940, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.905,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.976,
            'Wrong CI upper value.')

    def test_gwet_cont3x3abstractors_radical_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='radical')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.875, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.804,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.947,
            'Wrong CI upper value.')

    def test_gwet_cont3x3abstractors_ratio_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ratio')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.963, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.942,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.985,
            'Wrong CI upper value.')

    def test_gwet_cont4x4diagnosis(self):
        cac = CAC(self.cont4x4diagnosis)
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.456, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.043, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.371,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.542,
            'Wrong CI upper value.')

    def test_gwet_cont4x4diagnosis_bipolar_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='bipolar')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.315, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.068, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.181,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.450,
            'Wrong CI upper value.')

    def test_gwet_cont4x4diagnosis_circular_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='circular')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.488, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.045, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.399,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.578,
            'Wrong CI upper value.')

    def test_gwet_cont4x4diagnosis_linear_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='linear')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.377, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.056, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.267,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.488,
            'Wrong CI upper value.')

    def test_gwet_cont4x4diagnosis_ordinal_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='ordinal')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.324, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.069, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.189,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.459,
            'Wrong CI upper value.')

    def test_gwet_cont4x4diagnosis_quadratic_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='quadratic')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.299, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.075, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.150,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.447,
            'Wrong CI upper value.')

    def test_gwet_cont4x4diagnosis_radical_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='radical')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.418, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.048, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.323,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.513,
            'Wrong CI upper value.')

    def test_gwet_cont4x4diagnosis_ratio_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='ratio')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.359, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.070, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.221,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.496,
            'Wrong CI upper value.')

