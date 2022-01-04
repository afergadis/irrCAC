from unittest import TestCase
from irrCAC.datasets import table_cont3x3abstractors, table_cont4x4diagnosis
from irrCAC.table import CAC


class TestPA(TestCase):
    def setUp(self) -> None:
        self.cont3x3abstractors = table_cont3x3abstractors()
        self.cont4x4diagnosis = table_cont4x4diagnosis()

    def test_pa2_cont3x3abstractors(self):
        cac = CAC(self.cont3x3abstractors)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.890, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.031, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.828,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.952,
            'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_bipolar_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='bipolar')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.963, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.010, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.943,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.984,
            'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_circular_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='circular')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.890, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.031, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.828,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.952,
            'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_linear_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='linear')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.945, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.016, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.914,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.976,
            'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_ordinal_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ordinal')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.963, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.010, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.943,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.984,
            'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_quadratic_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='quadratic')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.972, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.008, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.957,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.988,
            'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_radical_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='radical')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.922, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.022, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.878,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.966,
            'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_ratio_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ratio')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.982, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.005, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.972,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.992,
            'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis(self):
        cac = CAC(self.cont4x4diagnosis)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.587, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.033, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.522,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.652,
            'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_bipolar_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='bipolar')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.769, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.024, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.722,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.816,
            'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_circular_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='circular')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.735, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.023, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.690,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.781,
            'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_linear_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='linear')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.728, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.025, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.679,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.777,
            'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_ordinal_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='ordinal')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.773, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.024, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.726,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.820,
            'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_quadratic_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='quadratic')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.788, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.024, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.741,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.834,
            'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_radical_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='radical')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.673, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.027, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.620,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.727,
            'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_ratio_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='ratio')
        results = cac.pa2()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.786, 'Wrong coeff value.')
        self.assertEqual(round(est['se'], 3), 0.024, 'Wrong stderr.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.739,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.833,
            'Wrong CI upper value.')

