from unittest import TestCase
from irrCAC.datasets import table_cont3x3abstractors, table_cont4x4diagnosis
from irrCAC.table import CAC


class TestPA(TestCase):
    def setUp(self) -> None:
        self.cont3x3abstractors = table_cont3x3abstractors()
        self.cont4x4diagnosis = table_cont4x4diagnosis()

    def test_pa2_cont3x3abstractors(self):
        cac = CAC(self.cont3x3abstractors, digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.890, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.031, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.828, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.952, 'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_bipolar_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='bipolar', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.963, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.010, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.943, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.984, 'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_circular_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='circular', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.890, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.031, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.828, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.952, 'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_linear_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='linear', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.945, 'Wrong coeff value.')
        self.assertEqual(
            est['confidence_interval'][0], 0.914, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.976, 'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_ordinal_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ordinal', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.963, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.010, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.943, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.984, 'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_quadratic_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='quadratic', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.972, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.008, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.957, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.988, 'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_radical_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='radical', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.922, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.022, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.878, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.966, 'Wrong CI upper value.')

    def test_pa2_cont3x3abstractors_ratio_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ratio', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.982, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.005, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.972, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.992, 'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis(self):
        cac = CAC(self.cont4x4diagnosis, digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.587, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.033, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.522, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.652, 'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_bipolar_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='bipolar', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.769, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.024, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.722, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.816, 'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_circular_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='circular', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.735, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.023, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.690, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.781, 'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_linear_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='linear', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.728, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.025, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.679, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.777, 'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_ordinal_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='ordinal', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.773, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.024, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.726, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.820, 'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_quadratic_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='quadratic', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.788, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.024, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.741, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.834, 'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_radical_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='radical', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.673, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.027, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.620, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.727, 'Wrong CI upper value.')

    def test_pa2_cont4x4diagnosis_ratio_weights(self):
        cac = CAC(self.cont4x4diagnosis, weights='ratio', digits=3)
        results = cac.pa2()
        est = results['est']
        self.assertEqual(est['coefficient_value'], 0.786, 'Wrong coeff value.')
        self.assertEqual(est['se'], 0.024, 'Wrong stderr.')
        self.assertEqual(
            est['confidence_interval'][0], 0.739, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.833, 'Wrong CI upper value.')
