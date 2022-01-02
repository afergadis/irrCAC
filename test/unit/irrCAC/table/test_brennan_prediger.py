from pathlib import Path
from unittest import TestCase
from irrCAC.datasets import table_cont3x3abstractors
from irrCAC.table import CAC

dataset_dir = Path(__file__).parent / 'datasets'


class TestBP(TestCase):
    def setUp(self) -> None:
        self.cont3x3abstractors = table_cont3x3abstractors()

    def test_bp_cont3x3abstractors(self):
        cac = CAC(self.cont3x3abstractors)
        results = cac.bp()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.835, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.742,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.928,
            'Wrong CI upper value.')

    def test_bp_cont3x3abstractors_bipolar_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='bipolar')
        results = cac.bp()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.901, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.845,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.957,
            'Wrong CI upper value.')

    def test_bp_cont3x3abstractors_circular_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='circular')
        results = cac.bp()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.835, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.742,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.928,
            'Wrong CI upper value.')

    def test_bp_cont3x3abstractors_linear_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='linear')
        results = cac.bp()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.876, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.806,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.946,
            'Wrong CI upper value.')

    def test_bp_cont3x3abstractors_ordinal_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ordinal')
        results = cac.bp()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.901, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.845,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.957,
            'Wrong CI upper value.')

    def test_bp_cont3x3abstractors_quadratic_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='quadratic')
        results = cac.bp()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.918, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.871,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.964,
            'Wrong CI upper value.')

    def test_bp_cont3x3abstractors_radical_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='radical')
        results = cac.bp()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.855, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.773,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.937,
            'Wrong CI upper value.')

    def test_bp_cont3x3abstractors_ratio_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ratio')
        results = cac.bp()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.951, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.923,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.978,
            'Wrong CI upper value.')
