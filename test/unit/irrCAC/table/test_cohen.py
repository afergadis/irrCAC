from pathlib import Path
from unittest import TestCase
from irrCAC.datasets import table_cont3x3abstractors
from irrCAC.table import CAC

dataset_dir = Path(__file__).parent / 'datasets'


class Test(TestCase):
    def setUp(self) -> None:
        self.cont3x3abstractors = table_cont3x3abstractors()

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
