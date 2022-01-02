import pandas as pd
from pathlib import Path
from unittest import TestCase
from irrCAC.datasets import table_cont3x3abstractors
from irrCAC.table import CAC

dataset_dir = Path(__file__).parent / 'datasets'


class Test(TestCase):
    def setUp(self) -> None:
        self.cont3x3abstractors = table_cont3x3abstractors()

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
