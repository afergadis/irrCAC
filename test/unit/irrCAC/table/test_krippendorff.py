from unittest import TestCase
from irrCAC.datasets import table_cont3x3abstractors
from irrCAC.table import CAC


class TestKrippendorff(TestCase):
    def setUp(self) -> None:
        self.cont3x3abstractors = table_cont3x3abstractors()

    def test_krippendorff_cont3x3abstractors(self):
        cac = CAC(self.cont3x3abstractors)
        results = cac.krippendorff()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.797, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.680,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.914,
            'Wrong CI upper value.')

    def test_krippendorff_cont3x3abstractors_bipolar_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='bipolar')
        results = cac.krippendorff()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.873, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.792,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.953,
            'Wrong CI upper value.')

    def test_krippendorff_cont3x3abstractors_circular_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='circular')
        results = cac.krippendorff()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.797, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.680,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.914,
            'Wrong CI upper value.')

    def test_krippendorff_cont3x3abstractors_linear_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='linear')
        results = cac.krippendorff()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.844, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.748,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.939,
            'Wrong CI upper value.')

    def test_krippendorff_cont3x3abstractors_ordinal_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ordinal')
        results = cac.krippendorff()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.873, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.792,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.953,
            'Wrong CI upper value.')

    def test_krippendorff_cont3x3abstractors_quadratic_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='quadratic')
        results = cac.krippendorff()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.893, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.822,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.963,
            'Wrong CI upper value.')

    def test_krippendorff_cont3x3abstractors_radical_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='radical')
        results = cac.krippendorff()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.819, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.713,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.926,
            'Wrong CI upper value.')

    def test_krippendorff_cont3x3abstractors_ratio_weights(self):
        cac = CAC(self.cont3x3abstractors, weights='ratio')
        results = cac.krippendorff()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 3), 0.927, 'Wrong coeff value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 3), 0.877,
            'Wrong CI lower value.')
        self.assertEqual(
            round(est['confidence_interval'][1], 3), 0.977,
            'Wrong CI upper value.')
