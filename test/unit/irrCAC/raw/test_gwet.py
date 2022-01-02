from unittest import TestCase
from irrCAC.datasets import raw_4raters, raw_5observers, raw_g1g2, raw_gender
from irrCAC.raw import CAC


class TestGwet(TestCase):

    def test_gwet_raw4raters(self):
        data = raw_4raters()
        cac = CAC(data)
        results = cac.gwet()['est']
        self.assertEqual(
            round(results['coefficient_value'], 5), 0.77544, 'Wrong '
            'coefficient value.')
        self.assertEqual(
            round(results['confidence_interval'][0], 5), 0.46081,
            'Wrong lower value for confidence interval.')
        self.assertEqual(
            round(results['confidence_interval'][1], 5), 1,
            'Wrong upper value for confidence interval.')
        self.assertEqual(
            round(results['p_value'], 5), 0.00021, 'Wrong p value.')

    def test_gwet_raw4raters_bipolar_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights='bipolar')
        results = cac.gwet()
        est = results['est']
        self.assertEqual(
            round(est['coefficient_value'], 5), 0.90037, 'Wrong coefficient '
            'value.')
        self.assertEqual(
            round(est['confidence_interval'][0], 5), 0.66747,
            'Wrong lower value for confidence interval.')
        self.assertEqual(
            round(est['confidence_interval'][1], 5), 1,
            'Wrong upper value for confidence interval.')
        self.assertEqual(round(est['p_value'], 5), 0.0, 'Wrong p value.')

    def test_gwet_raw4raters_quadratic_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights='quadratic')
        results = cac.gwet()['est']
        self.assertEqual(
            round(results['coefficient_value'], 5), 0.914, 'Wrong coefficient '
            'value.')
        self.assertEqual(
            round(results['confidence_interval'][0], 5), 0.68518,
            'Wrong lower value for confidence interval.')
        self.assertEqual(
            round(results['confidence_interval'][1], 5), 1,
            'Wrong upper value for confidence interval.')
        self.assertEqual(round(results['p_value'], 5), 0.0, 'Wrong p value.')

    def test_gwet_raw5observers(self):
        data = raw_5observers()
        cac = CAC(data)
        results = cac.gwet()['est']
        self.assertEqual(
            round(results['coefficient_value'], 5), 0.49662, 'Wrong '
            'coefficient value.')
        self.assertEqual(
            round(results['confidence_interval'][0], 5), 0.23502,
            'Wrong lower value for confidence interval.')
        self.assertEqual(
            round(results['confidence_interval'][1], 5), 0.75823,
            'Wrong upper value for confidence interval.')
        self.assertEqual(
            round(results['p_value'], 5), 0.00114, 'Wrong p value.')

    def test_gwet_raw_g1g2(self):
        data = raw_g1g2()
        cac = CAC(data)
        results = cac.gwet()['est']
        self.assertEqual(
            round(results['coefficient_value'], 5), 0.6846,
            'Wrong coefficient '
            'value.')
        self.assertEqual(
            round(results['confidence_interval'][0], 5), 0.39612,
            'Wrong lower value for confidence interval.')
        self.assertEqual(
            round(results['confidence_interval'][1], 5), 0.97308,
            'Wrong upper value for confidence interval.')
        self.assertEqual(
            round(results['p_value'], 5), 0.00019, 'Wrong p value.')

    def test_gwet_raw_gender(self):
        data = raw_gender()
        cac = CAC(data)
        results = cac.gwet()['est']
        self.assertEqual(
            round(results['coefficient_value'], 5), 0.20808, 'Wrong '
            'coefficient value.')
        self.assertEqual(
            round(results['confidence_interval'][0], 5), 0.01356,
            'Wrong lower value for confidence interval.')
        self.assertEqual(
            round(results['confidence_interval'][1], 5), 0.40261,
            'Wrong upper value for confidence interval.')
        self.assertEqual(
            round(results['p_value'], 5), 0.03776, 'Wrong p value.')
