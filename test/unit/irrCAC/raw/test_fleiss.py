from unittest import TestCase
from irrCAC.raw import CAC
from irrCAC.datasets import raw_4raters, raw_5observers, raw_g1g2, raw_gender, \
    raw_ben_gerry


class TestFleiss(TestCase):

    def test_fleiss_kappa_raw4raters(self):
        data = raw_4raters()
        cac = CAC(data)
        results = cac.fleiss()['est']
        self.assertEqual(
            round(results['coefficient_value'], 5), 0.76117, 'Wrong '
            'coefficient value.')
        self.assertEqual(
            round(results['confidence_interval'][0], 5), 0.42438,
            'Wrong lower value for confidence interval.')
        self.assertEqual(
            round(results['confidence_interval'][1], 5), 1,
            'Wrong upper value for confidence interval.')
        self.assertEqual(
            round(results['p_value'], 5), 0.00042, 'Wrong p value.')

    def test_fleiss_kappa_raw5observers(self):
        data = raw_5observers()
        cac = CAC(data)
        results = cac.fleiss()['est']
        self.assertEqual(
            round(results['coefficient_value'], 5), 0.45762,
            'Wrong coefficient value.')
        self.assertEqual(
            round(results['confidence_interval'][0], 5), 0.19928,
            'Wrong lower value for confidence interval.')
        self.assertEqual(
            round(results['confidence_interval'][1], 5), 0.71596,
            'Wrong upper value for confidence interval.')
        self.assertEqual(
            round(results['p_value'], 5), 0.00195, 'Wrong p value.')

    def test_fleiss_kappa_raw_g1g2(self):
        data = raw_g1g2()
        cac = CAC(data)
        results = cac.fleiss()['est']
        self.assertEqual(
            round(results['coefficient_value'], 5), 0.65725, 'Wrong '
            'coefficient value.')
        self.assertEqual(
            round(results['confidence_interval'][0], 5), 0.33337,
            'Wrong lower value for confidence interval.')
        self.assertEqual(
            round(results['confidence_interval'][1], 5), 0.98112,
            'Wrong upper value for confidence interval.')
        self.assertEqual(
            round(results['p_value'], 5), 0.00074, 'Wrong p value.')

    def test_fleiss_kappa_raw_gender(self):
        data = raw_gender()
        cac = CAC(data)
        results = cac.fleiss()['est']
        self.assertEqual(
            round(results['coefficient_value'], 5), 0.12889, 'Wrong '
            'coefficient value.')
        self.assertEqual(
            round(results['confidence_interval'][0], 5), -0.17246,
            'Wrong lower value for confidence interval.')
        self.assertEqual(
            round(results['confidence_interval'][1], 5), 0.43023,
            'Wrong upper value for confidence interval.')
        self.assertEqual(
            round(results['p_value'], 5), 0.37449, 'Wrong p value.')

    def test_fleiss_kappa_raw_ben_gerry(self):
        data = raw_ben_gerry()
        cac = CAC(data)
        results = cac.fleiss()['est']
        self.assertEqual(
            round(results['coefficient_value'], 5), 0.64935, 'Wrong coefficient value.')
        self.assertEqual(
            round(results['confidence_interval'][0], 5), 0.22831,
            'Wrong lower value for confidence interval.')
        self.assertEqual(
            round(results['confidence_interval'][1], 5), 1,
            'Wrong upper value for confidence interval.')
        self.assertEqual(
            round(results['p_value'], 5), 0.00599, 'Wrong p value.')
