from unittest import TestCase
from irrCAC.raw import CAC
from irrCAC.datasets import raw_4raters, raw_5observers, raw_g1g2, raw_gender, \
    raw_ben_gerry


class TestFleiss(TestCase):
    def test_fleiss_kappa_raw4raters(self):
        data = raw_4raters()
        cac = CAC(data, digits=3)
        est = cac.fleiss()['est']
        self.assertEqual(est['pa'], 0.818, 'Wrong pa value.')
        self.assertEqual(est['pe'], 0.239, 'Wrong pe value.')
        self.assertEqual(
            est['coefficient_value'], 0.761, 'Wrong '
            'coefficient value.')
        self.assertEqual(
            est['confidence_interval'][0], 0.424,
            'Wrong lower value for confidence interval.')
        self.assertEqual(
            est['confidence_interval'][1], 1,
            'Wrong upper value for confidence interval.')
        self.assertEqual(round(est['p_value'], 5), 0.00042, 'Wrong p value.')

    def test_fleiss_kappa_raw4raters_bipolar_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights='bipolar', digits=3)
        est = cac.fleiss()['est']
        self.assertEqual(est['pa'], 0.968, 'Wrong pa value.')
        self.assertEqual(est['pe'], 0.785, 'Wrong pe value.')
        self.assertEqual(est['coefficient_value'], 0.853)
        self.assertEqual(est['se'], 0.145, 'Wrong standard error.')
        self.assertEqual(
            est['confidence_interval'][0], 0.535, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 1, 'Wrong CI upper value.')
        self.assertEqual(round(est['p_value'], 5), 1e-04, 'Wrong p-value.')

    def test_fleiss_kappa_raw4raters_circular_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights='circular', digits=3)
        est = cac.fleiss()['est']
        self.assertEqual(est['pa'], 0.902, 'Wrong pa value.')
        self.assertEqual(est['pe'], 0.494, 'Wrong pe value.')
        self.assertEqual(est['coefficient_value'], 0.807)
        self.assertEqual(est['se'], 0.149, 'Wrong standard error.')
        self.assertEqual(
            est['confidence_interval'][0], 0.479, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 1, 'Wrong CI upper value.')
        self.assertEqual(round(est['p_value'], 5), 0.00021, 'Wrong p-value.')

    def test_fleiss_kappa_raw4raters_linear_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights='linear', digits=3)
        est = cac.fleiss()['est']
        self.assertEqual(est['pa'], 0.939, 'Wrong pa value.')
        self.assertEqual(est['pe'], 0.667, 'Wrong pe value.')
        self.assertEqual(est['coefficient_value'], 0.818)
        self.assertEqual(est['se'], 0.149, 'Wrong standard error.')
        self.assertEqual(
            est['confidence_interval'][0], 0.491, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 1, 'Wrong CI upper value.')
        self.assertEqual(round(est['p_value'], 5), 0.00018, 'Wrong p-value.')

    def test_fleiss_kappa_raw4raters_ordinal_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights='ordinal', digits=3)
        est = cac.fleiss()['est']
        self.assertEqual(est['pa'], 0.968, 'Wrong pa value.')
        self.assertEqual(est['pe'], 0.788, 'Wrong pe value.')
        self.assertEqual(est['coefficient_value'], 0.850)
        self.assertEqual(est['se'], 0.147, 'Wrong standard error.')
        self.assertEqual(
            est['confidence_interval'][0], 0.527, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 1, 'Wrong CI upper value.')
        self.assertEqual(round(est['p_value'], 5), 0.00012, 'Wrong p-value.')

    def test_fleiss_kappa_raw4raters_quadratic_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights='quadratic', digits=3)
        est = cac.fleiss()['est']
        self.assertEqual(est['pa'], 0.975, 'Wrong pa value.')
        self.assertEqual(est['pe'], 0.818, 'Wrong pe value.')
        self.assertEqual(est['coefficient_value'], 0.865)
        self.assertEqual(est['se'], 0.146, 'Wrong standard error.')
        self.assertEqual(
            est['confidence_interval'][0], 0.544, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 1, 'Wrong CI upper value.')
        self.assertEqual(round(est['p_value'], 5), 1e-04, 'Wrong p-value.')

    def test_fleiss_kappa_raw4raters_radical_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights='radical', digits=3)
        est = cac.fleiss()['est']
        self.assertEqual(est['pa'], 0.897, 'Wrong pa value.')
        self.assertEqual(est['pe'], 0.511, 'Wrong pe value.')
        self.assertEqual(est['coefficient_value'], 0.790)
        self.assertEqual(est['se'], 0.150, 'Wrong standard error.')
        self.assertEqual(
            est['confidence_interval'][0], 0.460, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 1, 'Wrong CI upper value.')
        self.assertEqual(round(est['p_value'], 5), 2.7e-04, 'Wrong p-value.')

    def test_fleiss_kappa_raw4raters_ratio_weights(self):
        data = raw_4raters()
        cac = CAC(data, weights='ratio', digits=3)
        est = cac.fleiss()['est']
        self.assertEqual(est['pa'], 0.954, 'Wrong pa value.')
        self.assertEqual(est['pe'], 0.743, 'Wrong pe value.')
        self.assertEqual(est['coefficient_value'], 0.821)
        self.assertEqual(est['se'], 0.152, 'Wrong standard error.')
        self.assertEqual(
            est['confidence_interval'][0], 0.486, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 1, 'Wrong CI upper value.')
        self.assertEqual(round(est['p_value'], 5), 2.2e-04, 'Wrong p-value.')

    def test_fleiss_kappa_raw5observers(self):
        data = raw_5observers()
        cac = CAC(data, digits=3)
        est = cac.fleiss()['est']
        self.assertEqual(est['pa'], 0.616, 'Wrong pa value.')
        self.assertEqual(est['pe'], 0.291, 'Wrong pe value.')
        self.assertEqual(
            est['coefficient_value'], 0.458, 'Wrong coefficient value.')
        self.assertEqual(est['se'], 0.120, 'Wrong standard error.')
        self.assertEqual(
            est['confidence_interval'][0], 0.199, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.716, 'Wrong CI upper value.')
        self.assertEqual(round(est['p_value'], 5), 0.00195, 'Wrong p value.')

    def test_fleiss_kappa_raw_g1g2(self):
        data = raw_g1g2()
        cac = CAC(data, digits=3)
        est = cac.fleiss()['est']
        self.assertEqual(est['pa'], 0.744, 'Wrong pa value.')
        self.assertEqual(est['pe'], 0.252, 'Wrong pe value.')
        self.assertEqual(
            est['coefficient_value'], 0.657, 'Wrong coefficient value.')
        self.assertEqual(est['se'], 0.150, 'Wrong standard error.')
        self.assertEqual(
            est['confidence_interval'][0], 0.333, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.981, 'Wrong CI upper value.')
        self.assertEqual(round(est['p_value'], 5), 0.00074, 'Wrong p value.')

    def test_fleiss_kappa_raw_gender(self):
        data = raw_gender()
        cac = CAC(data, digits=3)
        est = cac.fleiss()['est']
        self.assertEqual(est['pa'], 0.456, 'Wrong pa value.')
        self.assertEqual(est['pe'], 0.375, 'Wrong pe value.')
        self.assertEqual(
            est['coefficient_value'], 0.129, 'Wrong coefficient value.')
        self.assertEqual(est['se'], 0.141, 'Wrong standard error.')
        self.assertEqual(
            est['confidence_interval'][0], -0.172, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 0.430, 'Wrong CI upper value.')
        self.assertEqual(round(est['p_value'], 5), 0.37449, 'Wrong p value.')

    def test_fleiss_kappa_raw_ben_gerry(self):
        data = raw_ben_gerry()
        cac = CAC(data, digits=3)
        est = cac.fleiss()['est']
        self.assertEqual(est['pa'], 0.727, 'Wrong pa value.')
        self.assertEqual(est['pe'], 0.222, 'Wrong pe value.')
        self.assertEqual(
            est['coefficient_value'], 0.649, 'Wrong coefficient value.')
        self.assertEqual(est['se'], 0.191, 'Wrong standard error.')
        self.assertEqual(
            est['confidence_interval'][0], 0.228, 'Wrong CI lower value.')
        self.assertEqual(
            est['confidence_interval'][1], 1, 'Wrong CI upper value.')
        self.assertEqual(round(est['p_value'], 5), 0.00599, 'Wrong p value.')
