from unittest import TestCase
from irrCAC.benchmark import Benchmark


class TestBenchmark(TestCase):
    def setUp(self) -> None:
        self.benchmark = Benchmark(coeff=0.67, se=0.15)

    def test_altman(self):
        expected_cumprob = [0.18168, 0.67511, 0.96356, 0.99912, 1.0]
        result = self.benchmark.altman()
        cumprob = result['CumProb']
        self.assertListEqual(expected_cumprob, cumprob)

    def test_cicchetti_sparrow(self):
        expected_cumprob = [0.28699, 0.67511, 0.96356, 1.0]
        result = self.benchmark.cicchetti_sparrow()
        cumprob = result['CumProb']
        self.assertListEqual(expected_cumprob, cumprob)

    def test_fleiss(self):
        expected_cumprob = [0.28699, 0.96356, 1.0]
        result = self.benchmark.fleiss()
        cumprob = result['CumProb']
        self.assertListEqual(expected_cumprob, cumprob)

    def test_landis_koch(self):
        expected_cumprob = [0.18168, 0.67511, 0.96356, 0.99912, 1.0, 1.0]
        result = self.benchmark.landis_koch()
        cumprob = result['CumProb']
        self.assertListEqual(expected_cumprob, cumprob)

    def test_regier(self):
        expected_cumprob = [0.18168, 0.67511, 0.96356, 0.99912, 1.0]
        result = self.benchmark.regier()
        cumprob = result['CumProb']
        self.assertListEqual(expected_cumprob, cumprob)

    def test_custom_scale(self):
        my_scale = dict(
            lb=[0.0, 0.3, 0.6],
            ub=[0.3, 0.6, 1.0],
            interp=['Poor', 'Acceptable', 'Excellent'],
            scale_name='My Scale')
        expected_cumprob = [0.00691, 0.32488, 1.0]
        result = self.benchmark.interpret(my_scale)
        cumprob = result['CumProb']
        self.assertListEqual(expected_cumprob, cumprob)

    def test_interpretation_raises_valueerror(self):
        error_scale = dict()
        with self.assertRaises(ValueError):
            _ = self.benchmark.interpret(error_scale)