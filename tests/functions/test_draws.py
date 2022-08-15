"""
Test the draws module

:author: Michel Bierlaire
:data: Wed Apr 29 17:56:55 2020
"""
# Bug in pylint
# pylint: disable=no-member
#
# Too constraining
# pylint: disable=invalid-name, too-many-instance-attributes
#
# Not needed in test
# pylint: disable=missing-function-docstring, missing-class-docstring

import unittest
import biogeme.draws as dr
import numpy as np


class test_draws(unittest.TestCase):
    def setUp(self):
        np.random.seed(90267)

    def test_getUniform(self):
        draws = dr.getUniform(sampleSize=5, numberOfDraws=10, symmetric=False)
        r, c = draws.shape
        self.assertEqual(r, 5)
        self.assertEqual(c, 10)
        self.assertTrue(np.min(draws) > 0)
        self.assertTrue(np.max(draws) <= 1)

    def test_getUniformSymmetric(self):
        draws = dr.getUniform(sampleSize=5, numberOfDraws=10, symmetric=True)
        r, c = draws.shape
        self.assertEqual(r, 5)
        self.assertEqual(c, 10)
        self.assertTrue(np.min(draws) > -1)
        self.assertTrue(np.max(draws) <= 1)

    def test_getLatinHypercubeDraws(self):
        draws = dr.getLatinHypercubeDraws(sampleSize=5, numberOfDraws=10)
        r, c = draws.shape
        self.assertEqual(r, 5)
        self.assertEqual(c, 10)
        self.assertTrue(np.min(draws) > 0)
        self.assertTrue(np.max(draws) <= 1)

    def test_getLatinHypercubeDrawsSymmetric(self):
        draws = dr.getLatinHypercubeDraws(
            sampleSize=5, numberOfDraws=10, symmetric=True
        )
        r, c = draws.shape
        self.assertEqual(r, 5)
        self.assertEqual(c, 10)
        self.assertTrue(np.min(draws) > -1)
        self.assertTrue(np.max(draws) <= 1)

    def test_userDefinedDraws(self):
        myUnif = np.random.uniform(size=30)
        draws = dr.getLatinHypercubeDraws(
            sampleSize=3,
            numberOfDraws=10,
            symmetric=False,
            uniformNumbers=myUnif,
        )
        r, c = draws.shape
        self.assertEqual(r, 3)
        self.assertEqual(c, 10)
        self.assertTrue(np.min(draws) > 0)
        self.assertTrue(np.max(draws) <= 1)

    def test_halton(self):
        halton = dr.getHaltonDraws(sampleSize=1, numberOfDraws=100, base=3)
        h = np.array(
            [
                0.33333333,
                0.66666667,
                0.11111111,
                0.44444444,
                0.77777778,
                0.22222222,
                0.55555556,
                0.88888889,
                0.03703704,
                0.37037037,
                0.7037037,
                0.14814815,
                0.48148148,
                0.81481481,
                0.25925926,
                0.59259259,
                0.92592593,
                0.07407407,
                0.40740741,
                0.74074074,
                0.18518519,
                0.51851852,
                0.85185185,
                0.2962963,
                0.62962963,
                0.96296296,
                0.01234568,
                0.34567901,
                0.67901235,
                0.12345679,
                0.45679012,
                0.79012346,
                0.2345679,
                0.56790123,
                0.90123457,
                0.04938272,
                0.38271605,
                0.71604938,
                0.16049383,
                0.49382716,
                0.82716049,
                0.27160494,
                0.60493827,
                0.9382716,
                0.08641975,
                0.41975309,
                0.75308642,
                0.19753086,
                0.5308642,
                0.86419753,
                0.30864198,
                0.64197531,
                0.97530864,
                0.02469136,
                0.35802469,
                0.69135802,
                0.13580247,
                0.4691358,
                0.80246914,
                0.24691358,
                0.58024691,
                0.91358025,
                0.0617284,
                0.39506173,
                0.72839506,
                0.17283951,
                0.50617284,
                0.83950617,
                0.28395062,
                0.61728395,
                0.95061728,
                0.09876543,
                0.43209877,
                0.7654321,
                0.20987654,
                0.54320988,
                0.87654321,
                0.32098765,
                0.65432099,
                0.98765432,
                0.00411523,
                0.33744856,
                0.67078189,
                0.11522634,
                0.44855967,
                0.781893,
                0.22633745,
                0.55967078,
                0.89300412,
                0.04115226,
                0.3744856,
                0.70781893,
                0.15226337,
                0.48559671,
                0.81893004,
                0.26337449,
                0.59670782,
                0.93004115,
                0.0781893,
                0.41152263,
            ]
        )
        diff = halton - h
        norm = np.linalg.norm(diff)
        self.assertAlmostEqual(norm, 0, 7)
        halton = dr.getHaltonDraws(
            sampleSize=1, numberOfDraws=10, base=3, skip=10
        )
        h = [
            0.7037037,
            0.14814815,
            0.48148148,
            0.81481481,
            0.25925926,
            0.59259259,
            0.92592593,
            0.07407407,
            0.40740741,
            0.74074074,
        ]
        diff = halton - h
        norm = np.linalg.norm(diff)
        self.assertAlmostEqual(norm, 0, 7)

    def test_antithetic(self):
        draws = dr.getAntithetic(
            dr.getHaltonDraws, sampleSize=1, numberOfDraws=10
        )
        d = [0.5, 0.25, 0.75, 0.125, 0.625, 0.5, 0.75, 0.25, 0.875, 0.375]
        self.assertListEqual(d, draws[0].tolist())

    def test_normal(self):
        draws = dr.getNormalWichuraDraws(sampleSize=3, numberOfDraws=1000000)
        mean = np.linalg.norm(np.average(draws, axis=1))
        self.assertAlmostEqual(mean, 0, 2)
        draws = dr.getNormalWichuraDraws(
            sampleSize=3, numberOfDraws=1000000, antithetic=True
        )
        mean = np.linalg.norm(np.average(draws, axis=1))
        self.assertAlmostEqual(mean, 0, 5)


if __name__ == '__main__':
    unittest.main()
