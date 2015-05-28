from __future__ import division, print_function
"""Unit tests for plumes."""

import unittest
import numpy as np
from scipy.stats import multivariate_normal as mvn
import plume


class TruismsTestCase(unittest.TestCase):

    def test_truisms(self):
        self.assertTrue(True)


class Environment3dTestCase(unittest.TestCase):
    pass


class SimplePlumeTestCase(unittest.TestCase):

    def test_collimated_plume(self):
        xbins = np.linspace(-1, 10, 50)
        ybins = np.linspace(-1, 1, 20)
        zbins = np.linspace(-1, 1, 20)
        env = plume.Environment3d(xbins, ybins, zbins)

        pl = plume.CollimatedPlume(env, dt=.01)

        # test max concentration of plume
        max_conc = 250
        threshold = 10
        ymean = 0
        zmean = 0
        ystd = 0.2
        zstd = 0.2

        pl.set_params(max_conc=max_conc, threshold=threshold, ymean=ymean, zmean=zmean, ystd=ystd, zstd=zstd)

        pl.initialize()

        test_pos = (1., 0, 0)
        test_pos_idx = env.idx_from_pos(test_pos)
        self.assertAlmostEqual(pl.conc[test_pos_idx], max_conc, delta=.00001)

        # test various other concentrations of plume with arbitrary center
        ymean = 0.11
        zmean = -.23

        pl.set_params(max_conc=max_conc, threshold=threshold, ymean=ymean, zmean=zmean, ystd=ystd, zstd=zstd)

        pl.initialize()

        test_pos_idxs = [(32, 6, 11), (9, 3, 8), (47, 2, 18)]
        for test_pos_idx in test_pos_idxs:
            testx, testy, testz = env.pos_from_idx(test_pos_idx)
            exponent = (-0.5 * ((testy - ymean)**2) / (ystd**2)) + (-0.5 * ((testz - zmean)**2) / (zstd**2))
            theoretical_conc = max_conc * np.exp(exponent)
            self.assertAlmostEqual(pl.conc[test_pos_idx], theoretical_conc, delta=.00001)

        # ensure no dependence of concentration on x
        self.assertAlmostEqual(pl.conc[(3, 6, 11)], pl.conc[(38, 6, 11)])