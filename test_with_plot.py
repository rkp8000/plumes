from __future__ import print_function, division

import unittest
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.cm as cm

import plume
import logprob_odor

from config_test_plot import *


class SpreadingGaussianTestCase(unittest.TestCase):

    def setUp(self):
        self.env = ENV
        self.params = PARAMS

    def test_show_concentration_profile(self):

        pl = plume.SpreadingGaussianPlume(self.env)
        pl.set_params(**self.params)
        pl.initialize()

        _, ax = plt.subplots(1, 1, tight_layout=True)

        ax.matshow(pl.concxy.T, origin='lower', cmap=cm.hot)

        plt.show(block=True)

        self.assertTrue(True)


class AdvectionDiffusionBinaryTestCase(unittest.TestCase):

    def test_plume_profile(self):

        dx = np.linspace(-10.5, 119.5, 500)
        dy = np.linspace(-.5, .5, 100)
        dz = np.array([0])

        DX, DY, DZ = np.meshgrid(dx, dy, dz)

        _, axs = plt.subplots(2, 2, tight_layout=True)

        hit_rate = logprob_odor.advec_diff_mean_hit_rate(DX, DY, DZ, w=0.0, r=10, d=0.1, a=.002, tau=100, dim=2)
        axs[0, 0].matshow(hit_rate[:, :, 0])
        axs[0, 1].plot(dx, hit_rate[:, :, 0].T)

        hit_rate = logprob_odor.advec_diff_mean_hit_rate(DX, DY, DZ, w=0.0, r=10, d=0.1, a=.002, tau=10, dim=2)
        axs[1, 0].matshow(hit_rate[:, :, 0])
        axs[1, 1].plot(dx, hit_rate[:, :, 0].T)

        plt.show(block=True)


class NumericalErrorsTestCase(unittest.TestCase):

    def test_x_coordinate_too_big(self):
        dxs = np.linspace(36.5, 37.5, 100)
        dy = 0
        dz = 0

        hit_rates = []
        for dx in dxs:
            hit_rate = logprob_odor.advec_diff_mean_hit_rate(dx, dy, dz, w=0.4, r=10, d=0.01, a=.002, tau=100, dim=2)
            hit_rates += [hit_rate]

        _, ax = plt.subplots(1, 1)
        ax.plot(dxs, hit_rates)
        plt.show(block=True)


if __name__ == '__main__':
    unittest.main()