from __future__ import print_function, division

import unittest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import plume

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


if __name__ == '__main__':
    unittest.main()