"""Unit tests for plumes."""
from __future__ import division, print_function

import unittest
import numpy as np
from scipy.stats import multivariate_normal as mvn
import plume


class TruismsTestCase(unittest.TestCase):

    def test_truisms(self):
        self.assertTrue(True)

    def test_falsisms(self):
        self.assertFalse(True)


class Environment3dTestCase(unittest.TestCase):

    def test_pos_to_idx_conversion(self):

        xrbins = np.linspace(0, 10., 11)
        yrbins = np.linspace(0, 5., 6)
        zrbins = np.linspace(0, 5., 6)

        env = plume.Environment3d(xrbins, yrbins, zrbins)

        idx = (3, 2, 4)
        pos = (3.5, 2.5, 4.5)

        np.testing.assert_array_almost_equal(np.array(idx),
                                             env.idx_from_pos(pos),
                                             decimal=5)
        np.testing.assert_array_almost_equal(np.array(pos),
                                             env.pos_from_idx(idx),
                                             decimal=5)

        pos = (3.6, 2.6, 4.6)

        np.testing.assert_array_almost_equal(np.array(idx),
                                             env.idx_from_pos(pos),
                                             decimal=5)
        pos = (3.4, 2.4, 4.4)

        np.testing.assert_array_almost_equal(np.array(idx),
                                             env.idx_from_pos(pos),
                                             decimal=5)


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


class DiscretizationTestCase(unittest.TestCase):

    def setUp(self):
        xrbins = np.linspace(0, 1., 11)
        yrbins = np.linspace(0, 1., 11)
        zrbins = np.linspace(0, 1., 11)
        self.env = plume.Environment3d(xrbins, yrbins, zrbins)

    def test_diagonalest_lattice_path(self):
        starts_and_ends = (((0, 0, 0), (4, 0, 0)),
                           ((0, 0, 0), (1, 1, 1)),
                           ((0, 0, 0), (2, 2, 2)),
                           ((1, 2, 3), (0, 0, 1)))

        true_path_lengths = (4, 3, 6, 5)

        # loop through every start & end pair and check to make sure the generated path is correct
        for start_and_end, true_path_length in zip(starts_and_ends, true_path_lengths):
            start, end = start_and_end
            pos_idxs = self.env.diagonalest_lattice_path(start, end)

            # check path length
            self.assertEqual(len(pos_idxs), true_path_length)
            # check that first element is one different from start
            self.assertEqual(np.abs(np.array(start) - pos_idxs[0]).sum(), 1)
            # check that last element is end
            self.assertTrue(np.all(np.array(end) == pos_idxs[-1]))

            # check that each pos idx is one step away from the previous pos idx
            for pi_ctr, pos_idx in enumerate(pos_idxs[:-1]):
                next_pos_idx = np.array(pos_idxs[pi_ctr + 1])
                self.assertEqual(np.abs(pos_idx - next_pos_idx).sum(), 1)

    def test_straight_line_trajectory_discretization_by_environment3d(self):

        # this trajectory should have 11 timesteps when mapped onto the grid in env
        x = 0.55 * np.ones((30,))
        y = np.linspace(.15, .75, 30)
        z = np.linspace(.45, .05, 30)
        positions = np.array([x, y, z]).T

        pos_idxs = self.env.discretize_position_sequence(positions)

        # check to make sure duration is correct
        self.assertEqual(len(pos_idxs), 11)

    def test_perfectly_diagonal_trajectory_discretization_by_environment3d(self):
        x = np.linspace(.15, .75, 40)
        y = np.linspace(.15, .75, 40)
        z = np.linspace(.15, .75, 40)
        positions = np.array([x, y, z]).T

        pos_idxs = self.env.discretize_position_sequence(positions)

        # check to make sure duration is correct
        true_duration = 19
        self.assertEqual(len(pos_idxs), true_duration)

        # check that each pos idx is one step away from the previous pos idx
        for pi_ctr, pos_idx in enumerate(pos_idxs[:-1]):
            next_pos_idx = np.array(pos_idxs[pi_ctr + 1])
            self.assertEqual(np.abs(pos_idx - next_pos_idx).sum(), 1)

    def test_random_walk_trajectory_discretization_by_environment3d(self):

        # loop over some more random trajectories
        for _ in range(5):

            x = 0.5 + np.random.normal(0, .003, (1000,)).cumsum()
            y = 0.5 + np.random.normal(0, .003, (1000,)).cumsum()
            z = 0.5 + np.random.normal(0, .003, (1000,)).cumsum()
            positions = np.array([x, y, z]).T
            # truncate positions if any of them go beyond 1 or 0
            outside_env = [ts for ts, pos in enumerate(positions) if np.any(pos > 1) or np.any(pos < 0)]
            if outside_env:
                positions = positions[:outside_env[0]]

            pos_idxs = self.env.discretize_position_sequence(positions)

            # check that each pos idx is one step away from the previous pos idx
            for pi_ctr, pos_idx in enumerate(pos_idxs[:-1]):
                next_pos_idx = np.array(pos_idxs[pi_ctr + 1])
                self.assertEqual(np.abs(pos_idx - next_pos_idx).sum(), 1)

            # check that first and last pos idx are what env would give them
            first_pos_idx_env = np.array(self.env.idx_from_pos(positions[0]))
            last_pos_idx_env = np.array(self.env.idx_from_pos(positions[-1]))
            np.testing.assert_array_equal(first_pos_idx_env, np.array(pos_idxs[0]))
            np.testing.assert_array_equal(last_pos_idx_env, np.array(pos_idxs[-1]))


if __name__ == '__main__':
    unittest.main()