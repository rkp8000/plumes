import numpy as np
from plume import Environment3d

XRBINS = np.linspace(-0.3, 1.0, 66)
YRBINS = np.linspace(-0.15, 0.15, 16)
ZRBINS = np.linspace(-0.15, 0.15, 16)
ENV = Environment3d(XRBINS, YRBINS, ZRBINS)

PARAMS = {'Q': -0.26618286981003886,
          'u': 0.4,
          'u_star': 0.06745668765535813,
          'alpha_y': -0.066842568000323691,
          'alpha_z': 0.14538827993452938,
          'x_source': -0.64790143304753445,
          'y_source': .003,
          'z_source': .011,
          'bkgd': 400,
          'threshold': 450}