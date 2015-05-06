"""
Created on Thu Dec 18 12:07:31 2014

@author: rkp

Classes for various types of plumes.
"""

import numpy as np
from logprob_odor import advec_diff_mean_hit_rate


class Environment3d(object):
    """3D environment object."""

    def __init__(self, xbins, ybins, zbins):
        # store bins
        self.xbins = xbins
        self.ybins = ybins
        self.zbins = zbins

        # calculate bin centers
        self.x = 0.5 * (xbins[:-1] + xbins[1:])
        self.y = 0.5 * (ybins[:-1] + ybins[1:])
        self.z = 0.5 * (zbins[:-1] + zbins[1:])

        # get ranges
        self.xr = self.x[-1] - self.x[0]
        self.yr = self.y[-1] - self.y[0]
        self.zr = self.z[-1] - self.z[0]

        # get index environment
        self.xidx = np.arange(len(self.x), dtype=int)
        self.yidx = np.arange(len(self.y), dtype=int)
        self.zidx = np.arange(len(self.z), dtype=int)

        # store other useful information
        self.dx = xbins[1] - xbins[0]
        self.dy = ybins[1] - ybins[0]
        self.dz = zbins[1] - zbins[0]

        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nz = len(self.z)
        self.shape = (self.nx, self.ny, self.nz)

        # for quickly calculating idxs from positions
        self.xslope = self.xidx[-1]/self.xr
        self.yslope = self.yidx[-1]/self.yr
        if self.nz > 1:
            self.zslope = self.zidx[-1]/self.zr
        else:
            self.zslope = 0

        self.xint = self.x[0]
        self.yint = self.y[0]
        self.zint = self.z[0]

        # get center idxs
        self.center_xidx = int(np.floor(self.nx/2))
        self.center_yidx = int(np.floor(self.ny/2))
        self.center_zidx = int(np.floor(self.nz/2))

    def pos_from_idx(self, idx):
        """Get floating point position from index."""

        return self.x[idx[0]], self.y[idx[1]], self.z[idx[2]]

    def idx_from_pos(self, pos):
        """Return the index corresponding to the specified postion."""

        xidx = np.round((pos[0] - self.xint) * self.xslope)
        yidx = np.round((pos[1] - self.yint) * self.yslope)
        zidx = np.round((pos[2] - self.zint) * self.zslope)

        xidx = int(xidx)
        yidx = int(yidx)
        zidx = int(zidx)

        if xidx < 0:
            xidx = 0
        elif xidx >= self.nx:
            xidx = self.nx - 1

        if yidx < 0:
            yidx = 0
        elif yidx >= self.ny:
            yidx = self.ny - 1

        if zidx < 0:
            zidx = 0
        elif zidx >= self.nz:
            zidx = self.nz - 1

        return xidx, yidx, zidx

    def idx_out_of_bounds(self, pos_idx):
        """Check whether position idx is out of bounds."""
        if np.any(np.less(pos_idx, 0)):
            return True
        elif np.any(np.greater_equal(pos_idx, (self.nx, self.ny, self.nz))):
            return True

        return False


class Plume(object):
    
    def __init__(self, env, dt=.01):
        
        # set bins and timestep
        self.env = env
        self.dt = dt

        self.ts = 0
        self.t = 0.

        # set all variables to none
        self.src_pos = None
        self.src_pos_idx = None
        self.srcx = None
        self.srcy = None
        self.srcz = None

        self.srcxidx = None
        self.srcyidx = None
        self.srczidx = None

        self.conc = None
        self.concxy = None
        self.concxz = None

    def reset(self):
        """Reset plume params."""
        # reset time and timestep
        self.ts = 0
        self.t = 0.
    
    def set_src_pos(self, pos, is_idx=False):
        """Set source position."""

        if is_idx:
            xidx, yidx, zidx = pos
        else:
            xidx, yidx, zidx = self.env.idx_from_pos(pos)
            
        # store src position
        self.src_pos_idx = (xidx, yidx, zidx)
        # convert idxs to positions
        self.src_pos = self.env.pos_from_idx(self.src_pos_idx)

        # create some other useful variables
        self.srcxidx, self.srcyidx, self.srczidx = self.src_pos_idx
        self.srcx, self.srcy, self.srcz = self.src_pos
                
    def update_time(self):
        """Update time."""
        self.ts += 1
        self.t += self.dt
        
    def update(self):
        """Update everything."""
        self.update_time()


class EmptyPlume(Plume):
    
    name = 'empty'
    
    def set_aux_params(self):
        pass
    
    def initialize(self):
        
        # create meshgrid arrays for setting conc
        x, y, z = np.meshgrid(self.xr, self.yr, self.zr, indexing='ij')
        
        # create empty conc plume
        self.conc = np.zeros(x.shape, dtype=float)
        
        # store odor domain
        self.odor_domain = [0, 1]
    
    def sample(self, pos_idx):
        return 0


class PoissonPlume(Plume):
    """In Poisson Plumes, odor samples are given by draws from a Poisson 
    distribution with mean equal to concentration times timestep (dt)"""

    max_hit_number = 1

    def sample(self, pos_idx):
        # get concentration
        conc = self.conc[tuple(pos_idx)]
        
        # sample odor from concentration, capping it at max hit number
        odor = min(np.random.poisson(lam=conc*self.dt), self.max_hit_number)
        
        return odor
        

class BasicPlume(PoissonPlume):
    """Stationary advection-diffusion based plume.
    Specified by source rate, diffusivity, particle size, and decay time, wind,
    and integration time.

    Args:
        w: wind speed (wind blows from negative to positive x-direction) (m/s)
        R: source emission rate
        D: diffusivity (m^2/s)
        a: searcher size (m)
        tau: particle lifetime (s)
        """

    name = 'basic'

    def set_aux_params(self, w=0.4, r=10, d=0.1, a=.002, tau=1000):
        # store auxiliary parameters
        self.w = w
        self.r = r
        self.d = d
        self.a = a
        self.tau = tau
        if self.env.nz == 1:
            self.dim = 2
        else:
            self.dim = 3

    def initialize(self):
        # create meshgrid of all locations
        x, y, z = np.meshgrid(self.env.x, self.env.y, self.env.z, indexing='ij')
        # calculate displacement from source
        dx = x - self.src_pos[0]
        dy = y - self.src_pos[1]
        dz = z - self.src_pos[2]

        # calculate mean hit number at all locations
        self.mean_hit_num = self.dt * advec_diff_mean_hit_rate(dx, dy, dz,
                                                               self.w, self.r, self.d,
                                                               self.a, self.tau, self.dim)
        self.conc = self.mean_hit_num

        # get xy and xz cross slices of plume
        self.concxy = self.conc[:, :, self.env.center_zidx]
        self.concxz = self.conc[:, self.env.center_yidx, :]

        # store odor domain
        self.odor_domain = range(self.max_hit_number+1)

    def sample(self, pos_idx):
        # randomly sample from plume
        mean_hit_num = self.mean_hit_num[tuple(pos_idx)]
        if not np.isinf(mean_hit_num):
            hit_num = np.random.poisson(lam=mean_hit_num)
        else:
            hit_num = np.inf

        return min(hit_num, self.max_hit_number)


class CollimatedPlume(PoissonPlume):
    """Stationary collimated plume. Specified by width (meters) and
    peak concentration."""
    
    name = 'collimated'
    
    def set_aux_params(self, width, peak, max_hit_number):
        # store auxiliary parameters
        self.width = width
        self.peak = peak
        self.max_hit_number = int(max_hit_number)
    
    def initialize(self):

        # create meshgrid arrays for setting conc
        x, y, z = np.meshgrid(self.env.x, self.env.y, self.env.z, indexing='ij')
        
        # calculate conc concentration
        dr2 = (y - self.src_pos[1])**2 + (z - self.src_pos[2])**2
        
        self.conc = self.peak * np.exp(-dr2 / (2*self.width))
        
        # put mask over space upwind of src
        mask = (x < self.src_pos[0])
        self.conc[mask] = 0.
        
        # store odor domain
        self.odor_domain = range(self.max_hit_number+1)