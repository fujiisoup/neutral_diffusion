import numpy as np
import scipy
import scipy.optimize
import sparse

from . import basis1d
from .utils import vec2coo
from .units import EV


class Model(object):
    """ A base class for a 1-dimensional neutral diffusion model """
    def __init__(self, r, m):
        """
        Parameters
        ----------
        r: 1d array-like
            Radial coordinate of the system. This should be monotonically
            increasing [m].
        m: float
            Mass of atoms [kg].
        """
        if not (np.diff(r) > 0).all():
            raise ValueError('Coordinate should be monotonically increasing.')
        self.r = r.astype(float)
        self.size = len(self.r)
        n = self.size
        self.m = m

        # basises
        self.phi_ijk = basis1d.phi_ijk(r)
        self.phi_ijkl = basis1d.phi_ijkl(r)
        self.phi_i_dj_dk_l = basis1d.phi_i_dj_dk_l(r)
        self.phi_ijk_dl_m = basis1d.phi_ijk_dl_m(r)
        self.phi_ij_dk_dl_m = basis1d.phi_ij_dk_dl_m(r)
        self.slice_l = sparse.COO([np.arange(n-1), np.arange(n-1)],
                                  np.ones(n-1), shape=(n, n-1))


class Cylindrical(Model):
    """
    A class to solve a neutral diffusion model in one-dimensional cylindrical
    plasmas.
    """
    def solve(self, rate_ion, rate_cx, t_ion, t_edge,
              n_init=None, t_init=None):
        """
        Solve a diffusion equation with particular paramters.

        rate_ion: 1d array-like
            Ionization rate [/s].
        rate_cx: 1d array-like
            Charge exchange rate [/s].
        t_ion: 1d array-like
            Ion temperature in plasmas [eV]
        t_edge: float
            Edge temperature of atoms [eV]
        n_init: 1d array-like
            Initial guess of atom density [m^-3]
        t_init: 1d array-like
            Initial guess of atom temperature [eV]

        Returns
        -------
        n: 1d array-like
            Neutral atom density. This is normalized so that the edge density
            is 1.
        t_atom: 1d array-like
            Neutral atom temperature.
        """
        for v in [rate_ion, rate_cx, t_ion]:
            if v.shape != self.r.shape:
                raise ValueError('Shape mismatch, {} and {}'.format(
                                                v.shape, t_ion.shape))
        if n_init is None:
            n_init = 1.0 / t_ion
        n_init = n_init * t_ion

        if t_init is None:
            t_init = np.ones_like(self.r) * t_edge
        t_init = t_init / t_ion
        t_edge = t_edge / t_ion[-1]

        t_ion = EV * t_ion
        t_ion_grad = np.gradient(t_ion, self.r)
        r_ion_cx = rate_ion + rate_cx
        rmu = self.r / (self.m * r_ion_cx)

        r_rion_tion = self.r * rate_ion / t_ion
        rmu_t_ion = vec2coo(rmu * t_ion)
        rmu_t_ion_grad = vec2coo(rmu * t_ion_grad)
        rmu = vec2coo(rmu)

        Dijl_tmp = sparse.tensordot(rmu, self.phi_i_dj_dk_l, axes=(0, -1))
        Dijl = Dijl_tmp + np.moveaxis(Dijl_tmp, 0, 1)

        Ril = -sparse.tensordot(vec2coo(r_rion_tion), self.phi_ijk,
                                axes=(0, -1))
        Dijl *= EV  # This normalizes the order or these tensors
        Ril *= EV

        Fijkl_tmp1 = sparse.tensordot(rmu_t_ion_grad, self.phi_ijk_dl_m,
                                      axes=(0, -1))
        Fijkl_tmp2 = sparse.tensordot(rmu_t_ion, self.phi_ij_dk_dl_m,
                                      axes=(0, -1))
        Fijkl = (Fijkl_tmp1 + Fijkl_tmp2 + np.moveaxis(Fijkl_tmp2, 0, 1)
                 + np.moveaxis(Fijkl_tmp2, 0, 2)) * 2.5

        Gijl = -1.5 * sparse.tensordot(vec2coo(self.r * r_ion_cx),
                                       self.phi_ijkl, axes=(0, -1))
        Hil = 1.5 * sparse.tensordot(vec2coo(self.r * rate_cx),
                                     self.phi_ijk, axes=(0, -1))

        # Remove last item of last dimension
        Dijl = sparse.tensordot(Dijl, self.slice_l, axes=(-1, 0))
        Ril = sparse.tensordot(Ril, self.slice_l, axes=(-1, 0))
        Fijkl = sparse.tensordot(Fijkl, self.slice_l, axes=(-1, 0))
        Gijl = sparse.tensordot(Gijl, self.slice_l, axes=(-1, 0))
        Hil = sparse.tensordot(Hil, self.slice_l, axes=(-1, 0))

        # Callable to return a residual.
        def fun(x):
            x = np.exp(x)
            # x: size (2*(size-1), 1).
            n = vec2coo(np.concatenate([x[:self.size-1], [1.0]],
                                       axis=0))
            t = vec2coo(np.concatenate([x[self.size-1:], [t_edge]],
                                       axis=0))

            # particle balance
            lhs = sparse.tensordot(n, sparse.tensordot(t, Dijl, axes=(0, 1)),
                                   axes=(0, 0))
            rhs = sparse.tensordot(n, Ril, axes=(0, 0))
            res_particle = lhs - rhs

            # energy balance
            lhs = sparse.tensordot(n, sparse.tensordot(t, sparse.tensordot(
                        t, Fijkl, axes=(0, 2)), axes=(0, 1)), axes=(0, 0))
            rhs = sparse.tensordot(n, sparse.tensordot(t, Gijl, axes=(0, 1)),
                                   axes=(0, 0)) + sparse.tensordot(n, Hil,
                                                                   axes=(0, 0))
            res_energy = lhs - rhs
            return np.concatenate([res_particle, res_energy])

        x_init = np.log(np.concatenate([n_init[:-1], t_init[:-1]]))
        #x_init = np.concatenate([n_init[:-1], t_init[:-1]])
        print(scipy.optimize.least_squares(fun, x_init, max_nfev=100, method='lm'))
        raise ValueError
