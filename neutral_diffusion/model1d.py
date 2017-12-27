import numpy as np
import scipy
import sparse

from . import basis1d
from . import solvers
from .utils import vec2coo


EV = 1.60217662E-19  # [J/eV]


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
        self.phi_ij_dk_dl_m = basis1d.phi_ij_dk_dl_m(r)
        self.slice_l = sparse.COO([np.arange(n-1), np.arange(n-1)],
                                  np.ones(n-1), shape=(n, n-1))
        self.slice_last = sparse.COO([[1]], [1], shape=(n, ))


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
            n_init = t_ion / t_edge
        n_init = n_init * EV * t_ion

        if t_init is None:
            t_init = t_ion
        t_init = t_init / t_ion

        t_ion = EV * t_ion
        r_ion_cx = rate_ion + rate_cx
        rmu = self.r / (self.m * r_ion_cx)
        Dijl_tmp = sparse.tensordot(vec2coo(rmu), self.phi_i_dj_dk_l,
                                    axes=(0, -1))
        Dijl = Dijl_tmp + np.moveaxis(Dijl_tmp, 0, 1)

        r_rion_tion = self.r * rate_ion / t_ion
        Ril = -sparse.tensordot(vec2coo(r_rion_tion), self.phi_ijk,
                                axes=(0, -1))

        t_ion = vec2coo(t_ion)
        Fijkl_tmp1 = sparse.tensordot(t_ion, self.phi_ij_dk_dl_m, axes=(0, -1))
        Fijkl_tmp2 = sparse.tensordot(t_ion, self.phi_ij_dk_dl_m, axes=(0, -2))
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

        # construct equations
        def get_A(x):
            # x: size (2*(size-1), 1).
            n = vec2coo(np.concatenate([x[:self.size-1, 0], [1.0]],
                                       axis=0))
            t = vec2coo(np.concatenate([x[self.size-1:, 0], [t_edge]],
                                       axis=0))
            # matrix for n
            An1 = sparse.tensordot(t, Dijl, axes=(0, 1)) + Ril
            An2 = sparse.tensordot(t, sparse.tensordot(t, Fijkl, axes=(0, 1)),
                                   axes=(0, 1)) + Hil
            An = scipy.sparse.hstack([An1.to_scipy_sparse().T,
                                      An2.to_scipy_sparse().T])
            # matrix for t
            At1 = sparse.tensordot(n, Dijl, axes=(0, 0))
            At2 = sparse.tensordot(t, sparse.tensordot(n, Fijkl, axes=(0, 0)),
                                   axes=(0, 1)) * 2.0
            At = scipy.sparse.hstack([At1.to_scipy_sparse().T,
                                      At2.to_scipy_sparse().T])
            return scipy.sparse.vstack([An, At])

        return get_A(np.concatenate([n_init[:-1, np.newaxis],
                                     t_init[:-1, np.newaxis]]))
