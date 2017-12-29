import numpy as np
import scipy
import scipy.optimize
import sparse

from . import basis1d
from .utils import vec2coo
from .constants import EV


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
    def solve(self, rate_ion, rate_cx, t_ion, t_edge, n_init=None, t_init=None,
              always_positive=False, **kwargs):
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
        self.initialize(rate_ion, rate_cx, t_ion, t_edge)

        if n_init is None:
            n_init = 1.0 / t_ion

        if t_init is None:
            # with complete thermalize limit
            t_1 = t_ion
            n_1, t_1, _ = self.solve_n(n_init, t_1, True, False, **kwargs)
            n_1, t_1, res_1 = self.solve_t(n_1, t_1, True, False, **kwargs)

            t_2 = np.ones_like(t_ion) * t_edge
            n_2, t_2, _ = self.solve_n(n_init, t_2, True, False, **kwargs)
            n_2, t_2, res_2 = self.solve_t(n_2, t_2, True, False, **kwargs)

            n_3, t_3, res_3 = self.solve_t((n_1 + n_2) / 2, t_2, True, False,
                                           **kwargs)

            n_inits = [n_1, n_2, n_3]
            t_inits = [t_1, t_2, t_3]
            errors = [res_1['optimality'], res_2['optimality'],
                      res_3['optimality']]
            n_init = n_inits[np.argmin(errors)]
            t_init = t_inits[np.argmin(errors)]
        return self.solve_nt(n_init, t_init, True, always_positive, **kwargs)

    def initialize(self, rate_ion, rate_cx, t_ion, t_edge):
        for v in [rate_ion, rate_cx, t_ion]:
            if v.shape != self.r.shape:
                raise ValueError('Shape mismatch, {} and {}'.format(
                                                v.shape, self.r.shape))
        self.kt_ion = EV * t_ion
        self.kt_edge = t_edge * EV / self.kt_ion[-1]

        rmu = self.r / (self.m * (rate_ion + rate_cx))

        r_rion_tion = self.r * rate_ion / self.kt_ion
        rmu_t_ion = vec2coo(rmu * self.kt_ion)
        rmu_t_ion_grad = vec2coo(rmu * np.gradient(self.kt_ion, self.r))
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
        Fijkl = (Fijkl_tmp1 + Fijkl_tmp2 + np.moveaxis(Fijkl_tmp2, 2, 0)
                 + np.moveaxis(Fijkl_tmp2, 2, 1)) * 2.5

        Gijl = -1.5 * sparse.tensordot(vec2coo(self.r * (rate_ion + rate_cx)),
                                       self.phi_ijkl, axes=(0, -1))
        Hil = 1.5 * sparse.tensordot(vec2coo(self.r * rate_cx),
                                     self.phi_ijk, axes=(0, -1))

        # Remove last item of last dimension
        self.Dijl = sparse.tensordot(Dijl, self.slice_l, axes=(-1, 0))
        self.Ril = sparse.tensordot(Ril, self.slice_l, axes=(-1, 0))
        self.Fijkl = sparse.tensordot(Fijkl, self.slice_l, axes=(-1, 0))
        self.Gijl = sparse.tensordot(Gijl, self.slice_l, axes=(-1, 0))
        self.Hil = sparse.tensordot(Hil, self.slice_l, axes=(-1, 0))

    def get_n(self, x, always_positive):
        n = np.exp(x) if always_positive else x
        return vec2coo(np.concatenate([n, [1.0]], axis=0))

    def get_t(self, x, always_positive):
        t = np.exp(x) if always_positive else x
        return vec2coo(np.concatenate([t, [self.kt_edge]], axis=0))

    def _fun_particle(self, n, t):
        Dij = sparse.tensordot(t, self.Dijl, axes=(0, 1))
        return sparse.tensordot(n, Dij - self.Ril, axes=(0, 0))

    def _fun_energy(self, n, t):
        Fij = sparse.tensordot(t, sparse.tensordot(
                    t, self.Fijkl, axes=(0, 2)), axes=(0, 1))
        Gil = sparse.tensordot(t, self.Gijl, axes=(0, 1))
        return sparse.tensordot(n, Fij - Gil - self.Hil, axes=(0, 0))

    def _jac_n_particle(self, n, t):
        # returns a jacobian for density. dfi_dnj
        Dij = sparse.tensordot(t, self.Dijl, axes=(0, 1))
        jac_n = Dij - self.Ril
        return sparse.tensordot(jac_n, self.slice_l, axes=(0, 0))

    def _jac_t_particle(self, n, t):
        # returns a jacobian for density. dfi_dtj
        jac_t = sparse.tensordot(n, self.Dijl, axes=(0, 0))
        return sparse.tensordot(jac_t, self.slice_l, axes=(0, 0))

    def _jac_n_energy(self, n, t):
        Fij = sparse.tensordot(t, sparse.tensordot(
                    t, self.Fijkl, axes=(0, 2)), axes=(0, 1))
        Gil = sparse.tensordot(t, self.Gijl, axes=(0, 1))
        jac_n = Fij - Gil - self.Hil
        return sparse.tensordot(jac_n, self.slice_l, axes=(0, 0))

    def _jac_t_energy(self, n, t):
        Fij = sparse.tensordot(n, sparse.tensordot(
                    t, self.Fijkl, axes=(0, 2)), axes=(0, 0))
        Gil = sparse.tensordot(n, self.Gijl, axes=(0, 0))
        jac_t = 2.0 * Fij - Gil
        return sparse.tensordot(jac_t, self.slice_l, axes=(0, 0))

    def solve_n(self, n_init, t_init, use_jac, always_positive, **kwargs):
        def fun(x):
            res = self._fun_particle(self.get_n(x, always_positive),
                                     vec2coo(t_init))
            return res.todense() if isinstance(res, sparse.COO) else res

        def jac(x):
            jac_n = self._jac_n_particle(self.get_n(x, always_positive),
                                         vec2coo(t_init))
            if always_positive:
                jac_n *= vec2coo(np.exp(x))
            return jac_n.todense()

        # initial guess
        n_init = n_init * self.kt_ion / self.kt_ion[-1]
        t_init = t_init * EV / self.kt_ion
        x_init = np.log(n_init[:-1]) if always_positive else n_init[:-1]
        if use_jac:
            res = scipy.optimize.least_squares(fun, x_init, jac=jac, **kwargs)
        else:
            res = scipy.optimize.least_squares(fun, x_init, **kwargs)
        n = self.get_n(res['x'], always_positive).todense()
        n = n / self.kt_ion * self.kt_ion[-1]
        t = t_init * self.kt_ion / EV
        return n, t, res

    def solve_t(self, n_init, t_init, use_jac, always_positive, **kwargs):
        def fun(x):
            res = self._fun_energy(vec2coo(n_init),
                                   self.get_t(x, always_positive))
            return res.todense() if isinstance(res, sparse.COO) else res

        def jac(x):
            jac_t = self._jac_t_energy(vec2coo(n_init),
                                       self.get_t(x, always_positive))
            if always_positive:
                jac_t *= vec2coo(np.exp(x))
            return jac_t.todense()

        # initial guess
        n_init = n_init * self.kt_ion / self.kt_ion[-1]
        t_init = t_init * EV / self.kt_ion
        x_init = np.log(t_init[:-1]) if always_positive else t_init[:-1]
        if use_jac:
            res = scipy.optimize.least_squares(fun, x_init, jac=jac, **kwargs)
        else:
            res = scipy.optimize.least_squares(fun, x_init, **kwargs)
        n = n_init / self.kt_ion * self.kt_ion[-1]
        t = self.get_t(res['x'], always_positive).todense() * self.kt_ion / EV
        return n, t, res

    def solve_nt(self, n_init, t_init, use_jac, always_positive, **kwargs):
        def fun(x):
            n = self.get_n(x[:self.size-1], always_positive)
            t = self.get_t(x[self.size-1:], always_positive)
            return np.concatenate([self._fun_particle(n, t),
                                   self._fun_energy(n, t)])

        def jac(x):
            n = self.get_n(x[:self.size-1], always_positive)
            t = self.get_t(x[self.size-1:], always_positive)
            jac_nt = np.concatenate(
                [np.concatenate([self._jac_n_particle(n, t).todense(),
                                 self._jac_n_energy(n, t).todense()], axis=0),
                 np.concatenate([self._jac_n_particle(n, t).todense(),
                                 self._jac_n_energy(n, t).todense()], axis=0)],
                axis=1)
            if always_positive:
                jac_nt *= np.exp(x)
            return jac_nt

        # initial guess
        n_init = n_init * self.kt_ion / self.kt_ion[-1]
        t_init = t_init * EV / self.kt_ion
        x_init = np.concatenate([n_init[:-1], t_init[:-1]])
        x_init = np.log(x_init) if always_positive else x_init
        if use_jac:
            res = scipy.optimize.least_squares(fun, x_init, jac=jac, **kwargs)
        else:
            res = scipy.optimize.least_squares(fun, x_init, **kwargs)
        n = self.get_n(res['x'][:self.size-1], always_positive)
        n = n.todense() / self.kt_ion * self.kt_ion[-1]
        t = self.get_t(res['x'][self.size-1:], always_positive)
        t = t.todense() * self.kt_ion / EV
        return n, t, res
