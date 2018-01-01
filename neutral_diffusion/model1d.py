import numpy as np
import scipy
import scipy.optimize
import scipy.interpolate
import sparse

from . import basis1d
from .utils import vec2coo
from .constants import EV


class Rate(object):
    """ A simple object to store rates for particular atom """
    def __init__(self, r, m):
        if not (np.diff(r) > 0).all():
            raise ValueError('Coordinate should be monotonically increasing.')
        self.r = r.astype(float)
        self.m = m
        n = len(self.r)
        # basises
        self.phi_ijk = basis1d.phi_ijk(r)
        self.phi_ijkl = basis1d.phi_ijkl(r)
        self.phi_i_dj_dk_l = basis1d.phi_i_dj_dk_l(r)
        self.phi_ijk_dl_m = basis1d.phi_ijk_dl_m(r)
        self.phi_ij_dk_dl_m = basis1d.phi_ij_dk_dl_m(r)
        self.slice_l = sparse.COO([np.arange(n-1), np.arange(n-1)],
                                  np.ones(n-1), shape=(n, n-1))

    def initialize(self, rate_dep, rate_cx, t_ion):
        raise NotImplementedError


class Rate_cylindrical(Rate):
    """ A simple object to store rates for particular atom """
    def initialize(self, rate_dep, rate_cx, t_ion):
        self.rate_dep = rate_dep
        self.rate_cx = rate_cx

        kt_ion = EV * t_ion
        size = len(kt_ion)
        rmu = self.r / (self.m * self.rate_dep)
        rmu_t_ion = vec2coo(rmu * kt_ion)
        rmu_t_ion_grad = vec2coo(rmu * np.gradient(kt_ion, self.r))
        rmu = vec2coo(rmu)

        # Components for particle balance equation
        # gradient term
        Dijl_tmp = sparse.tensordot(rmu, self.phi_i_dj_dk_l, axes=(0, -1))
        Dijl = Dijl_tmp + np.moveaxis(Dijl_tmp, 0, 1)
        # depletion term
        Ril = -sparse.tensordot(vec2coo(self.r * self.rate_dep / kt_ion),
                                self.phi_ijk, axes=(0, -1))
        # source term
        Sil = sparse.tensordot(vec2coo(self.r * self.rate_cx / kt_ion),
                               self.phi_ijk, axes=(0, -1))
        # Normalize the order or these tensors
        #Dijl *= EV
        #Ril *= EV
        #Sil *= EV
        # Remove last item of last dimension
        kt_ion_diag = sparse.COO([np.arange(size), np.arange(size)],
                                 kt_ion, shape=(size, size))
        slice_l = sparse.tensordot(kt_ion_diag, self.slice_l, axes=(1, 0))
        # slice_l = self.slice_l * EV
        self.Dijl = sparse.tensordot(Dijl, slice_l, axes=(-1, 0))
        self.Ril = sparse.tensordot(Ril, slice_l, axes=(-1, 0))
        self.Sil = sparse.tensordot(Sil, slice_l, axes=(-1, 0))

        # Components for energy balance equation
        Fijkl_tmp1 = sparse.tensordot(rmu_t_ion_grad, self.phi_ijk_dl_m,
                                      axes=(0, -1))
        Fijkl_tmp2 = sparse.tensordot(rmu_t_ion, self.phi_ij_dk_dl_m,
                                      axes=(0, -1))
        Fijkl = (Fijkl_tmp1 + Fijkl_tmp2 + np.moveaxis(Fijkl_tmp2, 2, 0)
                 + np.moveaxis(Fijkl_tmp2, 2, 1)) * 2.5

        Gijl = -1.5 * sparse.tensordot(vec2coo(self.r * self.rate_dep),
                                       self.phi_ijkl, axes=(0, -1))
        Hil = 1.5 * sparse.tensordot(vec2coo(self.r * rate_cx),
                                     self.phi_ijk, axes=(0, -1))
        # Remove last item of last dimension
        self.Fijkl = sparse.tensordot(Fijkl, self.slice_l, axes=(-1, 0))
        self.Gijl = sparse.tensordot(Gijl, self.slice_l, axes=(-1, 0))
        self.Hil = sparse.tensordot(Hil, self.slice_l, axes=(-1, 0))


class Cylindrical(object):
    """
    A class to solve a neutral diffusion model in one-dimensional cylindrical
    plasmas.
    """
    def __init__(self, r, m):
        self.rate = Rate_cylindrical(r, m)
        self.r = self.rate.r
        self.size = len(self.r)

    def solve(self, rate_ion, rate_cx, t_ion, t_edge, n_edge=1.0,
              n_init=None, t_init=None,
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
        self.initialize(rate_ion, rate_cx, t_ion, t_edge, n_edge)

        if n_init is None:
            n_therm = 1.0 / t_ion
            # penetration view
            v_th = np.sqrt(EV * t_edge / self.rate.m)
            integrand = rate_ion / v_th
            integ = [np.trapz(integrand[::-1][:i], self.r[::-1][:i]) for i in
                     range(self.size)]
            n_penet = 1.0 * np.exp(np.array(integ[::-1]))

        if t_init is None:
            # with complete thermalize limit
            n_1 = (n_therm + n_penet) / 2.0
            t_1 = t_ion
            n_1, t_1, _ = self.solve_t(n_1, t_1, True, False, **kwargs)
            t_1 = np.minimum(np.maximum(np.min(t_ion), t_1), np.max(t_ion))
            n_1, t_1, _ = self.solve_n(n_1, t_1, True, False, **kwargs)
            n_1, t_1, res_1 = self.solve_t(n_1, t_1, True, False, **kwargs)
            t_1 = np.minimum(np.maximum(np.min(t_ion), t_1), np.max(t_ion))
            return self.solve_nt(n_1, t_1, True, always_positive, **kwargs)

            n_2 = n_init
            t_2 = np.ones_like(t_ion) * t_edge
            n_2, t_2, _ = self.solve_t(n_2, t_2, True, True, **kwargs)
            n_2, t_2, _ = self.solve_n(n_2, t_2, True, False, **kwargs)
            n_2, t_2, res_2 = self.solve_t(n_2, t_2, True, True, **kwargs)

            n_3, t_3, res_3 = self.solve_t((n_1 + n_2) / 2, t_2, True, True,
                                           **kwargs)

            n_inits = [n_1, n_2, n_3]
            t_inits = [t_1, t_2, t_3]
            errors = [res_1['optimality'], res_2['optimality'],
                      res_3['optimality']]
            n_init = n_inits[np.argmin(errors)]
            t_init = t_inits[np.argmin(errors)]
        return self.solve_nt(n_init, t_init, True, always_positive, **kwargs)

    def initial_guess(self, r, m, rate, t_ion, t_edge):
        """ make an initial guess based on the mean free path consideration
        t_ion: ion temperature [eV]
        m: mass of atom [kg]
        rate: ionization rate [/s]
        """
        def moving_average(r, f, w, rmin, rmax, num):
            r_interp = np.linspace(rmin, rmax, num)
            f_interp = scipy.interpolate.interp1d(r, f)
            f_interp = np.array([f_interp(ri) if (r[0] < ri and ri < r[-1])
                                 else f[0] if ri <= r[0] else f[-1]
                                 for ri in r_interp])
            avg = np.zeros_like(r)
            for i in range(len(r)):
                weight = np.exp(-0.5 * ((r_interp - r[i]) / w[i])**2)
                avg[i] = np.trapz(f_interp * weight,
                                  r_interp) / np.trapz(weight, r_interp)
            return avg

        t = np.ones_like(r) * t_edge
        n = np.ones_like(r)
        for _ in range(1):
            n[:-1] = 1.0 / t[:-1]
            # mean free path with t_edge
            tau = np.sqrt(EV * t / m) / rate
            # weighted sum average of t
            t_new = moving_average(r, n * np.log(t_ion), tau, -r[-1], 2*r[-1],
                                   100)
            t = np.exp(t_new / n)
        return n, t

    def initialize(self, rate_ion, rate_cx, t_ion, t_edge, n_edge):
        for v in [rate_ion, rate_cx, t_ion]:
            if v.shape != self.r.shape:
                raise ValueError('Shape mismatch, {} and {}'.format(
                                                v.shape, self.r.shape))
        self.rate.initialize(rate_ion + rate_cx, rate_cx, t_ion)
        self.kt_ion = EV * t_ion
        self.kt_edge = t_edge * EV / self.kt_ion[-1]
        self.n_edge = n_edge

    def get_n(self, x, always_positive):
        n = np.exp(x) if always_positive else x
        return vec2coo(np.concatenate([n, [self.n_edge]], axis=0))

    def get_t(self, x, always_positive):
        t = np.exp(x) if always_positive else x
        return vec2coo(np.concatenate([t, [self.kt_edge]], axis=0))

    def _fun_particle(self, n, t):
        Dij = sparse.tensordot(t, self.rate.Dijl, axes=(0, 1))
        return sparse.tensordot(n, Dij - self.rate.Ril - self.rate.Sil,
                                axes=(0, 0))

    def _fun_energy(self, n, t):
        Fij = sparse.tensordot(t, sparse.tensordot(
                    t, self.rate.Fijkl, axes=(0, 2)), axes=(0, 1))
        Gil = sparse.tensordot(t, self.rate.Gijl, axes=(0, 1))
        return sparse.tensordot(n, Fij - Gil - self.rate.Hil, axes=(0, 0))

    def _jac_n_particle(self, n, t):
        # returns a jacobian for density. dfi_dnj
        Dij = sparse.tensordot(t, self.rate.Dijl, axes=(0, 1))
        jac_n = Dij - self.rate.Ril - self.rate.Sil
        return sparse.tensordot(jac_n, self.rate.slice_l, axes=(0, 0))

    def _jac_t_particle(self, n, t):
        # returns a jacobian for density. dfi_dtj
        jac_t = sparse.tensordot(n, self.rate.Dijl, axes=(0, 0))
        return sparse.tensordot(jac_t, self.rate.slice_l, axes=(0, 0))

    def _jac_n_energy(self, n, t):
        Fij = sparse.tensordot(t, sparse.tensordot(
                    t, self.rate.Fijkl, axes=(0, 2)), axes=(0, 1))
        Gil = sparse.tensordot(t, self.rate.Gijl, axes=(0, 1))
        jac_n = Fij - Gil - self.rate.Hil
        return sparse.tensordot(jac_n, self.rate.slice_l, axes=(0, 0))

    def _jac_t_energy(self, n, t):
        Fij = sparse.tensordot(n, sparse.tensordot(
                    t, self.rate.Fijkl, axes=(0, 2)), axes=(0, 0))
        Gil = sparse.tensordot(n, self.rate.Gijl, axes=(0, 0))
        jac_t = 2.0 * Fij - Gil
        return sparse.tensordot(jac_t, self.rate.slice_l, axes=(0, 0))

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
        n_init = n_init * self.kt_ion / self.kt_ion[-1] * self.n_edge
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

    def update_n(self, n_init, t_init, use_jac, always_positive):
        def fun(x):
            n = self.get_n(x[:self.size-1], always_positive)
            t = self.get_t(x[self.size-1:], always_positive)
            return np.concatenate([self._fun_particle(n, t), vec2coo(t_init)])

        def jac(x):
            pass


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
        n_init = n_init * self.kt_ion / self.kt_ion[-1] * self.n_edge
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
        n_init = n_init * self.kt_ion / self.kt_ion[-1] * self.n_edge
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


class Cylindrical_mixed(object):
    """
    A class to solve a neutral diffusion model in one-dimensional cylindrical
    plasmas, with two isotope atoms, such as hydrogen and deuterium.
    """
    def __init__(self, r, m1, m2):
        self.rate1 = Rate_cylindrical(r, m1)
        self.rate2 = Rate_cylindrical(r, m2)
        self.r = self.rate1.r
        self.size = len(self.r)
        n = self.size

    def solve(self, rate_ion, rate1_cx, rate2_cx, t_ion, t1_edge, t2_edge,
              n1_init=None, n2_init=None, t1_init=None, t2_init=None,
              always_positive=False, **kwargs):
        """
        Solve a diffusion equation with particular paramters.

        rate_ion: 1d array-like
            Ionization rate [/s].
        rate_cx, rate2_cx: 1d array-like
            Charge exchange rate [/s] for atom1 and atom2.
        t_ion: 1d array-like
            Ion temperature in plasmas [eV]
        t_edge and t2_edge: float
            Edge temperature of atoms [eV] for atom1 and atom2
        n_init, n2_init: 1d array-like
            Initial guess of atom density [m^-3]
        t_init, t2_init: 1d array-like
            Initial guess of atom temperature [eV]

        Returns
        -------
        n: 1d array-like
            Neutral atom density. This is normalized so that the edge density
            is 1.
        t_atom: 1d array-like
            Neutral atom temperature.
        """
        self.initialize(rate_ion, rate1_cx, rate2_cx, t_ion, t1_edge, t2_edge)

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

    def initialize(self, rate_ion, rate1_cx, rate2_cx, t_ion,
                   t1_edge, t2_edge, n1_edge, n2_edge):
        for v in [rate_ion, rate1_cx, rate2_cx, t_ion]:
            if v.shape != self.r.shape:
                raise ValueError('Shape mismatch, {} and {}'.format(
                                                v.shape, self.r.shape))
        self.rate1.initialize(rate_ion + rate1_cx + rate2_cx, rate1_cx, t_ion)
        self.rate2.initialize(rate_ion + rate1_cx + rate2_cx, rate2_cx, t_ion)
        self.kt_ion = EV * t_ion
        self.kt1_edge = t1_edge * EV / self.kt_ion[-1]
        self.kt2_edge = t2_edge * EV / self.kt_ion[-1]
        self.n1_edge = n1_edge
        self.n2_edge = n2_edge

    def get_n(self, x, always_positive):
        n = np.exp(x) if always_positive else x
        return (vec2coo(np.concatenate([n[:self.size-1], [self.n1_edge]])),
                vec2coo(np.concatenate([n[self.size-1:], [self.n2_edge]])))

    def get_t(self, x, always_positive):
        t = np.exp(x) if always_positive else x
        return (vec2coo(np.concatenate([t[:self.size-1], [self.kt1_edge]])),
                vec2coo(np.concatenate([t[self.size-1:], [self.kt2_edge]])))

    def _fun_particle(self, n, t):
        n1, n2 = n
        t1, t2 = t
        # for atom1
        Dij = sparse.tensordot(t1, self.rate1.Dijl, axes=(0, 1))
        res1 = sparse.tensordot(n1, Dij - self.rate1.Ril + self.rate1.Sil,
                                axes=(0, 0))
        res1 += sparse.tensordot(n2, self.rate1.Ril, axes=(0, 0))
        # for atom2
        Dij = sparse.tensordot(t2, self.rate2.Dijl, axes=(0, 1))
        res2 = sparse.tensordot(n2, Dij - self.rate2.Ril + self.rate2.Sil,
                                axes=(0, 0))
        res2 += sparse.tensordot(n1, self.rate2.Ril, axes=(0, 0))
        return np.concatenate([res1, res2])

    def _fun_energy(self, n, t):
        n1, n2 = n
        t1, t2 = t
        # for atom1
        Fij = sparse.tensordot(t1, sparse.tensordot(
                    t1, self.rate1.Fijkl, axes=(0, 2)), axes=(0, 1))
        Gil = sparse.tensordot(t1, self.rate1.Gijl, axes=(0, 1))
        res1 = sparse.tensordot(n1, Fij - Gil - self.rate1.Hil, axes=(0, 0))
        res1 -= sparse.tensordot(n2, self.rate1.Hil, axes=(0, 0))
        # for atom2
        Fij = sparse.tensordot(t2, sparse.tensordot(
                    t2, self.rate2.Fijkl, axes=(0, 2)), axes=(0, 1))
        Gil = sparse.tensordot(t2, self.rate2.Gijl, axes=(0, 1))
        res2 = sparse.tensordot(n2, Fij - Gil - self.rate2.Hil, axes=(0, 0))
        res2 -= sparse.tensordot(n1, self.rate2.Hil, axes=(0, 0))
        return np.concatenate([res1, res2])

    def _jac_n_particle(self, n, t):
        # returns a jacobian for density. dfi_dnj
        n1, n2 = n
        t1, t2 = t
        # derivative by atom1
        Dij = sparse.tensordot(t1, self.rate1.Dijl, axes=(0, 1))
        df1_dn1 = Dij - self.rate1.Ril + self.rate1.Sil
        df2_dn1 = self.rate2.Ril
        df_dn1 = sparse.tensordot(
                sparse.concatenate([df1_dn1, df2_dn1], axis=1),
                self.rate1.slice_l, axes=(0, 0))

        Dij = sparse.tensordot(t2, self.rate2.Dijl, axes=(0, 1))
        df2_dn2 = Dij - self.rate2.Ril + self.rate2.Sil
        df1_dn2 = self.rate1.Ril
        df_dn2 = sparse.tensordot(
                sparse.concatenate([df1_dn2, df2_dn2], axis=1),
                self.rate1.slice_l, axes=(0, 0))
        return sparse.concatenate([df_dn1, df_dn2], axis=1)

    def _jac_t_particle(self, n, t):
        # returns a jacobian for density. dfi_dnj
        n1, n2 = n
        t1, t2 = t
        Dij = sparse.tensordot(n1, self.rate1.Dijl, axes=(0, 0))
        df_dt1 = sparse.tensordot(
                sparse.concatenate([
                    Dij, sparse.COO(None, shape=(self.size, self.size-1))
                ], axis=1), self.rate1.slice_l, axes=(0, 0))

        Dij = sparse.tensordot(n2, self.rate2.Dijl, axes=(0, 0))
        df_dt2 = sparse.tensordot(
                sparse.concatenate([
                    sparse.COO(None, shape=(self.size, self.size-1)), Dij
                ], axis=1), self.rate2.slice_l, axes=(0, 0))
        return sparse.concatenate([df_dt1, df_dt2], axis=1)

    def _jac_n_energy(self, n, t):
        n1, n2 = n
        t1, t2 = t
        # dn1
        Fij = sparse.tensordot(t1, sparse.tensordot(
                    t1, self.rate1.Fijkl, axes=(0, 2)), axes=(0, 1))
        Gil = sparse.tensordot(t1, self.rate1.Gijl, axes=(0, 1))
        df1_dn1 = Fij - Gil - self.rate1.Hil
        df2_dn1 = -self.rate2.Hil
        df_dn1 = sparse.tensordot(
                sparse.concatenate([df1_dn1, df2_dn1], axis=1),
                self.rate1.slice_l, axes=(0, 0))
        # dn2
        Fij = sparse.tensordot(t2, sparse.tensordot(
                    t2, self.rate2.Fijkl, axes=(0, 2)), axes=(0, 1))
        Gil = sparse.tensordot(t2, self.rate2.Gijl, axes=(0, 1))
        df2_dn2 = Fij - Gil - self.rate2.Hil
        df1_dn2 = -self.rate1.Hil
        df_dn2 = sparse.tensordot(
                sparse.concatenate([df1_dn2, df2_dn2], axis=1),
                self.rate2.slice_l, axes=(0, 0))
        return sparse.concatenate([df_dn1, df_dn2], axis=1)

    def _jac_t_energy(self, n, t):
        n1, n2 = n
        t1, t2 = t

        Fij = 2.0 * sparse.tensordot(n1, sparse.tensordot(
                    t1, self.rate1.Fijkl, axes=(0, 2)), axes=(0, 0))
        Gil = sparse.tensordot(n1, self.rate1.Gijl, axes=(0, 0))
        df_dt1 = sparse.tensordot(
                sparse.concatenate([
                    Fij - Gil, sparse.COO(None, shape=(self.size, self.size-1))
                ], axis=1), self.rate1.slice_l, axes=(0, 0))

        Fij = 2.0 * sparse.tensordot(n2, sparse.tensordot(
                    t2, self.rate2.Fijkl, axes=(0, 2)), axes=(0, 0))
        Gil = sparse.tensordot(n2, self.rate2.Gijl, axes=(0, 0))
        df_dt2 = sparse.tensordot(
                sparse.concatenate([
                    sparse.COO(None, shape=(self.size, self.size-1)), Fij - Gil
                ], axis=1), self.rate1.slice_l, axes=(0, 0))
        return sparse.concatenate([df_dt1, df_dt2], axis=1)

    def solve_n(self, n1_init, n2_init, t1_init, t2_init, use_jac,
                always_positive, **kwargs):
        def fun(x):
            return self._fun_particle(self.get_n(x, always_positive),
                                      (vec2coo(t1_init), vec2coo(t2_init)))

        def jac(x):
            jac_n = self._jac_n_particle(self.get_n(x, always_positive),
                                         (vec2coo(t1_init), vec2coo(t2_init)))
            if always_positive:
                jac_n *= vec2coo(np.exp(x))
            return jac_n.todense()

        # initial guess
        n1_init = n2_init * self.kt_ion / self.kt_ion[-1] * self.n1_edge
        n2_init = n2_init * self.kt_ion / self.kt_ion[-1] * self.n2_edge
        x_init = np.concatenate([n1_init[:-1], n2_init[:-1]])
        x_init = np.log(x_init) if always_positive else x_init

        t1_init = t1_init * EV / self.kt_ion
        t2_init = t2_init * EV / self.kt_ion
        if use_jac:
            res = scipy.optimize.least_squares(fun, x_init, jac=jac, **kwargs)
        else:
            res = scipy.optimize.least_squares(fun, x_init, **kwargs)
        n1, n2 = self.get_n(res['x'][:self.size-1], always_positive)
        n1 = n1.todense() / self.kt_ion * self.kt_ion[-1]
        n2 = n2.todense() / self.kt_ion * self.kt_ion[-1]
        t1 = t1_init * self.kt_ion / EV
        t2 = t2_init * self.kt_ion / EV
        return n1, n2, t1, t2, res

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
