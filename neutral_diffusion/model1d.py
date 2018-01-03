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
        self.phi_di_dj_k = basis1d.phi_di_dj_k(r)
        self.slice_l = sparse.COO([np.arange(n-1), np.arange(n-1)],
                                  np.ones(n-1), shape=(n, n-1))
        self.slice_last = sparse.COO([(n-1, )], [1.0], shape=(n, ))

    def initialize(self, rate_dep, rate_cx, t_ion):
        raise NotImplementedError


class Rate_cylindrical(Rate):
    """ A simple object to store rates for particular atom """
    def initialize(self, rate_dep, rate_cx, t_ion):
        self.rate_dep = rate_dep
        self.rate_cx = rate_cx

        rmu = self.r / (self.m * self.rate_dep)
        rmuk = vec2coo(rmu * EV)

        # Components for particle balance equation
        # gradient term
        Dij = sparse.tensordot(rmuk, self.phi_di_dj_k, axes=(0, -1))
        # depletion term by ionization and charge exchange
        Rij = -sparse.tensordot(vec2coo(self.r * self.rate_dep),
                                self.phi_ijk, axes=(0, -1))
        # source term by charge exchange
        Sij = sparse.tensordot(vec2coo(self.r * self.rate_cx),
                               self.phi_ijk, axes=(0, -1))
        # energy source term by charge exchange
        Eij = sparse.tensordot(vec2coo(self.r * self.rate_cx * t_ion),
                               self.phi_ijk, axes=(0, -1))
        # Remove last item of last dimension
        self.Dij = sparse.tensordot(Dij, self.slice_l, axes=(-1, 0))
        self.Rij = sparse.tensordot(Rij, self.slice_l, axes=(-1, 0))
        self.Sij = sparse.tensordot(Sij, self.slice_l, axes=(-1, 0))
        self.Eij = sparse.tensordot(Eij, self.slice_l, axes=(-1, 0))


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
        n_init = 1.0 / t_ion * t_ion[-1]
        t_init = t_ion / t_ion[-1] * t_edge
        return self.solve_core(n_init, t_init)

    def initialize(self, rate_ion, rate_cx, t_ion, t_edge, n_edge):
        for v in [rate_ion, rate_cx, t_ion]:
            if v.shape != self.r.shape:
                raise ValueError('Shape mismatch, {} and {}'.format(
                                                v.shape, self.r.shape))
        self.rate.initialize(rate_ion + rate_cx, rate_cx, t_ion)
        self.t_ion = t_ion
        self.n_edge = n_edge
        self.nt_edge = n_edge * t_edge
        self.ntt_edge = n_edge * t_edge * t_edge

        n = self.size-1
        # prepare matrices
        # matrix for the particle balance
        A_part = sparse.concatenate([
            -sparse.tensordot(self.rate.Rij + self.rate.Sij, self.rate.slice_l,
                              axes=(0, 0)),
            sparse.tensordot(self.rate.Dij, self.rate.slice_l, axes=(0, 0)),
            sparse.COO([], [], shape=(n, n))
        ], axis=0).T
        # matrix for the energy balance
        A_engy = sparse.concatenate([
            -1.5 * sparse.tensordot(self.rate.Eij, self.rate.slice_l,
                                    axes=(0, 0)),
            -1.5 * sparse.tensordot(self.rate.Rij, self.rate.slice_l,
                                    axes=(0, 0)),
            2.5 * sparse.tensordot(self.rate.Dij, self.rate.slice_l,
                                   axes=(0, 0))
        ], axis=0).T
        # balance matrix.
        self.A = sparse.concatenate([A_part, A_engy], axis=0)

        # boundary conditions
        b_part = (- self.n_edge * sparse.tensordot(
                        self.rate.Rij + self.rate.Sij,
                        self.rate.slice_last, axes=(0, 0))
                  + self.nt_edge * sparse.tensordot(
                        self.rate.Dij,
                        self.rate.slice_last, axes=(0, 0))).todense()

        b_engy = (- 1.5 * self.n_edge * sparse.tensordot(
                        self.rate.Eij, self.rate.slice_last,
                        axes=(0, 0))
                  - 1.5 * self.nt_edge * sparse.tensordot(
                        self.rate.Rij, self.rate.slice_last,
                        axes=(0, 0))
                  + 2.5 * self.ntt_edge * sparse.tensordot(
                        self.rate.Dij, self.rate.slice_last,
                        axes=(0, 0))).todense()
        self.b = - np.concatenate([b_part, b_engy])

        # matrix for the constraint
        self.L = scipy.sparse.hstack([scipy.sparse.identity(n),
                                      scipy.sparse.identity(n) * (-2.0),
                                      scipy.sparse.identity(n)])

    def solve_core(self, n_init, t_init, maxiter=100, rho=1.0, xtol=1.0e-3):
        # initial values
        x = np.concatenate([np.log(n_init[:-1]),
                            np.log(n_init[:-1]) + np.log(t_init[:-1]),
                            np.log(n_init[:-1]) + 2 * np.log(t_init[:-1])])
        # Lagrange constants
        lam = np.zeros(self.size-1)

        # TODO take care the very high density case.
        # If n << 1, then the matrix Ay would be singular.
        for it in range(maxiter):
            # update by neuton method
            y = np.exp(x)
            Ay = (self.A * vec2coo(y)).to_scipy_sparse()
            X = Ay.T @ Ay + rho * self.L.T @ self.L
            coef = - Ay.T @ (self.A @ y - self.b)
            coef += self.L.T @ lam
            coef += -rho * self.L.T @ self.L @ x
            x_new = x + scipy.sparse.linalg.spsolve(X, coef)

            if np.max(np.abs(x_new - x)) < xtol:
                x = x_new
                break

            x = x_new
            lam -= rho * (self.L @ x)

        n = np.concatenate([np.exp(x[:self.size-1]), [self.n_edge]])
        nt = np.concatenate([np.exp(x[self.size-1:2*(self.size-1)]),
                             [self.nt_edge]])
        return n, nt / n, it


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
        raise NotImplementedError

    def initialize(self, rate_ion, rate1_cx, rate2_cx, t_ion,
                   t1_edge, t2_edge, n1_edge, n2_edge):
        for v in [rate_ion, rate1_cx, rate2_cx, t_ion]:
            if v.shape != self.r.shape:
                raise ValueError('Shape mismatch, {} and {}'.format(
                                                v.shape, self.r.shape))
        self.rate1.initialize(rate_ion + rate1_cx + rate2_cx, rate1_cx, t_ion)
        self.rate2.initialize(rate_ion + rate1_cx + rate2_cx, rate2_cx, t_ion)
        self.t_ion = t_ion
        self.n1_edge = n1_edge
        self.n2_edge = n2_edge
        self.nt1_edge = n1_edge * t1_edge
        self.nt2_edge = n2_edge * t2_edge
        self.ntt1_edge = n1_edge * t1_edge * t1_edge
        self.ntt2_edge = n2_edge * t2_edge * t2_edge

        n = self.size-1
        # prepare matrices
        # particle balance
        # atom1
        A_part1 = sparse.concatenate([
            -sparse.tensordot(self.rate1.Rij + self.rate1.Sij,
                              self.rate1.slice_l, axes=(0, 0)),
            sparse.tensordot(self.rate1.Dij, self.rate1.slice_l, axes=(0, 0)),
            sparse.COO([], [], shape=(n, n)),
            -sparse.tensordot(self.rate1.Sij, self.rate1.slice_l, axes=(0, 0)),
            sparse.COO([], [], shape=(n, n)),
            sparse.COO([], [], shape=(n, n)),
        ], axis=0).T
        # atom2
        A_part2 = sparse.concatenate([
            -sparse.tensordot(self.rate2.Sij, self.rate2.slice_l, axes=(0, 0)),
            sparse.COO([], [], shape=(n, n)),
            sparse.COO([], [], shape=(n, n)),
            -sparse.tensordot(self.rate2.Rij + self.rate2.Sij,
                              self.rate2.slice_l, axes=(0, 0)),
            sparse.tensordot(self.rate2.Dij, self.rate2.slice_l, axes=(0, 0)),
            sparse.COO([], [], shape=(n, n)),
        ], axis=0).T

        # energy balance
        # atom1
        A_engy1 = sparse.concatenate([
            -1.5 * sparse.tensordot(self.rate1.Eij, self.rate1.slice_l,
                                    axes=(0, 0)),
            -1.5 * sparse.tensordot(self.rate1.Rij, self.rate1.slice_l,
                                    axes=(0, 0)),
            2.5 * sparse.tensordot(self.rate1.Dij, self.rate1.slice_l,
                                   axes=(0, 0)),
            -1.5 * sparse.tensordot(self.rate1.Eij, self.rate2.slice_l,
                                    axes=(0, 0)),
            sparse.COO([], [], shape=(n, n)),
            sparse.COO([], [], shape=(n, n)),
        ], axis=0).T
        # atom2
        A_engy2 = sparse.concatenate([
            -1.5 * sparse.tensordot(self.rate2.Eij, self.rate1.slice_l,
                                    axes=(0, 0)),
            sparse.COO([], [], shape=(n, n)),
            sparse.COO([], [], shape=(n, n)),
            -1.5 * sparse.tensordot(self.rate2.Eij, self.rate2.slice_l,
                                    axes=(0, 0)),
            -1.5 * sparse.tensordot(self.rate2.Rij, self.rate2.slice_l,
                                    axes=(0, 0)),
            2.5 * sparse.tensordot(self.rate2.Dij, self.rate2.slice_l,
                                   axes=(0, 0)),
        ], axis=0).T

        # balance matrix.
        self.A = sparse.concatenate([A_part1, A_part2, A_engy1, A_engy2],
                                    axis=0)
        # boundary conditions
        b_part1 = (- self.n1_edge * sparse.tensordot(
                        self.rate1.Rij + self.rate1.Sij,
                        self.rate1.slice_last, axes=(0, 0))
                   + self.nt1_edge * sparse.tensordot(
                        self.rate1.Dij,
                        self.rate1.slice_last, axes=(0, 0))
                   - self.n2_edge * sparse.tensordot(
                        self.rate1.Sij,
                        self.rate1.slice_last, axes=(0, 0))).todense()

        b_part2 = (- self.n2_edge * sparse.tensordot(
                        self.rate2.Rij + self.rate2.Sij,
                        self.rate2.slice_last, axes=(0, 0))
                   + self.nt2_edge * sparse.tensordot(
                        self.rate2.Dij,
                        self.rate2.slice_last, axes=(0, 0))
                   - self.n1_edge * sparse.tensordot(
                        self.rate2.Sij,
                        self.rate2.slice_last, axes=(0, 0))).todense()

        b_engy1 = (- 1.5 * self.n1_edge * sparse.tensordot(
                        self.rate1.Eij, self.rate1.slice_last,
                        axes=(0, 0))
                   - 1.5 * self.nt1_edge * sparse.tensordot(
                        self.rate1.Rij, self.rate1.slice_last,
                        axes=(0, 0))
                   + 2.5 * self.ntt1_edge * sparse.tensordot(
                        self.rate1.Dij, self.rate1.slice_last,
                        axes=(0, 0))
                   - 1.5 * self.n2_edge * sparse.tensordot(
                        self.rate1.Eij, self.rate1.slice_last,
                        axes=(0, 0))).todense()

        b_engy2 = (- 1.5 * self.n2_edge * sparse.tensordot(
                        self.rate2.Eij, self.rate2.slice_last,
                        axes=(0, 0))
                   - 1.5 * self.nt2_edge * sparse.tensordot(
                        self.rate2.Rij, self.rate2.slice_last,
                        axes=(0, 0))
                   + 2.5 * self.ntt2_edge * sparse.tensordot(
                        self.rate2.Dij, self.rate2.slice_last,
                        axes=(0, 0))
                   - 1.5 * self.n1_edge * sparse.tensordot(
                        self.rate2.Eij, self.rate2.slice_last,
                        axes=(0, 0))).todense()

        self.b = - np.concatenate([b_part1, b_part2, b_engy1, b_engy2])

        # matrix for the constraint
        L1 = scipy.sparse.hstack([scipy.sparse.identity(n),
                                  scipy.sparse.identity(n) * (-2.0),
                                  scipy.sparse.identity(n),
                                  scipy.sparse.coo_matrix((n, n)),
                                  scipy.sparse.coo_matrix((n, n)),
                                  scipy.sparse.coo_matrix((n, n))])
        L2 = scipy.sparse.hstack([scipy.sparse.coo_matrix((n, n)),
                                  scipy.sparse.coo_matrix((n, n)),
                                  scipy.sparse.coo_matrix((n, n)),
                                  scipy.sparse.identity(n),
                                  scipy.sparse.identity(n) * (-2.0),
                                  scipy.sparse.identity(n)])
        self.L = scipy.sparse.vstack([L1, L2])

    def solve_core(self, n1_init, n2_init, t1_init, t2_init, maxiter=100,
                   rho=1.0, xtol=1.0e-3):
        # initial values
        x = np.concatenate([np.log(n1_init[:-1]),
                            np.log(n1_init[:-1]) + np.log(t1_init[:-1]),
                            np.log(n1_init[:-1]) + 2 * np.log(t1_init[:-1]),
                            np.log(n2_init[:-1]),
                            np.log(n2_init[:-1]) + np.log(t2_init[:-1]),
                            np.log(n2_init[:-1]) + 2 * np.log(t2_init[:-1])])
        # Lagrange constants
        n = self.size-1
        lam = np.zeros(2 * n)

        # TODO take care the very high density case.
        # If n << 1, then the matrix Ay would be singular.
        for it in range(maxiter):
            # update by neuton method
            y = np.exp(x)
            Ay = (self.A * vec2coo(y)).to_scipy_sparse()
            X = Ay.T @ Ay + rho * self.L.T @ self.L
            coef = - Ay.T @ (self.A @ y - self.b)
            coef += self.L.T @ lam
            coef += -rho * self.L.T @ self.L @ x
            x_new = x + scipy.sparse.linalg.spsolve(X, coef)

            if np.max(np.abs(x_new - x)) < xtol:
                x = x_new
                break

            x = x_new
            lam -= rho * (self.L @ x)

        n1 = np.concatenate([np.exp(x[:n]), [self.n1_edge]])
        nt1 = np.concatenate([np.exp(x[n:2*n]), [self.nt1_edge]])
        n2 = np.concatenate([np.exp(x[3*n:4*n]), [self.n2_edge]])
        nt2 = np.concatenate([np.exp(x[4*n:5*n]), [self.nt2_edge]])
        return n1, n2, nt1 / n1, nt2 / n2, it
