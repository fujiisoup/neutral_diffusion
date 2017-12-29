import numpy as np
import scipy
import scipy.optimize
import sparse

from . import basis1d
from .utils import vec2coo


class Model(object):
    """ A base class for a 1-dimensional diffusion model.
    The model is written by

    grad (D grad n) + S n = 0

    where
    D : diffusion coefficient [m^2/s],
    S : source rate [/s].
    """
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
        self.phi_di_dj_k = basis1d.phi_di_dj_k(r)
        self.slice_l = sparse.COO([np.arange(n-1), np.arange(n-1)],
                                  np.ones(n-1), shape=(n, n-1))


class Cylindrical(Model):
    """
    A class to solve a diffusion model in one-dimensional cylindrical system.
    The diffusion equation is

    1/r  d/dr (rD d_dr n) + S n = 0

    where
    D : diffusion coefficient [m^2/s],
    S : source rate [/s].
    """
    def solve(self, diffusion_coef, source, n_init=None,
              use_jac=True, always_positive=False, **kwargs):
        """
        Solve a diffusion equation with particular paramters.

        diffusion_coef: 1d array-like
            Diffusion coefficient [m2/s].
        source: 1d array-like
            Source rate [/s].
        n_init: 1d array-like
            Initial guess of density [m^-3]

        use_jac: Boolean (optional)
            Use Jacobian to solve the diffusion equation.
        always_positive: Boolean (optional)
            Map solution to the positive space. This case, the model becomes
            non-linear.

        Returns
        -------
        n: 1d array-like
            Neutral atom density. This is normalized so that the edge density
            is 1.
        """
        for v in [diffusion_coef, source]:
            if v.shape != self.r.shape:
                raise ValueError('Shape mismatch, {} and {}'.format(
                                                v.shape, self.r.shape))
        if n_init is None:
            n_init = np.ones_like(self.r)

        Dij = sparse.tensordot(vec2coo(self.r * diffusion_coef),
                               self.phi_di_dj_k, axes=(0, -1))
        Rij = -sparse.tensordot(vec2coo(self.r * source),
                                self.phi_ijk, axes=(0, -1))

        # Remove last item of last dimension
        Dij = sparse.tensordot(Dij - Rij, self.slice_l, axes=(-1, 0))

        def fun(x):
            if always_positive:
                x = np.exp(x)
            n = vec2coo(np.concatenate([x, [1.0]], axis=0))
            lhs = sparse.tensordot(n, Dij, axes=(0, 0))
            return lhs

        def jac(x):
            if always_positive:
                x = scipy.sparse.diags(np.exp(x))
                jacobian = sparse.tensordot(Dij, self.slice_l, axes=(0, 0))
                return jacobian.to_scipy_sparse().dot(x)
            else:
                return sparse.tensordot(Dij, self.slice_l, axes=(0, 0))

        def jac_dense(x):
            # TODO The use of the sparse jacobian looks very slow in scipy
            return np.array(jac(x).todense())

        # initial guess
        x_init = np.log(n_init[:-1]) if always_positive else n_init[:-1]
        if use_jac:
            res = scipy.optimize.least_squares(fun, x_init, jac=jac_dense,
                                               **kwargs)
        else:
            res = scipy.optimize.least_squares(fun, x_init, **kwargs)

        n = np.exp(res['x']) if always_positive else res['x']
        return np.concatenate([n, [1.0]]), res['success']
