import numpy as np
import scipy as sp


def solve(fun, jac, x, maxiter=1000, tol=1.0e-5, method='neuton'):
    """
    Solve non-linear equation A(x) x = b.

    Parameters
    ----------
    A: callable
        Returns A(x) with given x.
        A should be scipy.sparse.coo_matrix or np.ndarray.
        Size of A(x) should be [n, n].
    x: 1d-array sized [n]
        An initial guess of the solution.
    b: 1d-.array sized [n]

    maxiter: integer
        Maximum number os updates.
    tol: float
        Convergence criteria.

    Returns
    -------
    x: np.array sized [n]
    """
    # reshape x, b to [n, 1]
    b = b.reshape(-1, 1)
    x = x.reshape(-1, 1)

    if method == 'iterative':
        it, x = solve_iterative(A, x, b, maxiter, tol, inner_iter)
        return it, x.reshape(-1)
    elif method == 'scipy':
        raise NotImplementedError
