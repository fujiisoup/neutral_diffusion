import pytest
import numpy as np
import scipy as sp

from neutral_diffusion import solvers


rng = np.random.RandomState(0)
n = 30
ind = rng.randint(0, n, size=(2, 50))
mat1 = sp.sparse.coo_matrix((rng.randn(50), ind), shape=(n, n))
mat2 = sp.sparse.coo_matrix((rng.randn(50), ind), shape=(n, n))
b = rng.randn(n)
x0 = rng.randn(n)


def assert_close(x, y, **kwargs):
    assert np.allclose(x, y)


def loss(A, x, bvec):
    """
    Calculate loss |A(x) x - b|^2 with a normalization.
    """
    Amat = A(x)
    x = x.reshape(-1, 1)
    # normalize
    Adiag = Amat.diagonal(0).reshape(-1, 1)
    bvec = bvec.reshape(-1, 1) / Adiag
    Amat = Amat / Adiag
    return np.sum(np.square(np.dot(Amat, x) - bvec))


def assert_minimum(A, x, bvec, tol=1.0e-5, n=100):
    current_loss = loss(A, x, bvec)
    for _ in range(n):
        dx = np.random.randn(*x.shape) * tol
        assert current_loss <= loss(A, x + dx, bvec)


def constant_A(x):
    return mat1 + sp.sparse.identity(n) * 0.8


def linear_A(x):
    return sp.sparse.identity(n) + 0.1 * mat2 * x.reshape(-1, 1)


@pytest.mark.parametrize('A', [constant_A, linear_A])
@pytest.mark.parametrize('method', ['iterative'])
def test_solvers_iterative(A, method):
    it, x = solvers.solve(A, x0, b, tol=1.0e-5, method=method)
    assert_minimum(A, x, b, tol=1.0e-5)
