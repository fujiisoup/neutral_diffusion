import numpy as np

from neutral_diffusion import basis1d


rng = np.random.RandomState(0)
n = 10
x = np.cumsum(np.exp(rng.randn(n) * 0.2) + 1.0) + 3.0
dx = np.diff(x)
dx_inv = 1.0 / dx


def assert_close(x, y, **kwargs):
    assert np.allclose(x, y, **kwargs)


class Phi(object):
    """ 1-st order b-splines """
    def __init__(self, i, x):
        self.x = x
        assert i >= 0 and i < len(x)
        self.i = i

    def __call__(self, x):
        i = self.i
        # x is too small
        if i > 0 and x <= self.x[i-1] or i == 0 and x <= self.x[i]:
            return 0.0
        # x is in the increasing part
        if i > 0 and x <= self.x[i]:
            return (x - self.x[i-1]) / (self.x[i] - self.x[i-1])
        # x is in the decreasing part
        if i < len(self.x) and x <= self.x[i+1]:
            return (self.x[i+1] - x) / (self.x[i+1] - self.x[i])
        return 0.0


class dPhi(Phi):
    """ Derivative of the 1-st order b-splines """
    def __call__(self, x):
        i = self.i
        # x is too small
        if i > 0 and x <= self.x[i-1] or i == 0 and x <= self.x[i]:
            return 0.0
        # x is in the increasing part
        if i > 0 and x <= self.x[i]:
            return 1.0 / (self.x[i] - self.x[i-1])
        # x is in the decreasing part
        if i < len(self.x) and x <= self.x[i+1]:
            return -1.0 / (self.x[i+1] - self.x[i])
        return 0.0


def integrate(*phi_list):
    """ Numerically integrate phi_i * phi_j * ... """
    n_fine = 1000
    x_fine = np.linspace(x[0], x[-1], n_fine)
    integrand = np.ones_like(x_fine)
    for phi in phi_list:
        for i in range(n_fine):
            integrand[i] *= phi(x_fine[i])
    return np.trapz(integrand, x_fine)


def randint_neibor(i):
    j = i + rng.randint(-1, 2)
    if j < 0:
        return 0
    if j >= n:
        return n-1
    return j


def test_phi_ij():
    phi = basis1d.phi_ij(x).todense()
    for (i, j) in rng.randint(0, n, size=(100, 2)):
        assert_close(integrate(Phi(i, x), Phi(j, x)), phi[i, j], atol=1.0e-2)
        j = randint_neibor(i)
        assert_close(integrate(Phi(i, x), Phi(j, x)), phi[i, j], atol=1.0e-2)


def test_phi_ijk():
    phi = basis1d.phi_ijk(x).todense()
    for (i, j, k) in rng.randint(0, n, size=(100, 3)):
        assert_close(integrate(Phi(i, x), Phi(j, x), Phi(k, x)),
                     phi[i, j, k], atol=1.0e-2)
        j = randint_neibor(i)
        k = randint_neibor(i)
        assert_close(integrate(Phi(i, x), Phi(j, x), Phi(k, x)),
                     phi[i, j, k], atol=1.0e-2)


def test_phi_di_dj_k():
    phi = basis1d.phi_di_dj_k(x).todense()
    for (i, j, k) in rng.randint(0, n, size=(100, 3)):
        assert_close(integrate(dPhi(i, x), dPhi(j, x), Phi(k, x)),
                     phi[i, j, k], atol=1.0e-2)
        j = randint_neibor(i)
        k = randint_neibor(i)
        assert_close(integrate(dPhi(i, x), dPhi(j, x), Phi(k, x)),
                     phi[i, j, k], atol=1.0e-2)


def test_phi_ijkl():
    phi = basis1d.phi_ijkl(x).todense()

    for (i, j, k, l) in rng.randint(0, n, size=(100, 4)):
        assert_close(integrate(Phi(i, x), Phi(j, x), Phi(k, x), Phi(l, x)),
                     phi[i, j, k, l], atol=1.0e-2)
        j = randint_neibor(i)
        k = randint_neibor(i)
        l = randint_neibor(i)
        assert_close(integrate(Phi(i, x), Phi(j, x), Phi(k, x), Phi(l, x)),
                     phi[i, j, k, l], atol=1.0e-2)


def test_phi_i_dj_dk_l():
    phi = basis1d.phi_i_dj_dk_l(x).todense()

    for (i, j, k, l) in rng.randint(0, n, size=(100, 4)):
        assert_close(integrate(Phi(i, x), dPhi(j, x), dPhi(k, x), Phi(l, x)),
                     phi[i, j, k, l], atol=1.0e-2)
        j = randint_neibor(i)
        k = randint_neibor(i)
        l = randint_neibor(i)
        assert_close(integrate(Phi(i, x), dPhi(j, x), dPhi(k, x), Phi(l, x)),
                     phi[i, j, k, l], atol=1.0e-2)


def test_phi_ijk_dl_m():
    phi = basis1d.phi_ijk_dl_m(x).todense()

    for (i, j, k, l, m) in rng.randint(0, n, size=(100, 5)):
        assert_close(integrate(Phi(i, x), Phi(j, x), Phi(k, x), dPhi(l, x),
                               Phi(m, x)),
                     phi[i, j, k, l, m], atol=1.0e-2)
        j = randint_neibor(i)
        k = randint_neibor(i)
        l = randint_neibor(i)
        m = randint_neibor(i)
        assert_close(integrate(Phi(i, x), Phi(j, x), Phi(k, x), dPhi(l, x),
                               Phi(m, x)),
                     phi[i, j, k, l, m], atol=1.0e-2)


def test_phi_ij_dk_dl_m():
    phi = basis1d.phi_ij_dk_dl_m(x).todense()

    for (i, j, k, l, m) in rng.randint(0, n, size=(100, 5)):
        assert_close(integrate(Phi(i, x), Phi(j, x), dPhi(k, x), dPhi(l, x),
                               Phi(m, x)),
                     phi[i, j, k, l, m], atol=1.0e-2)
        j = randint_neibor(i)
        k = randint_neibor(i)
        l = randint_neibor(i)
        m = randint_neibor(i)
        print(i, j, k, l, m)
        assert_close(integrate(Phi(i, x), Phi(j, x), dPhi(k, x), dPhi(l, x),
                               Phi(m, x)),
                     phi[i, j, k, l, m], atol=1.0e-2)
