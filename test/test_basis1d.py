import numpy as np

from neutral_diffusion import basis1d


x = np.cumsum(np.arange(1, 11) + 10.0)
dx = np.diff(x)
dx_inv = 1.0 / dx


def assert_close(x, y, **kwargs):
    assert np.allclose(x, y)


def test_phi_ij():
    phi = basis1d.phi_ij(x).todense()
    i = 0
    assert_close(phi[i, i], dx[i] / 3.0)
    assert_close(phi[i, i+1], dx[i] / 6.0)

    for i in range(1, len(x) - 1):
        assert_close(phi[i, i], dx[i] / 3.0 + dx[i-1] / 3.0)
        assert_close(phi[i, i+1], dx[i] / 6.0)
        assert_close(phi[i, i-1], dx[i-1] / 6.0)

    i = len(x) - 1
    assert_close(phi[i, i], dx[i-1] / 3.0)
    assert_close(phi[i, i-1], dx[i-1] / 6.0)


def test_phi_ijk():
    phi = basis1d.phi_ijk(x).todense()

    i = 0
    assert_close(phi[i, i, i], dx[i] / 4.0)
    assert_close(phi[i, i+1, i], dx[i] / 12.0)
    assert_close(phi[i, i, i+1], dx[i] / 12.0)
    assert_close(phi[i, i+1, i+1], dx[i] / 12.0)

    for i in range(1, len(x) - 1):
        assert_close(phi[i, i, i], dx[i] / 4.0 + dx[i-1] / 4.0)
        assert_close(phi[i, i+1, i], dx[i] / 12.0)
        assert_close(phi[i, i, i+1], dx[i] / 12.0)
        assert_close(phi[i, i+1, i+1], dx[i] / 12.0)
        assert_close(phi[i, i-1, i], dx[i-1] / 12.0)
        assert_close(phi[i, i, i-1], dx[i-1] / 12.0)
        assert_close(phi[i, i-1, i-1], dx[i-1] / 12.0)

    i = len(x) - 1
    assert_close(phi[i, i, i], dx[i-1] / 4.0)
    assert_close(phi[i, i-1, i], dx[i-1] / 12.0)
    assert_close(phi[i, i, i-1], dx[i-1] / 12.0)
    assert_close(phi[i, i-1, i-1], dx[i-1] / 12.0)


def test_phi_ijkl():
    phi = basis1d.phi_ijkl(x).todense()

    i = 0
    assert_close(phi[i, i, i, i], dx[i] / 5.0)
    assert_close(phi[i, i, i, i+1], dx[i] / 20.0)
    assert_close(phi[i, i, i+1, i], dx[i] / 20.0)
    assert_close(phi[i, i, i+1, i+1], dx[i] / 20.0)
    assert_close(phi[i, i+1, i, i], dx[i] / 20.0)
    assert_close(phi[i, i+1, i, i+1], dx[i] / 20.0)
    assert_close(phi[i, i+1, i+1, i], dx[i] / 20.0)
    assert_close(phi[i, i+1, i+1, i+1], dx[i] / 20.0)

    for i in range(1, len(x) - 1):
        assert_close(phi[i, i, i, i], dx[i] / 5.0 + dx[i-1] / 5.0)
        assert_close(phi[i, i, i, i+1], dx[i] / 20.0)
        assert_close(phi[i, i, i+1, i], dx[i] / 20.0)
        assert_close(phi[i, i, i+1, i+1], dx[i] / 20.0)
        assert_close(phi[i, i+1, i, i], dx[i] / 20.0)
        assert_close(phi[i, i+1, i, i+1], dx[i] / 20.0)
        assert_close(phi[i, i+1, i+1, i], dx[i] / 20.0)
        assert_close(phi[i, i+1, i+1, i+1], dx[i] / 20.0)

        assert_close(phi[i, i, i, i-1], dx[i-1] / 20.0)
        assert_close(phi[i, i, i-1, i], dx[i-1] / 20.0)
        assert_close(phi[i, i, i-1, i-1], dx[i-1] / 20.0)
        assert_close(phi[i, i-1, i, i], dx[i-1] / 20.0)
        assert_close(phi[i, i-1, i, i-1], dx[i-1] / 20.0)
        assert_close(phi[i, i-1, i-1, i], dx[i-1] / 20.0)
        assert_close(phi[i, i-1, i-1, i-1], dx[i-1] / 20.0)

    i = len(x) - 1
    assert_close(phi[i, i, i, i], dx[i-1] / 5.0)
    assert_close(phi[i, i, i, i-1], dx[i-1] / 20.0)
    assert_close(phi[i, i, i-1, i], dx[i-1] / 20.0)
    assert_close(phi[i, i, i-1, i-1], dx[i-1] / 20.0)
    assert_close(phi[i, i-1, i, i], dx[i-1] / 20.0)
    assert_close(phi[i, i-1, i, i-1], dx[i-1] / 20.0)
    assert_close(phi[i, i-1, i-1, i], dx[i-1] / 20.0)
    assert_close(phi[i, i-1, i-1, i-1], dx[i-1] / 20.0)


def test_phi_i_dj_dk_l():
    phi = basis1d.phi_i_dj_dk_l(x).todense()

    i = 0
    assert_close(phi[i, i, i, i], dx_inv[i] / 3.0)
    assert_close(phi[i, i, i+1, i], -dx_inv[i] / 3.0)
    assert_close(phi[i, i+1, i, i], -dx_inv[i] / 3.0)
    assert_close(phi[i, i+1, i+1, i], dx_inv[i] / 3.0)

    assert_close(phi[i, i, i, i+1], dx_inv[i] / 6.0)
    assert_close(phi[i, i, i+1, i+1], -dx_inv[i] / 6.0)
    assert_close(phi[i, i+1, i, i+1], -dx_inv[i] / 6.0)
    assert_close(phi[i, i+1, i+1, i+1], dx_inv[i] / 6.0)

    for i in range(1, len(x) - 1):
        assert_close(phi[i, i, i, i], dx_inv[i] / 3.0 + dx_inv[i-1] / 3.0)
        assert_close(phi[i, i, i+1, i], -dx_inv[i] / 3.0)
        assert_close(phi[i, i+1, i, i], -dx_inv[i] / 3.0)
        assert_close(phi[i, i+1, i+1, i], dx_inv[i] / 3.0)

        assert_close(phi[i, i, i, i+1], dx_inv[i] / 6.0)
        assert_close(phi[i, i, i+1, i+1], -dx_inv[i] / 6.0)
        assert_close(phi[i, i+1, i, i+1], -dx_inv[i] / 6.0)
        assert_close(phi[i, i+1, i+1, i+1], dx_inv[i] / 6.0)

        assert_close(phi[i, i, i, i], dx_inv[i] / 3.0 + dx_inv[i-1] / 3.0)
        assert_close(phi[i, i, i-1, i], -dx_inv[i-1] / 3.0)
        assert_close(phi[i, i-1, i, i], -dx_inv[i-1] / 3.0)
        assert_close(phi[i, i-1, i-1, i], dx_inv[i-1] / 3.0)

        assert_close(phi[i, i, i, i-1], dx_inv[i-1] / 6.0)
        assert_close(phi[i, i, i-1, i-1], -dx_inv[i-1] / 6.0)
        assert_close(phi[i, i-1, i, i-1], -dx_inv[i-1] / 6.0)
        assert_close(phi[i, i-1, i-1, i-1], dx_inv[i-1] / 6.0)

    i = len(x) - 1
    assert_close(phi[i, i, i, i], dx_inv[i-1] / 3.0)
    assert_close(phi[i, i, i-1, i], -dx_inv[i-1] / 3.0)
    assert_close(phi[i, i-1, i, i], -dx_inv[i-1] / 3.0)
    assert_close(phi[i, i-1, i-1, i], dx_inv[i-1] / 3.0)

    assert_close(phi[i, i, i, i-1], dx_inv[i-1] / 6.0)
    assert_close(phi[i, i, i-1, i-1], -dx_inv[i-1] / 6.0)
    assert_close(phi[i, i-1, i, i-1], -dx_inv[i-1] / 6.0)
    assert_close(phi[i, i-1, i-1, i-1], dx_inv[i-1] / 6.0)


def test_phi_ijk_dl_m():
    phi = basis1d.phi_ijk_dl_m(x).todense()
    # TODO add tests


def test_phi_ij_dk_dl_m():
    phi = basis1d.phi_ij_dk_dl_m(x).todense()
    # TODO add tests
