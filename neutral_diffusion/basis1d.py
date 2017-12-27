import numpy as np
import itertools
import sparse


def phi_ij(x):
    """
    Get a 2d-tensor \int phi_i(r) phi_j(r) dr

    Parameters
    ----------
    x: 1d np.array

    Returns
    -------
    phi_ij: sparse.COO
    """
    size = len(x)
    dx = np.diff(x)

    ind = []
    val = []
    for i in range(size):
        if i < size - 1:
            ind.append((i, i))
            val.append(dx[i] / 3.0)

            ind.append((i, i+1))
            val.append(dx[i] / 6.0)

        if i > 0:
            ind.append((i, i))
            val.append(dx[i-1] / 3.0)

            ind.append((i, i-1))
            val.append(dx[i-1] / 6.0)

    return sparse.COO(np.array(ind).T, val, shape=(size, ) * 2)


def phi_ijk(x):
    """
    Get a 3d-tensor \int phi_i(r) phi_j(r) phi_k(r) dr

    Parameters
    ----------
    x: 1d np.array

    Returns
    -------
    phi_ijk: sparse.COO
    """
    size = len(x)
    dx = np.diff(x)

    ind = []
    val = []
    for i in range(size):
        if i < size - 1:
            ind.append((i, i, i))
            val.append(dx[i] * 2.0 / 12.0)

            for index in itertools.product([i], [i, i+1], [i, i+1]):
                ind.append(index)
                val.append(dx[i] / 12.0)

        if i > 0:
            ind.append((i, i, i))
            val.append(dx[i-1] * 2.0 / 12.0)

            for index in itertools.product([i], [i, i-1], [i, i-1]):
                ind.append(index)
                val.append(dx[i-1] / 12.0)

    return sparse.COO(np.array(ind).T, val, shape=(size, ) * 3)


def phi_ijkl(x):
    """
    Get a 4d-tensor \int phi_i(r) phi_j(r) phi_k(r) phi_l(r) dr

    Parameters
    ----------
    x: 1d np.array

    Returns
    -------
    phi_ijkl: sparse.COO
    """
    size = len(x)
    dx = np.diff(x)

    ind = []
    val = []
    for i in range(size):
        if i < size - 1:
            ind.append((i, i, i, i))
            val.append(dx[i] * 3.0 / 20.0)

            for index in itertools.product([i], [i, i+1], [i, i+1], [i, i+1]):
                ind.append(index)
                val.append(dx[i] / 20.0)

        if i > 0:
            ind.append((i, i, i, i))
            val.append(dx[i-1] * 3.0 / 20.0)

            for index in itertools.product([i], [i, i-1], [i, i-1], [i, i-1]):
                ind.append(index)
                val.append(dx[i-1] / 20.0)

    return sparse.COO(np.array(ind).T, val, shape=(size, ) * 4)


def phi_i_dj_dk_l(x):
    """
    Get a 4d-tensor \int phi_i(r) dphi_j_dr(r) dphi_k_dr(r) phi_l(r) dr

    Parameters
    ----------
    x: 1d np.array

    Returns
    -------
    phi_ijkl: sparse.COO
    """
    size = len(x)
    dx_inv = 1.0 / np.diff(x)

    ind = []
    val = []
    for i in range(size):
        if i < size - 1:
            for index, sgn in [((i, i, i), 1.0), ((i, i+1, i), -1.0),
                               ((i, i, i+1), -1.0), ((i, i+1, i+1), 1.0)]:
                ind.append(index + (i, ))
                val.append(sgn * dx_inv[i] / 3.0)
                ind.append(index + (i+1, ))
                val.append(sgn * dx_inv[i] / 6.0)

        if i > 0:
            for index, sgn in [((i, i, i), 1.0), ((i, i-1, i), -1.0),
                               ((i, i, i-1), -1.0), ((i, i-1, i-1), 1.0)]:
                ind.append(index + (i, ))
                val.append(sgn * dx_inv[i-1] / 3.0)
                ind.append(index + (i-1, ))
                val.append(sgn * dx_inv[i-1] / 6.0)

    return sparse.COO(np.array(ind).T, val, shape=(size, ) * 4)


def phi_ijk_dl_m(x):
    """
    Get a 5d-tensor \int phi_i(r) phi_j(r) phi_k(r) dphi_l_dr(r) phi_m dr

    Parameters
    ----------
    x: 1d np.array

    Returns
    -------
    phi_ijkl: sparse.COO
    """
    size = len(x)

    ind = []
    val = []
    for i in range(size):
        if i < size - 1:
            for l, sgn in [(i, 1.0), (i+1, -1.0)]:
                ind.append((i, i, i, l, i))
                val.append(sgn / 5.0)

                for i, j, k, m in [(i, i, i, i+1), (i, i, i+1, i),
                                   (i, i+1, i, i), (i, i+1, i+1, i+1)]:
                    ind.append((i, j, k, l, m))
                    val.append(sgn / 20.0)

                for i, j, k, m in [(i, i, i+1, i+1), (i, i+1, i+1, i),
                                   (i, i+1, i, i+1)]:
                    ind.append((i, j, k, l, m))
                    val.append(sgn / 30.0)

        if i > 0:
            for l, sgn in [(i, 1.0), (i-1, -1.0)]:
                ind.append((i, i, i, l, i))
                val.append(sgn / 5.0)

                for i, j, k, m in [(i, i, i, i-1), (i, i, i-1, i),
                                   (i, i-1, i, i), (i, i-1, i-1, i-1)]:
                    ind.append((i, j, k, l, m))
                    val.append(sgn / 20.0)

                for i, j, k, m in [(i, i, i-1, i-1), (i, i-1, i-1, i),
                                   (i, i-1, i, i-1)]:
                    ind.append((i, j, k, l, m))
                    val.append(sgn / 30.0)

    return sparse.COO(np.array(ind).T, val, shape=(size, ) * 5)


def phi_ij_dk_dl_m(x):
    """
    Get a 5d-tensor \int phi_i(r) phi_j(r) dphi_k_dr(r) dphi_l_dr(r) phi_m dr

    Parameters
    ----------
    x: 1d np.array

    Returns
    -------
    phi_ijkl: sparse.COO
    """
    size = len(x)
    dx_inv = 1.0 / np.diff(x)

    ind = []
    val = []
    for i in range(size):
        if i < size - 1:
            for k, l, sgn in [(i, i, 1.0), (i, i+1, -1.0),
                              (i+1, i, -1.0), (i+1, i+1, 1.0)]:
                ind.append((i, i, k, l, i))
                val.append(sgn * dx_inv[i] / 4.0)

                for i, j, m in itertools.product([i], [i, i+1], [i, i+1]):
                    ind.append((i, j, k, l, m))
                    val.append(sgn * dx_inv[i] / 12.0)

        if i > 0:
            for k, l, sgn in [(i, i, 1.0), (i, i-1, -1.0),
                              (i-1, i, -1.0), (i-1, i-1, 1.0)]:
                ind.append((i, i, k, l, i))
                val.append(sgn * dx_inv[i-1] / 4.0)

                for i, j, m in itertools.product([i], [i, i-1], [i, i-1]):
                    ind.append((i, j, k, l, m))
                    val.append(sgn * dx_inv[i-1] / 12.0)

    return sparse.COO(np.array(ind).T, val, shape=(size, ) * 5)
