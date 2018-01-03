import pytest
import numpy as np
import sparse

from neutral_diffusion import model1d
from neutral_diffusion.utils import vec2coo
from neutral_diffusion.constants import EV


r = np.cumsum(np.linspace(0.1, 0.01, 30))
r /= r[-1]
size = len(r)
rion = ((1 - r**4) + 0.1) * 1.0e5  # [/s]
rcx = rion * 2  # [/s]
r2cx = rion * 4  # [/s]
tion = np.exp(- r * r * 4.0) * 1.0e3 + 3.0  # [eV]
tedge = 3.0
m = 1.6726219e-27  # [kg]
m2 = m * 2.0  # isotope
rng = np.random.RandomState(0)


def test_L():
    model = model1d.Cylindrical(r, m)
    model.initialize(rion, rcx, tion, n_edge=1.0, t_edge=3.0)
    L = model.L
    n = np.random.randn(size-1)
    t = np.random.randn(size-1)
    x = np.concatenate([n, n+t, n+t+t])
    assert np.allclose(L @ x, 0.0)


def test_solve():
    model = model1d.Cylindrical(r, m)
    model.initialize(rion, rcx, tion, n_edge=1.0, t_edge=3.0)

    n_init = 1.0 / tion * tion[-1]
    t_init = tion

    n, t, it = model.solve_core(n_init, t_init, rho=0.1)

    # make sure this solution satisfies the differential equation
    rmu = r / (m * (rion + rcx))
    dndr = np.gradient(EV * n * t, r)
    lhs = np.gradient(rmu * dndr, r)
    rhs = r * rion * n
    assert np.allclose(lhs[:-2], rhs[:-2], rtol=0.05, atol=1.0e2)


def test_solve_mix():
    model = model1d.Cylindrical_mixed(r, m, m2)
    model.initialize(rion, rcx, r2cx, tion, n1_edge=2.0, n2_edge=1.0,
                     t1_edge=3.0, t2_edge=3.0)

    n1_init = 1.0 / tion * tion[-1]
    t1_init = tion
    n2_init = 1.0 / tion * tion[-1]
    t2_init = tion

    n1, n2, t1, t2, it = model.solve_core(n1_init, n2_init, t1_init, t2_init,
                                          rho=0.1)
    # make sure this solution satisfies the differential equation
    rmu = r / (m * (rion + rcx + r2cx))
    dndr = np.gradient(EV * n1 * t1, r)
    lhs = np.gradient(rmu * dndr, r)
    rhs = r * (rion + rcx + r2cx) * n1 - r * rcx * (n1 + n2)
    assert np.allclose(lhs[:-2], rhs[:-2], rtol=0.1, atol=1.0e2)

    rmu = r / (m2 * (rion + rcx + r2cx))
    dndr = np.gradient(EV * n2 * t2, r)
    lhs = np.gradient(rmu * dndr, r)
    rhs = r * (rion + rcx + r2cx) * n2 - r * r2cx * (n1 + n2)
    print(lhs)
    print(rhs)
    print(lhs / rhs - 1)
    assert np.allclose(lhs[:-2], rhs[:-2], rtol=0.1, atol=3.0e2)
