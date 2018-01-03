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

    

'''def test_jac():
    ninit = np.cumsum(np.exp(rng.randn(size - 1) * 0.2))
    ninit /= ninit[-1]
    tinit = np.cumsum(np.exp(rng.randn(size - 1) * 0.2)) * 30
    tinit /= tion[-1]
    ntinit = ninit * tinit
    nt2init = ntinit * tinit

    model = model1d.Cylindrical(r, m)
    model.initialize(rion, rcx, tion, tinit[-1] * tion[-1], n_edge=1.0)

    # jac_n_particle
    jac = model._jac_n_particle(model.get_n(ninit, False),
                                model.get_nt(ntinit, False),
                                model.get_nt2(nt2init, False)).todense()
    fun = model._fun_particle(model.get_n(ninit, False),
                              model.get_nt(ntinit, False),
                              model.get_nt2(nt2init, False))
    for i in range(size-1):
        dn = np.zeros_like(ninit)
        dn[i] = 1.0e-5
        fun_dn = model._fun_particle(model.get_n(ninit + dn, False),
                                     model.get_nt(ntinit, False),
                                     model.get_nt2(nt2init, False))
        assert np.allclose(jac[:, i], (fun_dn - fun) / 1.0e-5)

    # jac_t_particle
    jac = model._jac_nt_particle(model.get_n(ninit, False),
                                 model.get_nt(ntinit, False),
                                 model.get_nt2(nt2init, False)).todense()
    for i in range(size-1):
        dt = np.zeros_like(ntinit)
        dt[i] = 1.0e-5
        fun_dt = model._fun_particle(model.get_n(ninit, False),
                                     model.get_nt(ntinit + dt, False),
                                     model.get_nt(nt2init, False))
        assert np.allclose(jac[:, i], (fun_dt - fun) / 1.0e-5)

    # jac_n_energy
    jac = model._jac_n_energy(model.get_n(ninit, False),
                              model.get_t(tinit, False)).todense()
    fun = model._fun_energy(model.get_n(ninit, False),
                            model.get_t(tinit, False))
    for i in range(size-1):
        dn = np.zeros_like(ninit)
        dn[i] = 1.0e-5
        fun_dn = model._fun_energy(model.get_n(ninit + dn, False),
                                   model.get_t(tinit, False))
        assert np.allclose(jac[:, i], (fun_dn - fun) / 1.0e-5)

    # jac_t_particle
    jac = model._jac_t_energy(model.get_n(ninit, False),
                              model.get_t(tinit, False)).todense()
    for i in range(size-1):
        dt = np.zeros_like(tinit)
        dt[i] = 1.0e-5
        fun_dt = model._fun_energy(model.get_n(ninit, False),
                                   model.get_t(tinit + dt, False))
        assert np.allclose(jac[:, i], (fun_dt - fun) / 1.0e-5)


def _test_jac_mix():
    n1init = np.cumsum(np.exp(rng.randn(size - 1) * 0.2))
    n1init /= n1init[-1] * (1.0 + 0.1 * rng.randn(1))
    t1init = np.exp(rng.randn(size - 1) * 0.2)
    t1init /= tion[-1] * (1.0 + 0.1 * rng.randn(1))
    n2init = np.cumsum(np.exp(rng.randn(size - 1) * 0.2))
    n2init /= n2init[-1] * (1.0 + 0.1 * rng.randn(1))
    t2init = np.exp(rng.randn(size - 1) * 0.2)
    t2init /= tion[-1] * (1.0 + 0.1 * rng.randn(1))

    n2init *= 2.0
    t2init *= 2.0

    ninit = np.concatenate([n1init, n2init])
    tinit = np.concatenate([t1init, t2init])

    model = model1d.Cylindrical_mixed(r, m, m2)
    model.initialize(rion, rcx, r2cx, tion,
                     t1init[-1] * tion[-1], t2init[-1] * tion[-1],
                     n1_edge=n1init[-1], n2_edge=n2init[-1])

    delta = 1.0e-6
    # jac_n_particle
    jac = model._jac_n_particle(model.get_n(ninit, False),
                                model.get_t(tinit, False)).todense()
    fun = model._fun_particle(model.get_n(ninit, False),
                              model.get_t(tinit, False))
    for i in range(len(ninit)):
        dn = np.zeros_like(ninit)
        dn[i] = delta
        fun_dn = model._fun_particle(model.get_n(ninit + dn, False),
                                     model.get_t(tinit, False))
        assert np.allclose(jac[:, i], (fun_dn - fun) / delta)

    # jac_t_particle
    jac = model._jac_t_particle(model.get_n(ninit, False),
                                model.get_t(tinit, False)).todense()
    for i in range(len(ninit)):
        dt = np.zeros_like(tinit)
        dt[i] = delta
        fun_dt = model._fun_particle(model.get_n(ninit, False),
                                     model.get_t(tinit + dt, False))
        assert np.allclose(jac[:, i], (fun_dt - fun) / delta)

    # jac_n_energy
    jac = model._jac_n_energy(model.get_n(ninit, False),
                              model.get_t(tinit, False)).todense()
    fun = model._fun_energy(model.get_n(ninit, False),
                            model.get_t(tinit, False))
    for i in range(len(ninit)):
        dn = np.zeros_like(ninit)
        dn[i] = delta
        fun_dn = model._fun_energy(model.get_n(ninit + dn, False),
                                   model.get_t(tinit, False))
        assert np.allclose(jac[:, i], (fun_dn - fun) / delta)

    # jac_t_energy
    jac = model._jac_t_energy(model.get_n(ninit, False),
                              model.get_t(tinit, False)).todense()
    for i in range(len(ninit)):
        dt = np.zeros_like(tinit)
        dt[i] = delta
        fun_dt = model._fun_energy(model.get_n(ninit, False),
                                   model.get_t(tinit + dt, False))
        assert np.allclose(jac[:, i], (fun_dt - fun) / delta, rtol=1.0e-3)


@pytest.mark.parametrize('use_jac', [False, True])
@pytest.mark.parametrize('always_positive', [False, True])
@pytest.mark.parametrize('method', ['leastsq'])
def test_cylindrical(use_jac, always_positive, method):
    tatom = 0.8 * tion - 2.0
    model.initialize(rion, rcx, tion, tatom[-1], n_edge=1.0)
    n, t, res = model.solve_nt(
        n_init=1.0 / tatom * tatom[-1], t_init=tatom, use_jac=use_jac,
        always_positive=always_positive, alpha=1.0e6, method=method,
        factor=1.0)
    #   assert res['success']
    print(n)
    print(t)
    print(n * t)
    print(res)
    # make sure this solution satisfies the differential equation
    rmu = r / (m * (rion + rcx))
    dndr = np.gradient(EV * n * t, r)
    lhs = np.gradient(rmu * dndr, r)
    rhs = r * rion * n

    if not always_positive:  # TODO When always_positive, this fails...
        assert np.allclose(lhs[:-2], rhs[:-2], rtol=0.3)


@pytest.mark.parametrize('use_jac', [False, True])
@pytest.mark.parametrize('always_positive', [False, True])
@pytest.mark.parametrize('method', [None])
def test_cylindrical_tfix(use_jac, always_positive, method):
    model = model1d.Cylindrical(r, m)
    tatom = tion * 0.5 - 2.0
    model.initialize(rion, rcx, tion, tatom[-1], n_edge=1.0)
    n, t, res = model.solve_n(n_init=1.0 / tion, t_init=tatom,
                              use_jac=use_jac, always_positive=always_positive)
    assert res['success']
    assert np.allclose(t, tatom)
    # make sure this solution satisfies the differential equation
    rmu = r / (m * (rion + rcx))
    dndr = np.gradient(EV * n * t, r)
    lhs = np.gradient(rmu * dndr, r)
    rhs = r * rion * n

    if not always_positive:  # TODO When always_positive, this fails...
        assert np.allclose(lhs[:-2], rhs[:-2], rtol=0.3)


@pytest.mark.parametrize('use_jac', [False, True])
@pytest.mark.parametrize('always_positive', [False, True])
@pytest.mark.parametrize('method', [None])
def test_cylindrical_mixed_tfix(use_jac, always_positive, method):
    model = model1d.Cylindrical_mixed(r, m, m2)
    tatom = tion * 0.5 - 2.0
    model.initialize(rion, rcx, r2cx, tion, tatom[-1], tatom[-1],
                     n1_edge=1.0, n2_edge=2.0)
    n1, n2, t1, t2, res = model.solve_n(
        n1_init=1.0 / tion, n2_init=1.0 / tion,
        t1_init=tatom, t2_init=tatom,
        use_jac=use_jac, always_positive=always_positive)
    assert res['success']
    assert np.allclose(t1, tatom)
    assert np.allclose(t2, tatom)
    # make sure this solution satisfies the differential equation
    print(n1)
    print(n2)
    rmu = r / (m * (rion + rcx + r2cx))
    dndr = np.gradient(EV * n1 * t1, r)
    lhs = np.gradient(rmu * dndr, r)
    rhs = r * (rion * n1 + r2cx * n1 - rcx * n2)

    print(lhs)
    print(rhs)
    #if not always_positive:  # TODO When always_positive, this fails...
    #    assert np.allclose(lhs[:-2], rhs[:-2], rtol=0.3)

    rmu = r / (m2 * (rion + rcx + r2cx))
    dndr = np.gradient(EV * n2 * t2, r)
    lhs = np.gradient(rmu * dndr, r)
    rhs = r * (rion * n2 + rcx * n2 - r2cx * n1)
    print(lhs)
    print(rhs)

    #if not always_positive:  # TODO When always_positive, this fails...
    #    assert np.allclose(lhs[:-2], rhs[:-2], rtol=0.3)
    raise ValueError


@pytest.mark.parametrize('use_jac', [False, True])
@pytest.mark.parametrize('always_positive', [False, True])
@pytest.mark.parametrize('method', [None])
def test_cylindrical_nfix(use_jac, always_positive, method):
    model = model1d.Cylindrical(r, m)
    tatom = tion * 0.5 - 2.0
    model.initialize(rion, rcx, tion, tatom[-1], n_edge=1.0)
    n, t, res = model.solve_n(n_init=1.0 / tion, t_init=tatom,
                              use_jac=use_jac, always_positive=always_positive)
    n, t, res = model.solve_t(n_init=n, t_init=t, use_jac=use_jac,
                              always_positive=always_positive)
    assert res['success']
    # make sure this solution satisfies the differential equation
    kT = EV * t
    rmu = r / (m * (rion + rcx))
    dndr = np.gradient(2.5 * n * kT**2, r)
    lhs = np.gradient(rmu * dndr, r)
    rhs = 1.5 * r * (rion + rcx) * n * kT - 1.5 * r * rcx * EV * tion
    lhs /= EV
    rhs /= EV
    print(t)
    print(lhs)
    print(rhs)
    # TODO validation test
    # assert np.allclose(lhs, rhs, rtol=1.0e-1)
    # raise ValueError


@pytest.mark.parametrize('always_positive', [False])
@pytest.mark.parametrize('method', [None])
def test_cylindrical(always_positive, method):
    model = model1d.Cylindrical(r, m)
    n, t, res = model.solve(rion, rcx, tion, 3.0,
                            always_positive=always_positive)
    assert res['success']
    assert (n > 1.0e-10).all()
    assert (t > 1.0e-10).all()
    # make sure this solution satisfies the differential equation
    rmu = r / (m * (rion + rcx))
    dndr = np.gradient(EV * n * t, r)
    lhs = np.gradient(rmu * dndr, r)
    rhs = r * rion * n
    # assert np.allclose(lhs, rhs, rtol=1.0e-1)
'''
