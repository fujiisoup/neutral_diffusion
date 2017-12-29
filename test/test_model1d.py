import pytest
import numpy as np

from neutral_diffusion import model1d
from neutral_diffusion.constants import EV


r = np.linspace(0, 1.0, 30) ** 2.0
size = len(r)
rion = ((1 - r**4) + 0.1) * 1.0e6  # [/s]
rcx = rion * 2  # [/s]
tion = np.exp(- r * r * 4.0) * 1.0e3 + 3.0  # [eV]
tedge = 3.0
m = 1.6726219e-27  # [kg]
rng = np.random.RandomState(0)


def test_jac():
    ninit = np.cumsum(np.exp(rng.randn(size - 1) * 0.2))
    ninit /= ninit[-1]
    tinit = np.cumsum(np.exp(rng.randn(size - 1) * 0.2)) * 30
    tinit /= tion[-1]

    model = model1d.Cylindrical(r, m)
    model.initialize(rion, rcx, tion, tinit[-1] * tion[-1])

    # jac_n_particle
    jac = model._jac_n_particle(model.get_n(ninit, False),
                                model.get_t(tinit, False)).todense()
    fun = model._fun_particle(model.get_n(ninit, False),
                              model.get_t(tinit, False))
    for i in range(size-1):
        dn = np.zeros_like(ninit)
        dn[i] = 1.0e-5
        fun_dn = model._fun_particle(model.get_n(ninit + dn, False),
                                     model.get_t(tinit, False))
        assert np.allclose(jac[:, i], (fun_dn - fun) / 1.0e-5)

    # jac_t_particle
    jac = model._jac_t_particle(model.get_n(ninit, False),
                                model.get_t(tinit, False)).todense()
    for i in range(size-1):
        dt = np.zeros_like(tinit)
        dt[i] = 1.0e-5
        fun_dt = model._fun_particle(model.get_n(ninit, False),
                                     model.get_t(tinit + dt, False))
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


@pytest.mark.parametrize('use_jac', [False, True])
@pytest.mark.parametrize('always_positive', [False, True])
@pytest.mark.parametrize('method', [None])
def test_cylindrical_tfix(use_jac, always_positive, method):
    model = model1d.Cylindrical(r, m)
    tatom = tion * 0.5 - 2.0
    n, t, success = model.solve(rion, rcx, tion, tatom[-1], t_init=tatom,
                                optimize_t=False, use_jac=use_jac,
                                always_positive=always_positive)
    assert success
    assert np.allclose(t, tatom)
    # make sure this solution satisfies the differential equation
    rmu = r / (m * (rion + rcx))
    dndr = np.gradient(EV * n * t, r)
    lhs = np.gradient(rmu * dndr, r)
    rhs = r * rion * n
    # TODO add validation test
    #assert np.allclose(lhs, rhs, rtol=1.0e-1)


@pytest.mark.parametrize('use_jac', [False, True])
@pytest.mark.parametrize('always_positive', [False, True])
@pytest.mark.parametrize('method', [None])
def test_cylindrical_nfix(use_jac, always_positive, method):
    model = model1d.Cylindrical(r, m)
    tatom = tion * 0.5 - 2.0
    n, t, success = model.solve(rion, rcx, tion, tatom[-1], t_init=tatom,
                                optimize_t=False, use_jac=use_jac,
                                always_positive=False)
    n, t, success = model.solve(rion, rcx, tion, 3.0, n_init=n, t_init=t,
                                optimize_n=False, use_jac=use_jac,
                                always_positive=always_positive)
    assert success
    # make sure this solution satisfies the differential equation
    rmu = r / (m * (rion + rcx))
    dndr = np.gradient(EV * n * t, r)
    lhs = np.gradient(rmu * dndr, r)
    rhs = r * rion * n
    #print(lhs)
    #print(rhs)
    # TODO validation test
    #assert np.allclose(lhs, rhs, rtol=1.0e-1)


@pytest.mark.parametrize('use_jac', [True, False])
@pytest.mark.parametrize('always_positive', [True])
@pytest.mark.parametrize('method', [None])
def test_cylindrical(use_jac, always_positive, method):
    model = model1d.Cylindrical(r, m)
    n, t, success = model.solve(rion, rcx, tion, 3.0,
                                optimize_t=True, use_jac=use_jac,
                                always_positive=always_positive)
    assert success
    print(n)
    print(t)
    print(tion)
    # make sure this solution satisfies the differential equation
    rmu = r / (m * (rion + rcx))
    dndr = np.gradient(EV * n * t, r)
    lhs = np.gradient(rmu * dndr, r)
    rhs = r * rion * n
    print(lhs)
    print(rhs)
    assert np.allclose(lhs, rhs, rtol=1.0e-1)



'''@pytest.mark.parametrize('use_jac', [False])
@pytest.mark.parametrize('always_positive', [False])
@pytest.mark.parametrize('method', [None])
def test_cylindrical(use_jac, always_positive, method):
    model = model1d.Cylindrical(r, m)
    n, t, success = model.solve(rion, rcx, tion, tedge, use_jac=use_jac,
                                always_positive=always_positive, max_nfev=100)
    print(n[:4])
    print(t[:4])
    raise ValueError
    #assert np.allclose(lhs, 0, atol=1.0e-1)
    assert success
'''
