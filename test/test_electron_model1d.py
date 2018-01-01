import pytest
import numpy as np

from neutral_diffusion import electron_model1d


r = np.linspace(0, 1.0, 60)
n = len(r)
diffusion_coef = np.ones(n) * 1.0  # [m^2/s]
source = 200 * np.exp(- r * r)  # [/s]
m = 1.6726219e-27  # [kg]


@pytest.mark.parametrize('use_jac', [True, False])
@pytest.mark.parametrize('always_positive', [False, True])
@pytest.mark.parametrize('method', [None])
def test_cylindrical(use_jac, always_positive, method):
    model = electron_model1d.Cylindrical(r, m)
    n, res = model.solve(diffusion_coef, source, use_jac=use_jac,
                             always_positive=always_positive)
    assert res['success']

    print(res)

    # make sure this solution satisfies the differential equation
    dndr = np.gradient(n, r)
    lhs = np.gradient(r * diffusion_coef * dndr, r)
    rhs = r * source * n
    assert np.allclose(lhs[:-2], rhs[:-2], atol=1.0e-2, rtol=0.1)
