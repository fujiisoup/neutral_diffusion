import numpy as np

from neutral_diffusion import model1d


r = np.cumsum(np.linspace(0, 1.0, 30))
n = len(r)
rion = np.ones(n) * 1.0e5  # [/s]
rcx = np.ones(n) * 2.0e5  # [/s]
tion = np.exp(- r * r) * 1.0e3 + 3.0  # [eV]
tedge = 3.0
m = 1.6726219e-27  # [kg]


def test_cylindrical():
    model = model1d.Cylindrical(r, m)
    model.solve(rion, rcx, tion, tedge)
