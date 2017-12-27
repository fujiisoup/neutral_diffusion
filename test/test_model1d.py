import numpy as np

from neutral_diffusion import model1d


r = np.cumsum(np.arange(1, 11)[::-1] + 10.0)
n = len(r)
rion = np.ones(n) * 1.0e4  # [/s]
rcx = np.ones(n) * 2.0e4  # [/s]
tion = np.ones(n) * 1.0e3  # [eV]
tedge = 3.0
m = 1.6726219e-27  # [kg]


def test_cylindrical():
    model = model1d.Cylindrical(r, m)
    model.solve(rion, rcx, tion, tedge)
