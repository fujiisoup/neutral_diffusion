# Neutral diffusion model for high temperature plasmas

This small package provides a diffusion model of neutral atoms in high temperature plasmas, where the dominant atomic processes are ionization and charge exchange collisions.

## Requirement and installation

The following packages are required,
+ scipy
+ sparse

In order to install this package, execute
```
git clone https://github.com/fujii-team/neutral_diffusion
cd neutral_diffusion
python setup.py install
```

## Usage

Currently, we only provides the model for 1-dimensional cylindrical plasmas.

You need to give radial distributions of *ionization rate*, *charge exchange rate*, *ion temperature*, and the boundary condition, the atom temperature and density at the outermost surface of plasma,

```python
# r: 1d-np.ndarray of radial coordinate
# m: float of atom mass
model = neutral_diffusion.model1d.Cylindrical(r, m)
# rate_ion: ionization rate
# rate_cx: charge exchange rate
# t_ion: ion temperature
# t_edge: edge atom temperature
n, t, it = model.solve(rate_ion, rate_cx, t_ion, t_edge)
```

More details can be found in the notebooks [here](examples).
