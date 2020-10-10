
<p align="center">
  <a href="https://github.com/caiofcm/pyequion">
    <img alt="pyequion" src="https://caiofcm.github.io/pyequion-onl/assets/pyequion_logo.png" width="100px">
  </a>
  <p align="center">Aqueous equilibrium calculation.</p>
</p>

[![Documentation Status](https://readthedocs.org/projects/pyequion/badge/?version=latest)](https://pyequion.readthedocs.io/en/latest/?badge=latest)
[![PyPi Version](https://img.shields.io/pypi/v/pyequion)](https://pypi.org/project/pyequion)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pyequion)](https://pypi.org/pypi/pyequion/)

# PyEquIon

A pure python implementation for electrolytes chemical equilibrium.

A simplified version is provided with a web based user interface: https://caiofcm.github.io/pyequion-onl/

## Features

- Pure python package: hence it is easy to install in any platform
- Calculation of equilibrium of inorganic salts in water solution with precipitation
- Automatic determination of reactions
- Provides information as: Ionic Strength, pH, Electric Conductivity and the concentrations of each species as well their activity coefficient
- A modular approach for the activity coefficient calculation allows the usage of new thermodynamic models
- Just in time compilation capabilities with `numba`
- Automatic differentiation with `sympy`
- Code generation for exporting the residual function for a giving system to other environments: suitable for kinetic simulations
- Automatic determination of the mean activity coefficient (often used in comparison with experiments)

## Installation

The package can be installed with `pip install pyequion` or `pip install <folder>`

## Basic Usage

```python
    import pyequion
    sol = pyequion.solve_solution({'NaHCO3': 50, 'CaCl2': 10})
    pyequion.print_solution(sol)
    >> Solution Results:
    >>    pH = 7.86640
    >>    sc = 6602.68061 uS/cm
    >>    I = 73.74077 mmol/L
    >>    DIC = 50.00000 mmol/L
    >> Saturation Index:
    >>    Halite: -4.77905928064043
    >>    Calcite: 2.083610139715626
    >>    Aragonite: 1.9398402923233906
    >>    Vaterite: 1.5171786455013265
```

## Documentation

https://pyequion.readthedocs.io/en/latest/

## Running Tests

To run unit tests:

```
pytest ./tests
```

To create the test report:

```
pytest --cov=pyequion ./tests/test_reactions_species_builder.py --cov-report=html
```


## Helpers

- Code can be JIT compiled with `numba` calling the function: `pyequion.jit_compile_functions()`
- When using JIT, running the code in `jupyter` becomes unstable, prefer regular python script.

## Contributors

- Caio Curitiba Marcellos
- Gerson Francisco da Silva Junior
- Elvis do Amaral Soares
- Fabio Ramos
- Amaro G. Barreto Jr

## References:

- Appelo, C. A. J., & Postma, D. (2004). Geochemistry, groundwater and pollution. CRC press.



[1]: https://packaging.python.org/guides/tool-recommendations/
