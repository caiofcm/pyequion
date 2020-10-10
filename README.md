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
