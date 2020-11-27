
<p align="center">
  <a href="https://github.com/caiofcm/pyequion">
    <img alt="pyequion" src="https://caiofcm.github.io/pyequion-onl/assets/pyequion_logo.png" width="100px">
  </a>
  <p align="center">Aqueous equilibrium calculation.</p>
</p>

[![Documentation Status](https://readthedocs.org/projects/pyequion/badge/?version=latest)](https://pyequion.readthedocs.io/en/latest/?badge=latest)
[![PyPi Version](https://img.shields.io/pypi/v/pyequion)](https://pypi.org/project/pyequion)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pyequion)](https://pypi.org/pypi/pyequion/)
[![Coverage Status](https://coveralls.io/repos/github/caiofcm/pyequion/badge.svg?branch=master)](https://coveralls.io/github/caiofcm/pyequion?branch=master)
[![Build Status](https://travis-ci.org/caiofcm/pyequion.svg?branch=master)](https://travis-ci.org/caiofcm/pyequion)

# PyEquIon

A pure python implementation for electrolytes chemical equilibrium.

A simplified version is provided with a web based user interface: https://caiofcm.github.io/pyequion-onl/

Repository: https://github.com/caiofcm/pyequion

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

## Contributing

The code is formatted with black following flack8 specifications. Run `black .` to format the code.

## Helpers

- Code can be JIT compiled with `numba` calling the function: `pyequion.jit_compile_functions()`
- When using JIT, running the code in `jupyter` becomes unstable, prefer regular python script.

## Contributors

- Caio Curitiba Marcellos
- Gerson Francisco da Silva Junior
- Elvis do Amaral Soares
- Fabio Ramos
- Amaro G. Barreto Jr
- Danilo Naiff

## Folder Structure

```
.
├── api # Application Programming Interface for web service
├── data # Parameters database file (was replaced by python dictionaries in pyequion/data)
├── docs # Documentation generation (sphinx)
├── LICENSE.md #License file
├── pyequion
│   ├── activity_coefficients.py # Built-in thermodynamic models
│   ├── conductivity.py # Conductivity function
│   ├── core.py # Core functionalities
│   ├── data # Parameters database as python files
│   │   ├── __init__.py
│   │   ├── reactions_solids.py
│   │   ├── reactions_solutions.py
│   │   └── species.py
│   ├── __init__.py # Auxiliary
│   ├── jit_helper.py # Auxiliary
│   ├── PengRobinson.py # EOS for pure gas
│   ├── pitzer.py # Pitzer model
│   ├── properties_utils.py # Auxiliary
│   ├── pyequion.py # Application Programming Interface for python call
│   ├── reactions_species_builder.py # Creation of equilibrium system
│   ├── read_from_phreeqc_db.py # Auxiliary
│   ├── symbolic_computations.py # Symbolic Computation with sympy
│   ├── utils_api.py # Auxiliary
│   ├── utils_for_numba.py # Auxiliary
│   ├── utils.py # Auxiliary
│   └── wateractivity.py # Auxiliary for water activity in pitzer
├── pyproject.toml #Linting configuration
├── README.md # General guide
├── requirements-dev.txt # Development Requirements
├── requirements.txt # Main Requirements
├── samples #Some samples using pyequion
├── setup.py #Installation file
└── tests # Automatic Unit tests folder (pytest based)

```


[1]: https://packaging.python.org/guides/tool-recommendations/
