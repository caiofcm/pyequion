# PyEquIon

A pure python implementation for electrolytes chemical equilibrium.

NIDF Scaling Group Project

## Installation

Package requirements are handled using pipenv, as recommendation from Python [[1]]. To install it do

```
git clone this-repositiory-path
cd pyequion
pipenv install
```

For development:

```
pipenv install --dev
```

Then, running tests:

```
pytest ./tests
pytest --cov=pyequion ./tests/test_reactions_species_builder.py --cov-report=html
```

## Features

- CaCl2 equilibrium: removed HCl equations - tested with Aqion and phreeqc
- NaHCO3 equilibrium - tested with Aqion and phreeqc
- NaHCO3 + CaCl2 equilibrium using DIC - tested with Aqion and phreeqc
- Automatic generation of jacobians using Sympy and the created auxiliary module to export as source code.

## Visualization in PyEquIon Online

- view at URL: https://caiofcm.github.io/pyequion-onl/
- Dev-code: https://gitlab.com/caiofcm/pyequion-viewer
- Deploy of backend API: https://gitlab.com/scaling-group/pyequion-distributed-gcp

## To do:

- Visualization
- Organize files and functions: remove duplications in `get_dict_from_solution_vector` functions
- Numbafy
- Continuous Delivery in gitlab

## References:

- Appelo, C. A. J., & Postma, D. (2004). Geochemistry, groundwater and pollution. CRC press.


## Development

- Run tests

Some test are using the phreeqcpython module, which requires the installation of phreeqc library.

## Installing phreeqcpython

- Install phreeqc software (steps not shown here)
- Go to: https://github.com/Vitens/VIPhreeqc and download viphreeqc.so ([linux](https://github.com/Vitens/VIPhreeqc))
- Past in `site-packages/phreeqcpython/lib`
- Get database folder from https://github.com/Vitens/VIPhreeqc and place in `site-packages/phreeqcpython/`
- pip install -U phreeqpython


## Datbase

Consider using the llnl.dat (or from Growndwater workbranch), many llnl_gamma. Also check others.

## For issue create:

- Size specie vs Size solver ? Force known species to be in the last places (tags also)


[1]: https://packaging.python.org/guides/tool-recommendations/
