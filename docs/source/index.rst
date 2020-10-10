.. pyequion documentation master file, created by
   sphinx-quickstart on Fri Jul 26 20:34:58 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyequion's documentation!
====================================

.. raw:: html

    <p align="center">
    <a href="https://github.com/caiofcm/pyequion">
        <img alt="pyequion" src="https://caiofcm.github.io/pyequion-onl/assets/pyequion_logo.png" width="100px">
    </a>
    <p align="center">Aqueous equilibrium calculation.</p>
    </p>

**PyEquIon**: A Python Package For Automatic Speciation Calculations of Aqueous Electrolyte Solutions

A simplified version is provided with a web based user interface: https://caiofcm.github.io/pyequion-onl/


Usage example: ::

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

NOTE: The chemical electrolyte calculations should be evaluated with attention,
as the thermodynamic of electrolyte solutions is not stablished, specially for high ionic strength solutions.
Additionally, the models are semi-empirical relying on parameter databases, which may not be correctly adjusted for a given use case.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   license
   installation
   getting-started
   samples
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
