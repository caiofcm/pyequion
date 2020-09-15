.. pyequion documentation master file, created by
   sphinx-quickstart on Fri Jul 26 20:34:58 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyequion's documentation!
====================================

*PyEquIon*: A Python Package For Automatic Speciation Calculations of Aqueous Electrolyte Solutions

Usage example: ::

    import pyequion
    sol = pyequion.solve_solution({'NaHCO3': 50, 'CaCl2': 10})
    pyequion.print_solution(sol)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   license
   installation
   getting-started

Main Calculations
======================================
.. automodule:: pyequion.pyequion
   :members:

Reaction and Species Builder
======================================
.. automodule:: pyequion.reactions_species_builder
   :members:

Activity Coefficient
======================================
.. automodule:: pyequion.activity_coefficients
   :members:

Activity Coefficient
======================================
.. automodule:: pyequion.activity_coefficients
   :members:

General Utilities
======================================
.. automodule:: pyequion.utils
   :members:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
