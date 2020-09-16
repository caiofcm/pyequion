Samples
====================

This page shows equilibrium calculation examples. Feel free to contribute with new samples.

CaCO3Equilibrium.ipynb
----------------------------------

Calculations for CaCO3 systems. The example shows basic interactions with the package. It also includes:

* The effect of NaCl on the mean activity coefficient for CaCO3 is presented;
* The influence of NaCl on the calcite solubility.
* Comparison of open and closed systems for pH, DIC and amount precipitated.
* Variation of CO2 partial pressure

mean_coef_mgcl2_cacl2_evarodil.py
----------------------------------

This example shows the comparison of calculated mean activity coefficients with experimental data reported in the literature.
The package can automaticly provide the mean activity coefficient for a chosen salt. The analyze considers MgCl2 and CaCl2.

caco3_dyn_sim_daetools.py
----------------------------------

This example shows the dynamic simulation of the crystallization of calcite with the mixture of NaHCO3 and CaCl2.
The dynamic model is built in the daetools framework and the chemical equilibrium is solved from exported functions from daetools.

The function `aux_create_dynsim.py` generates the equilibrium system and export the results for a python file.
This exported system is coupled in the daetools model and solved in the differential algebraic equation solver (SUNDIALS).


