"""
getting_started.py

Part of pyequion package

This is a set of simple calculations and functionalities using pyequion.

The `# %%` is used by VsCode to run a interactive embedded jupyter window while keeping the python script file.

Check samples/CaCO3Equilibrium.ipynb for more functionalities.

Obs:

Results here are model calculated value, as in B-dot formulation.
Results should be used with causion and no warranty is provided.
Users are wellcome to create issues and contribute with the software for enhanced validations.

"""

# %% Imports
import pyequion

# %% NaHCO3 Equilibrium

sol = pyequion.solve_solution({"NaHCO3": 15.0})

pyequion.print_solution(sol, conc_and_activity=True)

print("NaHCO3 at 15mM has a calculated pH of {}".format(sol.pH))

print("The reactions are (better visualized in jupyter)")
pyequion.ipython_display_reactions(sol)

# %% NaHCO3 + CaCl2 Equilibrium

sol = pyequion.solve_solution({"NaHCO3": 15.0, "CaCl2": 50.0})

pyequion.print_solution(sol, conc_and_activity=True)

print("NaHCO3=15mM and CaCl2=50mM has a calculated pH of {}".format(sol.pH))

print(
    "The mixture is supersaturated for calcite CaCO3 and undersaturated for Halite:"
)
print(sol.saturation_index)

# %% NaHCO3 + CaCl2 Equilibrium with Solid Phase

sol = pyequion.solve_solution(
    {"NaHCO3": 15.0, "CaCl2": 50.0},
    allow_precipitation=True,
    solid_equilibrium_phases=["Calcite"],
)
# OBS: if solid_equilibrium_phases the solid phase higher SI is used.

pyequion.print_solution(sol, conc_and_activity=True)

print(
    "The mixture allowing precipitation has a calculated pH of {}".format(
        sol.pH
    )
)

print("The amount of Calcite precipitated at equilibrium is (mol/L):")
print(sol.preciptation_conc["Calcite"])


# %% Barite formation

sol = pyequion.solve_solution({"BaSO4": 15.0})
# OBS: if solid_equilibrium_phases the solid phase higher SI is used.

pyequion.print_solution(sol, conc_and_activity=True)

print(
    "The supersaturation for barite is = {}".format(
        sol.saturation_index["Barite"]
    )
)

# %% Barite Formation on mixture

sol = pyequion.solve_solution({"Na2SO4": 15.0, "BaCl2": 5})

pyequion.print_solution(sol, conc_and_activity=True)

print(
    "The supersaturation for barite is = {}".format(
        sol.saturation_index["Barite"]
    )
)

print("Equilibrating with solid phase:")

sol = pyequion.solve_solution(
    {"Na2SO4": 15.0, "BaCl2": 5}, allow_precipitation=True
)
print(
    "Will precipitate = {} mM of Barite (BaSO4)".format(
        sol.preciptation_conc["Barite"] * 1e3
    )
)


# %% Increasing pressure in Calcite system

feed_caco3_mM = 200.0
sol = pyequion.solve_solution(
    {"CaCO3": feed_caco3_mM},
    close_type=pyequion.ClosingEquationType.OPEN,
    co2_partial_pressure=pyequion.pCO2_ref,  # default value
    allow_precipitation=True,
    solid_equilibrium_phases=["Calcite"],
)
caco3_solubility_atm_co2_mM = (
    feed_caco3_mM - sol.preciptation_conc["Calcite"] * 1e3
)
print(
    "Calculated CaCO3 solubility at CO2(g) atmospheric partial pressure = {:.2f} mM".format(
        caco3_solubility_atm_co2_mM
    )
)

# Increasing the CO2(g) partial pressure
sol = pyequion.solve_solution(
    {"CaCO3": feed_caco3_mM},
    close_type=pyequion.ClosingEquationType.OPEN,
    co2_partial_pressure=2,  # in atm
    allow_precipitation=True,
    solid_equilibrium_phases=["Calcite"],
)
caco3_solubility_atm_co2_mM = (
    feed_caco3_mM - sol.preciptation_conc["Calcite"] * 1e3
)
print(
    "Calculated CaCO3 solubility at CO2(g) in 2 atm partial pressure = {:.2f} mM".format(
        caco3_solubility_atm_co2_mM
    )
)

# Calculating with PR
sol = pyequion.solve_solution(
    {"CaCO3": feed_caco3_mM},
    close_type=pyequion.ClosingEquationType.OPEN,
    co2_partial_pressure=2,  # in atm
    allow_precipitation=True,
    solid_equilibrium_phases=["Calcite"],
    fugacity_calculation="pr",
)
caco3_solubility_atm_co2_mM = (
    feed_caco3_mM - sol.preciptation_conc["Calcite"] * 1e3
)
print(
    "Calculated CaCO3 solubility at CO2(g) in 2 atm partial pressure with PengRobinson EOS is = {:.2f} mM".format(
        caco3_solubility_atm_co2_mM
    )
)
