#%% Imports
import pyequion

#%% NaHCO3 Equilibrium

sol = pyequion.solve_solution({'NaHCO3': 15.0})

pyequion.print_solution(sol, conc_and_activity=True)

print('NaHCO3 at 15mM has a calculated pH of {}'.format(sol.pH))

# %% NaHCO3 + CaCl2 Equilibrium

sol = pyequion.solve_solution({'NaHCO3': 15.0, 'CaCl2': 50.0})

pyequion.print_solution(sol, conc_and_activity=True)

print('NaHCO3=15mM and CaCl2=50mM has a calculated pH of {}'.format(sol.pH))

print('The mixture is supersaturated for calcite CaCO3 and undersaturated for Halite:')
print(sol.saturation_index)

# %% NaHCO3 + CaCl2 Equilibrium with Solid Phase

sol = pyequion.solve_solution({'NaHCO3': 15.0, 'CaCl2': 50.0},
    allow_precipitation=True,
    solid_equilibrium_phases=['Calcite'])
# OBS: if solid_equilibrium_phases the solid phase higher SI is used.

pyequion.print_solution(sol, conc_and_activity=True)

print('The mixture allowing precipitation has a calculated pH of {}'.format(sol.pH))

print('The amount of Calcite precipitated at equilibrium is (mol/L):')
print(sol.preciptation_conc['Calcite'])


# %% Barite formation

sol = pyequion.solve_solution({'BaSO4': 15.0},
    # allow_precipitation=True,
    # solid_equilibrium_phases=['Calcite']
)
# OBS: if solid_equilibrium_phases the solid phase higher SI is used.

pyequion.print_solution(sol, conc_and_activity=True)

# %%
