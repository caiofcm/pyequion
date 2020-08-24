import numpy as np
import pyequion

sol = pyequion.solve_solution(
    {'NaHCO3': 50, 'CaCl2': 10},
#     allow_precipitation=True,
#     activity_model_type=pyequion.TypeActivityCalculation('BROMLEY'),
#     activity_model_type='PITZER',
#     activity_model_type='BROMLEY',
#     solid_equilibrium_phases=['Calcite']
)
pyequion.print_solution(sol)
print('pos first')

pyequion.jit_compile_functions()

sol = pyequion.solve_solution(
    {'NaHCO3': 50, 'CaCl2': 10},
#     allow_precipitation=True,
#     activity_model_type=pyequion.TypeActivityCalculation('BROMLEY'),
#     activity_model_type='PITZER',
#     activity_model_type='BROMLEY',
#     solid_equilibrium_phases=['Calcite']
)

pyequion.print_solution(sol)


print('Second')

sys_eq = pyequion.create_equilibrium(
        feed_compounds=['CaSO4', 'NaCl']
)

sol3 = pyequion.solve_solution({'CaSO4': 100, 'NaCl': 100}, sys_eq,
    activity_model_type='bromley')

pyequion.print_solution(sol3)

print('Trying with precipitation on')

sol4 = pyequion.solve_solution({'CaSO4': 100, 'NaCl': 100}, sys_eq,
    activity_model_type='bromley',
    allow_precipitation=True,
    solid_equilibrium_phases=['Gypsum'])

pyequion.print_solution(sol4)

print('End')

