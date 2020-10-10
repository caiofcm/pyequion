import pyequion

# pyequion.jit_compile_functions()

sys_eq = pyequion.create_equilibrium(feed_compounds=["CaSO4", "NaCl"])

sol3 = pyequion.solve_solution(
    {"CaSO4": 100, "NaCl": 100}, sys_eq, activity_model_type="bromley"
)
pyequion.print_solution(sol3)
