"""Pyequion sample

- Simple usage
"""
import os
import time
import numpy as np

os.environ["NUMBA_DISABLE_JIT"] = "0"
import pyequion

# %%
# CREATING SYSTEM FOR No-JOT (Regular Mode)
esys = pyequion.create_equilibrium(["NaHCO3", "CaCl2"])

t01 = time.time()
sol = pyequion.solve_solution(
    {"NaHCO3": 50, "CaCl2": 10},
    esys,
)
print("Not JIT Elapsed = {}".format(time.time() - t01))

print("NO JIT LOOP")
Tspan = np.linspace(24.0, 28.0, 101)
t01 = time.time()
sol_span = [
    pyequion.solve_solution({"NaHCO3": 50, "CaCl2": 10}, esys) for T in Tspan
]
pH = np.array([s.pH for s in sol_span])
print("NO JIT Loop -> Elapsed = {}".format(time.time() - t01))

# %%
## COMPILING JIT
pyequion.jit_compile_functions()

# Creating the system for numba Just In Time Compilation
esys = pyequion.create_equilibrium(["NaHCO3", "CaCl2"])

# First Run -> Run time compilation, so it is slow
t01 = time.time()
sol = pyequion.solve_solution({"NaHCO3": 50, "CaCl2": 10}, esys)
print("1o JIT Elapsed = {}".format(time.time() - t01))

# Next run is already compiled, faster
print("Second")
t01 = time.time()
sol = pyequion.solve_solution({"NaHCO3": 50, "CaCl2": 10}, esys)
print("2o JIT Elapsed = {}".format(time.time() - t01))

# Running with LOOP
print("JIT with LOOP")
Tspan = np.linspace(24.0, 28.0, 101)
t01 = time.time()
sol_span = [
    pyequion.solve_solution({"NaHCO3": 50, "CaCl2": 10}, esys) for T in Tspan
]
pH = np.array([s.pH for s in sol_span])
print("2o JIT Elapsed = {}".format(time.time() - t01))


# Create a different system after compilation is on
sys_eq = pyequion.create_equilibrium(feed_compounds=["CaSO4", "NaCl"])

t01 = time.time()
sol3 = pyequion.solve_solution(
    {"CaSO4": 100, "NaCl": 100}, sys_eq, activity_model_type="bromley"
)
print("JITTED Changed System -> Elapsed = {}".format(time.time() - t01))

pyequion.print_solution(sol3)

print("End")
