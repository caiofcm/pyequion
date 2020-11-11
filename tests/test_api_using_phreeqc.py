"""
test_api_using_phreeqc.py

Test file to compared pyequion solution with phreeqc using phreeqc.dat database,
which is a B-dot with Davies equation fallback.

The phreeqc solution was beforehand generated and saved to a file to avoid the dependency on this package.
"""

import os

# os.environ['NUMBA_DISABLE_JIT'] = '1' #SHOULD WORK WITH 0 ALSO
import time
import sys
import inspect
import numpy as np
import pytest
import json

# try:
#     from phreeqpython import PhreeqPython
#     pp = PhreeqPython()
# except OSError as e:
#     print('Failed loading Phreeqpython')
#     # raise(e)
#     pp = None
# from pyequion.reactions_species_builder import *
from utils_tests import (
    assert_solution_result,
    compare_with_expected_perc_tuple,
)
import pyequion
from pyequion import solve_solution, ClosingEquationType
from SAVE_SOLUTIONREF import SAVE_SOLUTIONREF as REF_SOLUTIONS


# THIS LINE WILL COMPILE FUNCTION with numba
# pyequion.core.jit_compile_functions()


def test_phreeqpython_nahco3_open():
    comp_dict = {"NaHCO3": 1.0}
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
    )
    fname = inspect.stack()[0][3]
    solution_ref = REF_SOLUTIONS[fname][0]
    assert_solution(solution, solution_ref)


def test_phreeqpython_nahco3_cacl2_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"NaHCO3": 2.0, "CaCl2": 1.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        initial_feed_mass_balance=["Cl-"],
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["NaHCO3"],
    )
    assert_solution(solution, solution_ref1)

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})

    solution_ref3 = REF_SOLUTIONS[fname][2]
    # si_max = max([(v, k) for k, v in solution_ref3["phases"].items()])[1]
    solution = solve_solution(
        comp_dict,
        initial_feed_mass_balance=["Cl-"],
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["NaHCO3"],
        allow_precipitation=True,
    )
    assert_solution(solution, solution_ref3, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref3, tol={"si": 1e-1})


def test_phreeqpython_nahco3_cacl2_open_15_5():
    fname = inspect.stack()[0][3]
    comp_dict = {"NaHCO3": 15.0, "CaCl2": 5.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]

    solution = solve_solution(comp_dict, initial_feed_mass_balance=["Cl-"])
    print("")
    assert_solid(solution, solution_ref1, {"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})

    solution_ref3 = REF_SOLUTIONS[fname][2]
    # si_max = max([(v, k) for k, v in solution_ref3["phases"].items()])[1]
    solution = solve_solution(
        comp_dict,
        initial_feed_mass_balance=["Cl-"],
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["NaHCO3"],
        allow_precipitation=True,
    )
    assert_solution(solution, solution_ref3, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref3, tol={"si": 1e-1})


def test_phreeqpython_nahco3_cacl2_closed_15_5():
    fname = inspect.stack()[0][3]
    comp_dict = {"NaHCO3": 15.0, "CaCl2": 5.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        initial_feed_mass_balance=["Cl-"],
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=15.0,
    )
    print("")
    assert_solid(solution, solution_ref1)

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})

    solution_ref3 = REF_SOLUTIONS[fname][2]
    # si_max = max([(v, k) for k, v in solution_ref3["phases"].items()])[1]
    solution = solve_solution(
        comp_dict,
        initial_feed_mass_balance=["Cl-"],
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["NaHCO3"],
        allow_precipitation=True,
    )
    assert_solution(solution, solution_ref3, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref3, tol={"si": 1e-1})


def test_phreeqpython_baco3_cacl2_closed_5_2():
    fname = inspect.stack()[0][3]
    comp_dict = {"BaCO3": 5.0, "CaCl2": 2.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        initial_feed_mass_balance=["Cl-"],
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=5.0,
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1)
    pyequion.print_solution(solution)

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def xtest_phreeqpython_baco3_open():
    fname = inspect.stack()[0][3]  # FIXME -> not sure why is breaking
    comp_dict = {"BaCO3": 1.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(comp_dict)
    assert_solution(solution, solution_ref)
    assert_solid(solution, solution_ref)


def test_phreeqpython_baco3_closed():
    fname = inspect.stack()[0][3]
    comp_dict = {"BaCO3": 1.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=1.0,
    )
    assert_solution(solution, solution_ref)
    assert_solid(solution, solution_ref, tol={"si": 1e-1})


def test_phreeqpython_baco3_closed_15():
    fname = inspect.stack()[0][3]
    comp_dict = {"BaCO3": 15.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["BaCO3"],
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref, tol={"si": 1e-2})


def test_phreeqpython_baco3_closed_100():
    fname = inspect.stack()[0][3]
    comp_dict = {"BaCO3": 100.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["BaCO3"],
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2, "I": 1e-1})
    assert_solid(solution, solution_ref, tol={"si": 1e-1})


def test_phreeqpython_caco3_closed_1():
    fname = inspect.stack()[0][3]
    comp_dict = {"CaCO3": 1.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(comp_dict, close_type=ClosingEquationType.NONE)
    assert_solution(solution, solution_ref, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref, tol={"si": 1e-1})


def test_phreeqpython_caco3_closed_100():
    fname = inspect.stack()[0][3]
    comp_dict = {"CaCO3": 100.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["CaCO3"],
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref, tol={"si": 1e-1})


def test_phreeqpython_caco3_closed_Thot():
    fname = inspect.stack()[0][3]
    TC = 40.0
    comp_dict = {"CaCO3": 5.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["CaCO3"],
        TC=TC,
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref, tol={"si": 1e-1})


@pytest.mark.xfail(
    reason="I did not implement delta_h for species without a logK(T) expression"
)
def test_phreeqpython_caco3_closed_Thotter():
    fname = inspect.stack()[0][3]
    """Higher Temperature starts to introduce error...
    Need to implement the delta_h van Hoff for the species that does not analytic log_K(T)
    """
    TC = 50.0
    comp_dict = {"CaCO3": 5.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["CaCO3"],
        TC=TC,
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref, tol={"si": 1e-1})


def test_phreeqpython_cacl2_baco3_nahco3_closed():
    fname = inspect.stack()[0][3]
    comp_dict = {"CaCl2": 10.0, "BaCO3": 10.0, "NaHCO3": 10.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    c_tot = comp_dict["BaCO3"] + comp_dict["NaHCO3"]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=c_tot,
        initial_feed_mass_balance=["Cl-"],
    )
    print("")
    assert_solid(solution, solution_ref)


def test_phreeqpython_cacl2_baco3_nahco3_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"CaCl2": 1.0, "BaCO3": 1.0, "NaHCO3": 1.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref, tol={"si": 1e-1})


# def test_phreeqpython_baso4_closed():
#     fname = inspect.stack()[0][3]
#     comp_dict = {'BaSO4': 1.0}
#     solution_ref = REF_SOLUTIONS[fname][0]
#     solution = solve_solution(comp_dict,
#         close_type=ClosingEquationType.CARBON_TOTAL, carbon_total=comp_dict['BaSO4'])
#     assert_solution(solution, solution_ref, tol={'pH': 1e-2})
#     assert_solid(solution, solution_ref, tol={'si': 1e-1})


def test_phreeqpython_cacl2_closed_and_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"CaCl2": 15.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_baso4_closed():
    fname = inspect.stack()[0][3]
    """
    Low accuracy!
    """
    comp_dict = {"BaSO4": 15.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Ba", "S"],
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref, tol={"si": 1e-2})


def test_phreeqpython_baso4_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"BaSO4": 15.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Ba", "S"],
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref, tol={"si": 1e-1})


def test_phreeqpython_baco3_baso4_closed():
    fname = inspect.stack()[0][3]
    comp_dict = {"BaCO3": 1.0, "BaSO4": 1.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["BaCO3"],
        element_mass_balance=["Ba", "S"],
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref, tol={"si": 1e-2})


def _test_phreeqpython_cacl2_caso4_closed():
    "Error in Gypsum"
    fname = inspect.stack()[0][3]
    "Small error in Gypsum"
    comp_dict = {"BaCO3": 100.0, "CaSO4": 50.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["BaCO3"],
        element_mass_balance=["Ba", "Ca", "S"],
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2, "I": 1e-1})
    print("")
    assert_solid(
        solution, solution_ref, tol={"si": 1e-1}
    )  # small error in gypsum...


def test_phreeqpython_cacl2_caco3_closed():
    fname = inspect.stack()[0][3]
    comp_dict = {"CaCO3": 80.0, "CaCl2": 33.25}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        initial_feed_mass_balance=["Cl-"],
        element_mass_balance=["Ca"],
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2, "I": 1e-1})
    print("")
    assert_solid(solution, solution_ref)


def test_phreeqpython_caso4_closed():
    fname = inspect.stack()[0][3]
    comp_dict = {"CaSO4": 1.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Ca", "S"],
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2})


def test_phreeqpython_caso4_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"CaSO4": 1.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Ca", "S"],
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2})


def test_phreeqpython_mnco3_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"MnCO3": 150.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Mn"],
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref, tol={"si": 1e-1})


def test_phreeqpython_dolamite_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"MgCO3": 15.0, "CaCl2": 15.0}
    solution_ref = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,  # element_mass_balance=['Mg', ''],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref, tol={"si": 1e-1})


def test_phreeqpython_mnco3_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"MnCO3": 1.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["MnCO3"],
        element_mass_balance=["Mn"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Mn"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_feco3_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"FeCO3": 20.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["FeCO3"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(comp_dict, close_type=ClosingEquationType.OPEN)
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_srco3_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"SrCO3": 20.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["SrCO3"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(comp_dict, close_type=ClosingEquationType.OPEN)
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_caso4_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"CaSO4": 20.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Ca", "S"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Ca", "S"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_srso4_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"SrSO4": 100.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Sr", "S"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Sr", "S"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_hydroxiapatite_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"H3PO4": 10.0, "CaCO3": 10.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.CARBON_TOTAL,
        element_mass_balance=["Ca", "P"],
        carbon_total=comp_dict["CaCO3"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Ca", "P"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_caf2_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"CaCl2": 10.0, "HF": 10.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Ca", "F"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Ca", "F"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_sio2_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"H4SiO4": 10.0, "HF": 10.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Si", "F"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Si", "F"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_gibbsite_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"Al(OH)3": 10.0, "NaCl": 10.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Al", "Na"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Al", "Na"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_kaolinite_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"Al(OH)3": 10.0, "H4SiO4": 10.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Al", "Si"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Al", "Si"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_albite_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"NaCl": 10.0, "H4SiO4": 10.0, "Al(OH)3": 10.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Na", "Si", "Al"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Na", "Si", "Al"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_anorthite_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"CaCO3": 10.0, "H4SiO4": 10.0, "Al(OH)3": 10.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Ca", "Si", "Al", "C"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Ca", "Si", "Al"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_K_feldspar_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"KCl": 10.0, "H4SiO4": 10.0, "Al(OH)3": 10.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["K", "Si", "Al"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["K", "Si", "Al"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_Chlorite_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"MgCl2": 10.0, "H4SiO4": 10.0, "Al(OH)3": 10.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Mg", "Si", "Al"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Mg", "Si", "Al"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


@pytest.mark.xfail(
    reason="Redox not implemented and Arcanite is not in the generated phreeqc.dat file"
)
def test_phreeqpython_hematite_and_many_more_closed_open():
    fname = inspect.stack()[0][3]
    """
    FAILING HERE: IMPORTANT
    The redux is needed to provide Fe+++ from Fe++
    Thus, for a proper execution of Fe solids I will need to implement the redox
    scheme.

    FAILING HERE: Arcanite not in the phreeqc.dat database
    """
    comp_dict = {"FeCO3": 10.0, "K2SO4": 10.0, "Al(OH)3": 10.0}
    start = time.time()
    solution_ref1 = REF_SOLUTIONS[fname][0]
    print("Elapsed PHREEQC = {}".format(time.time() - start))
    start = time.time()
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Fe", "K", "Al", "S", "C"],
    )
    print("Elapsed = {}".format(time.time() - start))
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    start = time.time()
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Fe", "K", "Al", "S"],
    )
    print("Elapsed = {}".format(time.time() - start))
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


def test_phreeqpython_caso4_bacl2_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"CaSO4": 1.0, "BaCl2": 1.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Ca", "S", "Ba"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Ca", "S", "Ba"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})


def test_phreeqpython_barite_whiterite_calcite_closed_open():
    fname = inspect.stack()[0][3]
    comp_dict = {"BaCO3": 80.0, "CaCl2": 33.25, "NaHSO4": 20.0}
    solution_ref1 = REF_SOLUTIONS[fname][0]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.NONE,
        element_mass_balance=["Ba", "Ca", "Na", "S", "C"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref1, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref1, tol={"si": 1e-1})

    solution_ref2 = REF_SOLUTIONS[fname][1]
    solution = solve_solution(
        comp_dict,
        close_type=ClosingEquationType.OPEN,
        element_mass_balance=["Ba", "Ca", "Na", "S"],
        initial_feed_mass_balance=["Cl-"],
    )
    assert_solution(solution, solution_ref2, tol={"pH": 1e-2})
    assert_solid(solution, solution_ref2, tol={"si": 1e-1})


#####################################################
#####################################################
# PRECIPITATION WITH AQION COMPARISON
#####################################################
#####################################################
def test_precip_nahco3_cacl2():
    EXPECTED = {
        "pH": (6.21, 3.0),
        "I": (96.99e-3, 1.0),
        "sat-conc": {"Calcite": (2.69e1 * 1e-3, 3.0)},
    }
    comp_dict = {"NaHCO3": 80.0, "CaCl2": 33.25}
    # comp_dict = {'NaHCO3':2.0, 'CaCl2':1.0}
    solution = solve_solution(
        comp_dict,
        initial_feed_mass_balance=["Cl-"],
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["NaHCO3"],
        allow_precipitation=True,
    )

    assert_solution_result(solution, EXPECTED)
    for tag in EXPECTED["sat-conc"]:
        assert compare_with_expected_perc_tuple(
            solution.preciptation_conc[tag], EXPECTED["sat-conc"][tag]
        )


def test_precip_baco3_cacl2():
    EXPECTED = {
        "pH": (9.0, 3.0),
        "I": (99.792e-3, 1.0),
        "sat-conc": {
            "Calcite": (1.54e01 * 1e-3, 3.0),
            "Witherite": (6.46e01 * 1e-3, 3.0),
        },
    }
    comp_dict = {"BaCO3": 80.0, "CaCl2": 33.25}
    # comp_dict = {'NaHCO3':2.0, 'CaCl2':1.0}
    solution = solve_solution(
        comp_dict,
        initial_feed_mass_balance=["Cl-"],
        close_type=ClosingEquationType.CARBON_TOTAL,
        carbon_total=comp_dict["BaCO3"],
        allow_precipitation=True,
    )

    assert_solution_result(solution, EXPECTED)
    for tag in EXPECTED["sat-conc"]:
        assert compare_with_expected_perc_tuple(
            solution.preciptation_conc[tag], EXPECTED["sat-conc"][tag]
        )


def test_precip_baco3_cacl2_nahso4():
    EXPECTED = {
        "pH": (6.34, 3.0),
        "I": (100.412e-3, 1.0),
        "sat-conc": {
            "Calcite": (1.85e01 * 1e-3, 3.0),
            "Barite": (2.00e01 * 1e-3, 3.0),
            "Witherite": (4.74e01 * 1e-3, 3.0),
        },
    }
    comp_dict = {"BaCO3": 80.0, "CaCl2": 33.25, "NaHSO4": 20.0}
    # comp_dict = {'NaHCO3':2.0, 'CaCl2':1.0}
    solution = solve_solution(
        comp_dict,
        initial_feed_mass_balance=["Cl-"],
        element_mass_balance=["Ba", "Ca", "Na", "S", "C"],
        close_type=ClosingEquationType.NONE,
        allow_precipitation=True,
    )

    assert_solution_result(solution, EXPECTED)
    for tag in EXPECTED["sat-conc"]:
        assert compare_with_expected_perc_tuple(
            solution.preciptation_conc[tag], EXPECTED["sat-conc"][tag]
        )


def test_precip_Al2SO43_K3PO4_CaCl2():
    EXPECTED = {
        "pH": (2.29, 3.0),
        "I": (187.467e-3, 3.0),
        "sat-conc": {
            "Gypsum": (8.78 * 1e-3, 25.0),
        },
    }
    comp_dict = {"Al2(SO4)3": 30.0, "H3PO4": 15.0, "CaCl2": 30.0}
    # comp_dict = {'NaHCO3':2.0, 'CaCl2':1.0}
    solution = solve_solution(
        comp_dict,
        element_mass_balance=["Al", "P", "Ca", "S"],
        initial_feed_mass_balance=["Cl-"],
        close_type=ClosingEquationType.NONE,
        allow_precipitation=True,
    )

    assert_solution_result(solution, EXPECTED)
    for tag in EXPECTED["sat-conc"]:
        assert compare_with_expected_perc_tuple(
            solution.preciptation_conc[tag], EXPECTED["sat-conc"][tag]
        )


#################################
####### NEW ACTIVITY COEFS MODELS
#################################

# def test_bromley_model():
#     comp_dict = {'CaCl2': 300.0, 'NaHCO3': 200}
#     solution_ref = REF_SOLUTIONS[fname][0]
#     solution = solve_solution(comp_dict,
#         close_type=ClosingEquationType.NONE,
#         activity_model_type=pyequion.TypeActivityCalculation.BROMLEY

#     )

#     assert_solution(solution, solution_ref, tol={'pH': 1e-1, 'I': 1e-1})
#     return

# def test_sit():
#     comp_dict = {'CaCl2': 300.0, 'NaHCO3': 200}
#     # comp_dict = {'NaCl': 2000}
#     solution_ref = REF_SOLUTIONS[fname][0]
#     solution = solve_solution(comp_dict,
#         close_type=ClosingEquationType.NONE,
#         setup_log_gamma_func=pyequion.activity_coefficients.setup_SIT_model,
#         calc_log_gamma=pyequion.activity_coefficients.calc_sit_method,
#         activity_model_type=pyequion.TypeActivityCalculation.SIT

#     )

#     assert_solution(solution, solution_ref, tol={'pH': 1e-1, 'I': 1e-1})
#     return

# def test_eNRTL():
#     comp_dict = {'NaCl': 100.0}
#     # comp_dict = {'NaCl': 2000}
#     solution_ref = REF_SOLUTIONS[fname][0]
#     solution = solve_solution(comp_dict,
#         close_type=ClosingEquationType.NONE,
#         setup_log_gamma_func=pyequion.activity_coefficients.setup_eNRTL_model,
#         calc_log_gamma=pyequion.activity_coefficients.calc_eNRTL_method,
#         # activity_model_type=pyequion.TypeActivityCalculation.SIT

#     )

#     assert_solution(solution, solution_ref, tol={'pH': 1e-1, 'I': 1e-1})
#     return


#################################
####### TEST HELPERs and UTILITIES
#################################


def test_from_ions_is_equal_to_from_compounds():
    comps = {"Na+": 10, "HCO3-": 10, "Ca++": 5.0, "Cl-": 10.0}
    solution_ions = pyequion.solve_solution(comps)
    pyequion.print_solution(solution_ions)

    comps = {"NaHCO3": 10, "CaCl2": 5.0}
    solution_comps = pyequion.solve_solution(comps)
    pyequion.print_solution(solution_comps)

    assert np.isclose(solution_ions.pH, solution_comps.pH)


def test_creating_solution_result_from_x():

    feed_compounds = ["Na+", "HCO3-", "Ca++", "Cl-"]
    fixed_elements = ["Cl-"]

    sys_eq = pyequion.create_equilibrium(
        feed_compounds,
        fixed_elements=fixed_elements,
    )

    comps = {"Na+": 10, "HCO3-": 10, "Ca++": 5.0, "Cl-": 10.0}
    solution_exp = pyequion.solve_solution(comps, sys_eq)
    pyequion.print_solution(solution_exp)

    x = solution_exp.x

    sys_eq2 = pyequion.create_equilibrium(
        feed_compounds,
        fixed_elements=fixed_elements,
    )

    solution_new = pyequion.get_solution_from_x(sys_eq2, x, comps)

    assert np.isclose(solution_exp.pH, solution_new.pH)


##### Auxiliaries
def assert_solution(solution, solution_ref, tol=None):
    tol_new = {"pH": 1e-3, "I": 1e-2}
    if isinstance(tol, dict):
        for key in tol:
            tol_new[key] = tol[key]
    assert np.isclose(solution.pH, solution_ref["pH"], tol_new["pH"])
    assert np.isclose(solution.I, solution_ref["I"], tol_new["I"])


def assert_solid(solution, solution_ref, tol={"si": 1e-2}):
    solid_names = solution.solid_names
    for tag, si_calc in solution.saturation_index.items():
        if tag == "":
            continue
        si_ref = solution_ref["phases"][tag]
        print("{}\t{:.3f}\t{:.3f}".format(tag, si_calc, si_ref))
        assert np.isclose(si_calc, si_ref, tol["si"])

    for ref_name in solution_ref["phases"]:
        if ref_name in solid_names or "(g)" in ref_name or "_pH" in ref_name:
            continue
        print("Phase not created = ", ref_name)


def run_all_tests():
    testfunctions = [
        (name, obj)
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if (inspect.isfunction(obj) and name.startswith("test_"))
    ]
    for name, f in testfunctions:
        print("Test: ", name)
        f()

    # print(SAVE_SOLUTIONS_FOR_LATER)
    # return any(f() for f in testfunctions)


if __name__ == "__main__":
    # test_phreeqpython_dolamite_open()
    run_all_tests()
