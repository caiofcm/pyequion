import os

# os.environ['NUMBA_DISABLE_JIT'] = '1' #SHOULD WORK WITH 0 ALSO
import pytest
from pyequion import create_equilibrium, solve_equilibrium
import pyequion
from pyequion import reactions_species_builder as rbuilder
from pyequion import ClosingEquationType
from pyequion import symbolic_computations as mod_sym

# from xtest_pychemeq import assert_solution_result, compare_with_expected_perc_tuple
from utils_tests import (
    assert_solution_result,
    compare_with_expected_perc_tuple,
)
import numpy as np
import sympy

# from pyequion import caco3_specific_study

# --------------------------------------------
# 	TAGS BASED GENERATION
# --------------------------------------------
EXAMPLE_SPECIES = [
    "H2O",
    "NaHCO3",
    "CaCl2",
    "H+",
    "OH-",
    "Na+",
    "NaOH",
    "HCO3-",
    "CO2",
    "CO2(g)",
    "CO3--",
    "NaCO3-",
    "Na2CO3",
    "Ca++",
    "CaHCO3+",
    "CaCO3",
    "CaOH+",
    "Cl-",
]
EXAMPLE_REACTIONS = [
    {"H+": -1, "H2O": 1, "OH-": -1, "type": "rev"},
    {"HCO3-": -1, "Na+": -1, "NaHCO3": 1, "type": "rev"},
    {"H+": 1, "H2O": -1, "Na+": -1, "NaOH": 1, "type": "rev"},
    {"CO3--": -1, "H+": -1, "HCO3-": 1, "type": "rev"},
    {"CO2": -1, "H+": 1, "HCO3-": 1, "type": "rev"},
    {"CO2": 1, "CO2(g)": -1, "type": "henry"},
    {"CO3--": -1, "Na+": -1, "NaCO3-": 1, "type": "rev"},
    {"CO3--": -1, "Na+": -2, "Na2CO3": 1, "type": "rev"},
    {"Ca++": -1, "CaCl2": 1, "Cl-": -2, "type": "irrev"},
    {"Ca++": -1, "CaHCO3+": 1, "HCO3-": -1, "type": "rev"},
    {"CO3--": -1, "Ca++": -1, "CaCO3": 1, "type": "rev"},
    {"Ca++": -1, "CaOH+": 1, "H+": 1, "H2O": -1, "type": "rev"},
]


def test_nahco3_open():
    initial_comp = ["H2O", "NaHCO3", "CO2(g)"]
    E_SPECIES = [
        "H2O",
        "NaHCO3",
        "H+",
        "OH-",
        "Na+",
        "NaOH",
        "HCO3-",
        "CO2",
        "CO2(g)",
        "CO3--",
        "NaCO3-",
        "Na2CO3",
    ]
    E_REACTIONS = [
        {"H+": -1, "H2O": 1, "OH-": -1, "id_db": 10, "type": "rev"},
        {"HCO3-": -1, "Na+": -1, "NaHCO3": 1, "id_db": 2, "type": "rev"},
        {"H+": 1, "H2O": -1, "Na+": -1, "NaOH": 1, "id_db": 0, "type": "rev"},
        {"CO3--": -1, "H+": -1, "HCO3-": 1, "id_db": 9, "type": "rev"},
        {"CO2": -1, "H+": 1, "HCO3-": 1, "id_db": 8, "type": "rev"},
        {"CO2": 1, "CO2(g)": -1, "id_db": 7, "type": "henry"},
        {"CO3--": -1, "Na+": -1, "NaCO3-": 1, "id_db": 1, "type": "rev"},
        {"CO3--": -1, "Na+": -2, "Na2CO3": 1, "id_db": 3, "type": "rev"},
    ]
    species, reactions = rbuilder.get_species_reactions_from_compounds(
        initial_comp
    )
    for s in species:
        assert s in E_SPECIES
    assert len(E_REACTIONS) == len(reactions)
    assert len(E_SPECIES) == len(species)


def test_cacl2_open():
    initial_comp = ["H2O", "CaCl2", "CO2(g)"]
    E_SPECIES = [
        "H2O",
        "CaCl2",
        "CO2(g)",
        "H+",
        "OH-",
        "Ca++",
        "CaOH+",
        "Cl-",
        "CO2",
        "HCO3-",
        "CO3--",
        "CaCO3",
        "CaHCO3+",
    ]
    E_REACTIONS = [
        {"H+": -1, "H2O": 1, "OH-": -1, "id_db": 10, "type": "rev"},
        {"Ca++": -1, "CaCl2": 1, "Cl-": -2, "id_db": -1, "type": "irrev"},
        {
            "Ca++": -1,
            "CaOH+": 1,
            "H+": 1,
            "H2O": -1,
            "id_db": 4,
            "type": "rev",
        },
        {"CO2": 1, "CO2(g)": -1, "id_db": 7, "type": "henry"},
        {"CO2": -1, "H+": 1, "HCO3-": 1, "id_db": 8, "type": "rev"},
        {"CO3--": -1, "H+": -1, "HCO3-": 1, "id_db": 9, "type": "rev"},
        {"CO3--": -1, "Ca++": -1, "CaCO3": 1, "id_db": 12, "type": "rev"},
        {"Ca++": -1, "CaHCO3+": 1, "HCO3-": -1, "id_db": 11, "type": "rev"},
    ]
    species, reactions = rbuilder.get_species_reactions_from_compounds(
        initial_comp
    )
    for s in species:
        assert s in E_SPECIES
    assert len(E_REACTIONS) == len(reactions)
    assert len(E_SPECIES) == len(species)


def test_cacl2_nahco3():
    initial_comp = ["H2O", "NaHCO3", "CaCl2"]
    E_SPECIES = [
        "H2O",
        "NaHCO3",
        "CaCl2",
        "H+",
        "OH-",
        "Na+",
        "NaOH",
        "HCO3-",
        "CO2",
        "CO2(g)",
        "CO3--",
        "NaCO3-",
        "Na2CO3",
        "Ca++",
        "CaHCO3+",
        "CaCO3",
        "CaOH+",
        "Cl-",
    ]
    E_REACTIONS = [
        {"H+": -1, "H2O": 1, "OH-": -1, "id_db": 10, "type": "rev"},
        {"HCO3-": -1, "Na+": -1, "NaHCO3": 1, "id_db": 2, "type": "rev"},
        {"H+": 1, "H2O": -1, "Na+": -1, "NaOH": 1, "id_db": 0, "type": "rev"},
        {"CO3--": -1, "H+": -1, "HCO3-": 1, "id_db": 9, "type": "rev"},
        {"CO2": -1, "H+": 1, "HCO3-": 1, "id_db": 8, "type": "rev"},
        {"CO2": 1, "CO2(g)": -1, "id_db": 7, "type": "henry"},
        {"CO3--": -1, "Na+": -1, "NaCO3-": 1, "id_db": 1, "type": "rev"},
        {"CO3--": -1, "Na+": -2, "Na2CO3": 1, "id_db": 3, "type": "rev"},
        {"Ca++": -1, "CaCl2": 1, "Cl-": -2, "id_db": -1, "type": "irrev"},
        {"Ca++": -1, "CaHCO3+": 1, "HCO3-": -1, "id_db": 11, "type": "rev"},
        {"CO3--": -1, "Ca++": -1, "CaCO3": 1, "id_db": 12, "type": "rev"},
        {
            "Ca++": -1,
            "CaOH+": 1,
            "H+": 1,
            "H2O": -1,
            "id_db": 4,
            "type": "rev",
        },
    ]
    species, reactions = rbuilder.get_species_reactions_from_compounds(
        initial_comp
    )
    for s in species:
        assert s in E_SPECIES
    assert len(E_REACTIONS) == len(reactions)
    assert len(E_SPECIES) == len(species)


def test_identify_element_in_species_list():
    tags_list = {
        "H+",
        "OH-",
        "CO2",
        "CO3--",
        "HCO3-",
        "Na+",
        "NaOH",
        "NaCO3-",
        "NaHCO3",
        "Na2CO3",
        "CaOH+",
        "CaHCO3+",
        "CaCO3",
        "Ca++",
        "Cl-",
        "H2O",
    }
    tags_coefs = rbuilder.get_species_tags_with_an_element(tags_list, "Na")

    assert len(tags_coefs) == 5
    for el in ["Na+", "NaOH", "NaCO3-", "NaHCO3", "Na2CO3"]:
        assert el in tags_coefs
    assert tags_coefs["Na2CO3"] == 2
    assert tags_coefs["NaHCO3"] == 1


# --------------------------------------------
# 	ENGINE DEFINITIONS
# --------------------------------------------

# def test_create_list_of_species_engine_nahco3():
#     initial_comp = ['H2O', 'NaHCO3', 'CO2(g)']
#     known_tag = ['H2O', 'CO2(g)']
#     species, reactions = rbuilder.get_species_reactions_from_compounds(initial_comp)
#     species_conv = convert_species_tag_for_engine(species)
#     engine_species = create_list_of_species_engine(species_conv, species)
#     engine_idxs = create_Indexes_instance(species_conv, len(known_tag))
#     names = [sp.name for sp in engine_species]
#     print(names)
#     assert len(engine_species) > 0
#     assert engine_idxs.size == 10
#     # Deterministic: - Keep order
#     assert names == ['H2O', 'NaHCO3', 'CO2g', 'Hp', 'OHm', 'Nap', 'NaOH', 'HCO3m', 'CO2', 'CO3mm', 'NaCO3m', 'Na2CO3']

# def test_create_list_of_reactions_engine_nahco3():
#     initial_comp = ['H2O', 'NaHCO3', 'CO2(g)']
#     species, reactions = rbuilder.get_species_reactions_from_compounds(initial_comp)
#     species_conv = convert_species_tag_for_engine(species)
#     reactions_conv = convert_species_tags_for_reactions(reactions)
#     # engine_species, engine_idxs = create_list_of_species_engine(species_conv)
#     dict_indexes_species = get_dict_indexes_of_species_to_variable_position(species_conv)
#     engine_reactions = create_list_of_reactions_engine(reactions_conv, dict_indexes_species)

#     assert isinstance(engine_reactions[0], pyequion.EqReaction)
#     assert len(engine_reactions) == 7
#     assert engine_reactions[0].idx_reaction_db >= 0
#     assert engine_reactions[0].idx_species[0] >= 0

# def test_create_list_of_mass_balances_engine_nahco3():
#     feed_compounds = ['NaHCO3']
#     initial_comp = ['H2O', 'NaHCO3', 'CO2(g)']
#     element = ['Na']
#     species, reactions = rbuilder.get_species_reactions_from_compounds(initial_comp)
#     mb_list_engine = create_list_of_mass_balances_engine(species, element,
#         feed_compounds)
#     assert len(mb_list_engine) == 1
#     assert mb_list_engine[0].idx_feed == [(0, 1.0)]
#     species_conv = convert_species_tag_for_engine(species)
#     dict_indexes_species = get_dict_indexes_of_species_to_variable_position(species_conv)
#     assert_list_in_numba(mb_list_engine[0].idx_species, dict_indexes_species['Nap'])
#     pass

# --------------------------------------------
# 	APPLYED CASES
# --------------------------------------------


def test_engine_nahco3_solve_15mM_open():
    EXPECTED = {
        "sc": (1336.4, -15.0),
        "I": (16.073e-3, 0.1),
        "pH": (9.24, 1.0),
        "DIC": (1.34e01 * 1e-3, -1),
    }
    TK = 25.0 + 273.15
    cNaHCO3 = 15e-3
    args = (np.array([cNaHCO3]), TK, pyequion.pCO2_ref)

    feed_compounds = ["NaHCO3"]
    initial_feed_mass_balance = None
    element_mass_balance = None
    closing_equation_type = ClosingEquationType.OPEN

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        element_mass_balance,
        initial_feed_mass_balance,
    )

    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    assert_solution_result(solution, EXPECTED)


def test_engine_cacl2_solve_open_0p01():
    EXPECTED = {
        "sc": (1336.4, -200),
        "I": (16.073e-3, 200),  # ONLY PH KOWN
        "pH": (5.61, 1.0),
        "DIC": (1.34e01 * 1e-3, -1),
    }
    TK = 25.0 + 273.15
    c_feed = 5 * 1e-3  # Forcing to be an array
    args = (np.array([c_feed]), TK, pyequion.pCO2_ref)

    """
    Big Issue: How to automate the known-mass-balance
    - A specie is removed (CaCl2)
    - A specie is still a variable (Ca)
    - A specie is placed as known (Cl)
    """

    feed_compounds = ["CaCl2"]
    initial_feed_mass_balance = ["Cl-"]
    closing_equation_type = ClosingEquationType.OPEN

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        initial_feed_mass_balance=initial_feed_mass_balance,
    )

    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    idx = sys_eq.idx_control.idx
    x_guess = np.full(idx["size"], -1e-3)
    x_guess[idx["Ca++"]] = np.log10(c_feed)
    solution = solve_equilibrium(
        sys_eq,
        x_guess=x_guess,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    pyequion.print_solution(solution)
    assert_solution_result(solution, EXPECTED)


def test_engine_with_parenthesis_Mn():
    EXPECTED = {
        "sc": (1336.4, -15.0),
        "I": (17.959e-3, 3.0),
        "pH": (9.18, 1.0),
        "DIC": (1.56e02 * 1e-3, 1.0),
        "SI": {
            "Pyrochroite": (0.61, 1.0),
            "Rhodochrosite": (4.65, 50.0),  # error in thisone...
        },
    }
    TK = 25.0 + 273.15
    cNaHCO3 = 150e-3
    args = (np.array([cNaHCO3]), TK, pyequion.pCO2_ref)

    feed_compounds = ["MnCO3"]
    initial_feed_mass_balance = None
    # initial_feed_mass_balance = ['Cl-']
    element_mass_balance = None
    closing_equation_type = ClosingEquationType.OPEN

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        element_mass_balance,
        initial_feed_mass_balance,
    )

    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
    )

    meanMnCO3 = pyequion.get_mean_activity_coeff(solution, feed_compounds[0])
    assert np.isclose(
        meanMnCO3, solution.gamma[sys_eq.idx_control.idx["Mn++"]], 1e-2
    )
    assert_solution_result(solution, EXPECTED)
    assert compare_with_expected_perc_tuple(
        solution.saturation_index["Pyrochroite"], EXPECTED["SI"]["Pyrochroite"]
    )
    assert compare_with_expected_perc_tuple(
        solution.saturation_index["Rhodochrosite"],
        EXPECTED["SI"]["Rhodochrosite"],
    )


def test_engine_mix_solve_closed():
    EXPECTED = {
        "sc": (1339.4, 20.0),
        "I": (16.102e-3, 1.0),
        "pH": (9.24, 1.0),
        "DIC": (1.34e01 * 1e-3, -1),
    }
    TK = 25.0 + 273.15
    cNaHCO3 = 15e-3
    cCaCl2 = 0.02e-3
    carbone_total = EXPECTED["DIC"][0]  # 1.34e+01 * 1e-3
    args = (
        np.array([cNaHCO3, cCaCl2]),
        TK,
        carbone_total,
    )  # Instead of pCO2->DIC

    feed_compounds = ["NaHCO3", "CaCl2"]
    initial_feed_mass_balance = ["Cl-"]
    closing_equation_type = ClosingEquationType.CARBON_TOTAL

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        initial_feed_mass_balance=initial_feed_mass_balance,
    )

    x_guess = np.full(sys_eq.idx_control.idx["size"], -1e-4)

    solution = solve_equilibrium(
        sys_eq,
        x_guess=x_guess,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    pyequion.print_solution(solution)
    # input('')
    assert_solution_result(solution, EXPECTED)


def test_engine_mix_closed_add_IS():
    EXPECTED = {
        "sc": (1314.9, 15),
        "I": (15.083e-3, 1.0),
        "pH": (8.2, 1.0),
        "DIC": (1.5e01 * 1e-3, -1),
        "SI": {
            "Calcite": (-0.53, 5.0),
            "Aragonite": (-0.67, 5.0),
            "Halite": (-7.91, 5.0),
        },
    }
    TK = 25.0 + 273.15
    cNaHCO3 = 15e-3
    cCaCl2 = 0.02e-3
    carbone_total = cNaHCO3
    args = (
        np.array([cNaHCO3, cCaCl2]),
        TK,
        carbone_total,
    )  # Instead of pCO2->DIC

    feed_compounds = ["NaHCO3", "CaCl2"]
    initial_feed_mass_balance = ["Cl-"]

    sys_eq = create_equilibrium(
        feed_compounds, initial_feed_mass_balance=initial_feed_mass_balance
    )

    x_guess = np.full(sys_eq.idx_control.idx["size"], -1e-4)

    solution = solve_equilibrium(
        sys_eq,
        x_guess=x_guess,
        args=args,
        # jac=nahco3_residual_jacobian
    )

    pyequion.print_solution(solution)
    assert_solution_result(solution, EXPECTED)
    assert compare_with_expected_perc_tuple(
        solution.saturation_index["Halite"], EXPECTED["SI"]["Halite"]
    )
    assert compare_with_expected_perc_tuple(
        solution.saturation_index["Calcite"], EXPECTED["SI"]["Calcite"]
    )
    assert compare_with_expected_perc_tuple(
        solution.saturation_index["Aragonite"], EXPECTED["SI"]["Aragonite"]
    )


def test_engine_mix_closed_add_IS_dic_modified():
    EXPECTED = {
        "sc": (1339.4, 15),
        "I": (16.102e-3, 1.0),
        "pH": (9.24, 1.0),
        "DIC": (1.34e01 * 1e-3, -1),
        "SI": (0.20, 200.0),  # FIXME: High error!
    }
    TK = 25.0 + 273.15
    cNaHCO3 = 15e-3
    cCaCl2 = 0.02e-3
    carbone_total = EXPECTED["DIC"][0]  # 1.34e+01 * 1e-3
    args = (
        np.array([cNaHCO3, cCaCl2]),
        TK,
        carbone_total,
    )  # Instead of pCO2->DIC

    feed_compounds = ["NaHCO3", "CaCl2"]
    initial_feed_mass_balance = ["Cl-"]
    closing_equation_type = ClosingEquationType.CARBON_TOTAL

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        initial_feed_mass_balance=initial_feed_mass_balance,
    )

    x_guess = np.full(sys_eq.idx_control.idx["size"], -1e-4)

    solution = solve_equilibrium(
        sys_eq,
        x_guess=x_guess,
        args=args,
        # jac=nahco3_residual_jacobian
    )

    pyequion.print_solution(solution)
    assert_solution_result(solution, EXPECTED)
    assert compare_with_expected_perc_tuple(
        solution.saturation_index["Calcite"], EXPECTED["SI"]
    )


def test_engine_mix_default_feed_mb_closed():
    EXPECTED = {
        "sc": (1339.4, 15),
        "I": (16.102e-3, 1.0),
        "pH": (9.24, 1.0),
        "DIC": (1.34e01 * 1e-3, -1),
    }
    TK = 25.0 + 273.15
    cNaHCO3 = 15e-3
    cCaCl2 = 0.02e-3
    carbone_total = EXPECTED["DIC"][0]  # 1.34e+01 * 1e-3
    args = (
        np.array([cNaHCO3, cCaCl2]),
        TK,
        carbone_total,
    )  # Instead of pCO2->DIC

    feed_compounds = ["NaHCO3", "CaCl2"]
    initial_feed_mass_balance = ["Cl-"]
    # element_mass_balance = ['Na', 'Ca']
    closing_equation_type = ClosingEquationType.CARBON_TOTAL

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        element_mass_balance=None,
        initial_feed_mass_balance=initial_feed_mass_balance,
    )

    x_guess = np.full(sys_eq.idx_control.idx["size"], -1e-4)

    solution = solve_equilibrium(
        sys_eq,
        x_guess=x_guess,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    assert_solution_result(solution, EXPECTED)


# def test_engine_mix_solve_near_saturation_closed():
#     EXPECTED = {
#         'sc': (1488.7, -200),
#         'I': (17.305e-3, 1.0),
#         'pH': (8.1, 1.0),
#         'DIC': (1.40e+01*1e-3, -1),
#         'SI': (1.13, 5.0),
#     }
#     solution, dict_map_idx = solve_near_saturation_mix_case(EXPECTED['DIC'][0])
#     assert_solution_result(solution, EXPECTED)
#     assert compare_with_expected_perc_tuple(solution.SI[1], EXPECTED['SI'])


def test_engine_baco3_solve_1mM_open():
    EXPECTED = {  # AQION
        "sc": (198.5, 15),  # FIXME: add parameters for BaCO3 conductivity
        "I": (2.9652e-3, 4.0),
        "pH": (8.48, 2.0),
        "DIC": (1.97e-3, 5.0),
        "SI": (0.85, 3.0),
    }
    TK = 25.0 + 273.15
    cFeed = 1e-3
    args = (np.array([cFeed]), TK, pyequion.pCO2_ref)

    feed_compounds = ["BaCO3"]
    # initial_feed_mass_balance = None
    # element_mass_balance = ['Ba']
    closing_equation_type = ClosingEquationType.OPEN

    sys_eq = create_equilibrium(
        feed_compounds, closing_equation_type=closing_equation_type
    )

    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    pyequion.print_solution(solution)
    assert_solution_result(solution, EXPECTED)
    assert compare_with_expected_perc_tuple(
        solution.saturation_index["Witherite"], EXPECTED["SI"]
    )


def test_engine_baco3_solve_1mM_closed():
    EXPECTED = {  # AQION
        "sc": (241.8, 15),  # FIXME: add parameters for BaCO3 conductivity
        "I": (3.092e-3, 4.0),
        "pH": (10.48, 2.0),
        "DIC": (1e-3, -1),
        "SI": (2.0, 3.0),
    }
    TK = 25.0 + 273.15
    cFeed = 1e-3
    DIC = EXPECTED["DIC"][0]
    args = (np.array([cFeed]), TK, DIC)

    feed_compounds = ["BaCO3"]
    # initial_feed_mass_balance = None
    # element_mass_balance = ['Ba']
    # closing_equation_type = ClosingEquationType.CARBONE_TOTAL

    sys_eq = create_equilibrium(feed_compounds)

    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    pyequion.print_solution(solution)
    assert_solution_result(solution, EXPECTED)
    assert compare_with_expected_perc_tuple(
        solution.saturation_index["Witherite"], EXPECTED["SI"]
    )


def test_engine_baco3_nahco3_solve_1mM_each_closed():
    EXPECTED = {  # AQION
        "sc": (300.1, 15),  # FIXME: add parameters for BaCO3 conductivity
        "I": (4.16e-3, 4.0),
        "pH": (10.04, 2.0),
        "DIC": (2e-3, -1),
        "SI": (2.08, 3.0),
    }
    TK = 25.0 + 273.15
    DIC = EXPECTED["DIC"][0]
    cNaHCO3 = 1e-3
    cBaCO3 = 1e-3
    args = (np.array([cNaHCO3, cBaCO3]), TK, DIC)

    feed_compounds = ["NaHCO3", "BaCO3"]
    # initial_feed_mass_balance = None
    # element_mass_balance = ['Na', 'Ba']
    # closing_equation_type = ClosingEquationType.CARBONE_TOTAL

    sys_eq = create_equilibrium(feed_compounds)

    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    assert_solution_result(solution, EXPECTED)
    assert compare_with_expected_perc_tuple(
        solution.saturation_index["Witherite"], EXPECTED["SI"]
    )


def test_engine_baco3_cacl2_solve_5_2_mM_closed():
    EXPECTED = {  # AQION
        "sc": (1099.5, 15),  # FIXME: add parameters for BaCO3 conductivity
        "I": (15.457e-3, 15.0),
        "pH": (10.71, 2.0),
        "DIC": (5e-3, -1),
        "SI-CaCO3-Calcite": (2.30, 10.0),
        "SI-BaCO3": (2.99, 10.0),
        # Include Vaterite, Aragonite, Amorph, Ikaite (6H2O)
    }

    sys_eq = pyequion.create_equilibrium(
        feed_compounds=["CaCl2", "BaCO3"], initial_feed_mass_balance=["Cl-"]
    )

    TK = 25.0 + 273.15
    cCaCl2 = 2e-3
    cBaCO3 = 5e-3
    DIC = 5e-3

    args = (np.array([cCaCl2, cBaCO3]), TK, DIC)
    solution = pyequion.solve_equilibrium(sys_eq, args=args)

    pyequion.print_solution(solution)

    assert_solution_result(solution, EXPECTED)
    assert compare_with_expected_perc_tuple(
        solution.saturation_index["Calcite"], EXPECTED["SI-CaCO3-Calcite"]
    )
    assert compare_with_expected_perc_tuple(
        solution.saturation_index["Witherite"], EXPECTED["SI-BaCO3"]
    )

    solution = pyequion.solve_equilibrium(
        sys_eq, args=args, allow_precipitation=True
    )

    pyequion.print_solution(solution)


def test_engine_zemaits_na2so4_caso4():
    # flake8: noqa
    "Was not able to get good aggrement"
    EXPECTED = {  # AQION
        "sc": (1099.5, 15),  # FIXME: add parameters for BaCO3 conductivity
        "I": (15.457e-3, 15.0),
        "pH": (10.71, 2.0),
        "DIC": (5e-3, -1),
        "SI-CaCO3-Calcite": (2.30, 10.0),
        "SI-BaCO3": (2.99, 10.0),
        # Include Vaterite, Aragonite, Amorph, Ikaite (6H2O)
    }

    sys_eq = pyequion.create_equilibrium(feed_compounds=["Na2SO4", "CaSO4"])

    TK = 25.0 + 273.15
    cNa2SO4 = 94.8e-3
    cCaSO4 = 21.8e-3

    args = (np.array([cNa2SO4, cCaSO4]), TK, np.nan)

    # solution = pyequion.solve_equilibrium(sys_eq, args=args)

    # pyequion.print_solution(solution)

    # assert_solution_result(solution, EXPECTED)
    # assert compare_with_expected_perc_tuple(solution.saturation_index['Calcite'], EXPECTED['SI-CaCO3-Calcite'])
    # assert compare_with_expected_perc_tuple(solution.saturation_index['Witherite'], EXPECTED['SI-BaCO3'])

    solution = pyequion.solve_equilibrium(
        sys_eq, args=args, allow_precipitation=True
    )

    pyequion.print_solution(solution)


def test_engine_greg_andr_caso4():

    sys_eq = pyequion.create_equilibrium(feed_compounds=["CaSO4"])

    TK = 25.0 + 273.15
    # cNa2SO4 = 94.8e-3
    cCaSO4 = 15.6e-3

    args = (np.array([cCaSO4]), TK, np.nan)
    solution = pyequion.solve_equilibrium(sys_eq, args=args)

    pyequion.print_solution(solution)

    # assert_solution_result(solution, EXPECTED)
    # assert compare_with_expected_perc_tuple(solution.saturation_index['Calcite'], EXPECTED['SI-CaCO3-Calcite'])
    # assert compare_with_expected_perc_tuple(solution.saturation_index['Witherite'], EXPECTED['SI-BaCO3'])

    solution = pyequion.solve_equilibrium(
        sys_eq, args=args, allow_precipitation=True
    )

    pyequion.print_solution(solution)


def test_engine_baco3_cacl2_nahco3_solve_1mM_each_closed():
    EXPECTED = {  # AQION
        "sc": (484.5, 15),  # FIXME: add parameters for BaCO3 conductivity
        "I": (6.213e-3, 15.0),
        "pH": (9.89, 2.0),
        "DIC": (2e-3, -1),
        "SI-Calcite": (1.73, 5.0),
        "SI-BaCO3": (1.91, 5.0),
    }
    TK = 25.0 + 273.15
    DIC = EXPECTED["DIC"][0]
    cCaCl2 = 1e-3
    cBaCO3 = 1e-3
    cNaHCO3 = 1e-3
    args = (np.array([cCaCl2, cBaCO3, cNaHCO3]), TK, DIC)

    feed_compounds = ["CaCl2", "BaCO3", "NaHCO3"]
    initial_feed_mass_balance = ["Cl-"]
    # element_mass_balance = ['Ca', 'Ba', 'Na']

    sys_eq = create_equilibrium(
        feed_compounds,
        initial_feed_mass_balance=initial_feed_mass_balance,
    )

    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    pyequion.print_solution(solution)
    assert_solution_result(solution, EXPECTED)
    assert compare_with_expected_perc_tuple(
        solution.saturation_index["Calcite"], EXPECTED["SI-Calcite"]
    )
    assert compare_with_expected_perc_tuple(
        solution.saturation_index["Witherite"], EXPECTED["SI-BaCO3"]
    )


# --------------------------------------------
# 	APPYED CASE - WITH IONS SPECIES
# --------------------------------------------


def test_engine_nap_hco3m_closed():
    EXPECTED = {
        "sc": (1310.8, -15.0),
        "I": (15.032e-3, 0.1),
        "pH": (8.2, 1.0),
        "DIC": (15 * 1e-3, -1),
    }
    TK = 25.0 + 273.15
    cNap = 15e-3
    cHCO3m = 15e-3
    carbone_total = 15e-3
    args = (np.array([cNap, cHCO3m]), TK, carbone_total)

    feed_compounds = ["Na+", "HCO3-"]
    # initial_feed_mass_balance = None
    # element_mass_balance = ['Na']
    # closing_equation_type = ClosingEquationType.CARBONE_TOTAL

    sys_eq = create_equilibrium(feed_compounds)
    # closing_equation_type, element_mass_balance,
    # initial_feed_mass_balance)

    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    assert_solution_result(solution, EXPECTED)


@pytest.mark.xfail(reason="Expected value for higher T to be obtained.")
def test_engine_nap_hco3m_higher_T_closed():
    EXPECTED = {
        "sc": (1310.8, -15.0),
        "I": (15.032e-3, 0.1),
        "pH": (8.2, 1.0),
        "DIC": (15 * 1e-3, -1),
    }
    TK = 80.0 + 273.15
    cNap = 15e-3
    cHCO3m = 15e-3
    carbone_total = 15e-3
    args = (np.array([cNap, cHCO3m]), TK, carbone_total)

    feed_compounds = ["Na+", "HCO3-"]
    # initial_feed_mass_balance = None
    # element_mass_balance = ['Na']
    # closing_equation_type = ClosingEquationType.CARBONE_TOTAL

    sys_eq = create_equilibrium(feed_compounds)
    # closing_equation_type, element_mass_balance,
    # initial_feed_mass_balance)

    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    assert_solution_result(solution, EXPECTED)


def test_engine_nap_hco3m_capp_clm_open():
    EXPECTED = {
        "sc": (1525.3, 25),  # FIXME
        "I": (20.3e-3, 3.0),
        "pH": (8.06, 1.0),
        "DIC": (15.0 * 1e-3, 5),
        "IS(s)": (2.39, -1),  # TODO nao incluido ainda
    }
    TK = 25.0 + 273.15
    cNap = 15e-3
    cHCO3m = 15e-3
    cBapp = 2e-3
    cClpp = 2 * 2e-3
    carbone_total = 15e-3
    # args = (np.array([cNap, cHCO3m, cBaqpp]), TK, carbone_total)
    args = (np.array([cNap, cHCO3m, cBapp, cClpp]), TK, carbone_total)

    feed_compounds = ["Na+", "HCO3-", "Ca++", "Cl-"]
    fixed_elements = ["Cl-"]
    closing_equation_type = ClosingEquationType.CARBON_TOTAL

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        # element_mass_balance,
        fixed_elements=fixed_elements,
    )

    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    pyequion.print_solution(solution)
    assert_solution_result(solution, EXPECTED)
    # assert compare_with_expected_perc_tuple(solution.c_molal[sys_eq.idx.BaCO3s], EXPECTED['BaCO3(s)'])


def test_engine_nap_hco3m_bacl2_open():
    EXPECTED = {
        "sc": (1525.3, 25),  # FIXME
        "I": (20.776e-3, 3.0),
        "pH": (9.22, 1.0),
        "DIC": (13.2 * 1e-3, 5),
        "IS(s)": (2.39, -1),  # TODO nao incluido ainda
    }
    TK = 25.0 + 273.15
    cNap = 15e-3
    cHCO3m = 15e-3
    cBaqpp = 2e-3
    # carbone_total = 15e-3
    # args = (np.array([cNap, cHCO3m, cBaqpp]), TK, carbone_total)
    args = (np.array([cNap, cHCO3m, cBaqpp]), TK, pyequion.pCO2_ref)

    feed_compounds = ["Na+", "HCO3-", "BaCl2"]
    initial_feed_mass_balance = ["Cl-"]
    # element_mass_balance = ['Na', 'Ba']
    closing_equation_type = ClosingEquationType.OPEN

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        # element_mass_balance,
        initial_feed_mass_balance=initial_feed_mass_balance,
    )

    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    pyequion.print_solution(solution)
    assert_solution_result(solution, EXPECTED)
    # assert compare_with_expected_perc_tuple(solution.c_molal[sys_eq.idx.BaCO3s], EXPECTED['BaCO3(s)'])


def test_engine_water_co2_solve_closed():
    EXPECTED = {  # from-aqion
        "sc": (8.2, -15.0),
        "I": (0.021e-3, 1.0),
        "pH": (4.68, 1.0),
        "DIC": (1e-3, -1),
    }
    TK = 25.0 + 273.15
    # cFeed = 15e-3
    carbone_total = EXPECTED["DIC"][0]  # mM
    args = (np.array([]), TK, carbone_total)

    feed_compounds = ["CO2"]
    initial_feed_mass_balance = None
    element_mass_balance = []
    closing_equation_type = ClosingEquationType.CARBON_TOTAL

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        element_mass_balance,
        initial_feed_mass_balance,
    )

    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    assert_solution_result(solution, EXPECTED)
    pass


# --------------------------------------------
# 	APPYED CASE - ENGINE - CO2(aq) CO32- AND H2O
# --------------------------------------------


def test_engine_water_co2_solve_fixed_pH_closing_equation():
    EXPECTED = {  # from-aqion
        "sc": (8.2, -15.0),
        "I": (0.021e-3, 1.0),
        "pH": (4.68, 1.0),
        "DIC": (1e-3, -1),
    }
    TK = 25.0 + 273.15
    # cFeed = 15e-3
    pH_fixed = EXPECTED["pH"][0]  # mM
    args = (np.array([]), TK, pH_fixed)

    feed_compounds = ["CO2"]
    initial_feed_mass_balance = None
    element_mass_balance = []
    closing_equation_type = ClosingEquationType.PH

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        element_mass_balance,
        initial_feed_mass_balance,
    )

    x_guess = np.full(sys_eq.idx_control.idx["size"], -2)

    solution, fsol, _ = solve_equilibrium(
        sys_eq, args=args, x_guess=x_guess, ret_fsol=True
    )
    print(fsol.message)
    assert_solution_result(solution, EXPECTED)
    pass


def test_engine_nap_hco3m_Capp_clm_closed():
    EXPECTED = {
        "sc": (1339.4, -200),
        "I": (16.102e-3, 1.0),
        "pH": (9.24, 1.0),
        "DIC": (1.34e01 * 1e-3, -1),
    }
    TK = 25.0 + 273.15
    cNap = 15e-3
    cHCO3m = 15e-3
    cCapp = 0.02e-3
    cClm = 0.02e-3
    carbone_total = EXPECTED["DIC"][0]
    # args = (np.array([cNap, cHCO3m, cBaqpp]), TK, carbone_total)

    feed_compounds = ["Na+", "HCO3-", "Ca++", "Cl-"]
    # initial_feed_mass_balance = ['Cl-']
    fixed_elements = ["Cl-"]
    allow_precipitation = False
    # element_mass_balance = ['Na', 'Ca']
    closing_equation_type = ClosingEquationType.CARBON_TOTAL

    sys_eq = create_equilibrium(
        feed_compounds,
        # closing_equation_type, element_mass_balance,
        closing_equation_type,
        allow_precipitation=allow_precipitation,
        fixed_elements=fixed_elements,
    )

    args = (np.array([cNap, cHCO3m, cCapp, cClm]), TK, carbone_total)
    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    pyequion.print_solution(solution)
    assert_solution_result(solution, EXPECTED)


def test_engine_nap_hco3m_Capp_clm_check_hco3_conc_closed():
    EXPECTED = {
        "sc": (1272.0, -1),
        "I": (15.885e-3, 1.0),
        "pH": (7.92, 1.0),
    }
    TK = 25.0 + 273.15
    cNap = 1e-3
    cHCO3m = 1e-3
    cCapp = 5e-3
    cClm = 10e-3  # CAREFUL HERE-> !! STOIC CL*2
    carbone_total = cHCO3m
    # args = (np.array([cNap, cHCO3m, cBaqpp]), TK, carbone_total)

    feed_compounds = ["Na+", "HCO3-", "Ca++", "Cl-"]
    # initial_feed_mass_balance = ['Cl-']
    fixed_elements = ["Cl-"]
    allow_precipitation = False
    # element_mass_balance = ['Na', 'Ca']

    sys_eq = create_equilibrium(
        feed_compounds,
        # element_mass_balance,
        allow_precipitation=allow_precipitation,
        fixed_elements=fixed_elements,
    )

    args = (np.array([cNap, cHCO3m, cCapp, cClm]), TK, carbone_total)
    # x_guess = np.full(nahco3_eq.idx.size, -1.0)
    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    pyequion.print_solution(solution)
    assert_solution_result(solution, EXPECTED)


def test_engine_mix_solve_1_and_5_closed():
    EXPECTED = {
        "sc": (1272.0, -1),
        "I": (15.885e-3, 1.0),
        "pH": (7.92, 1.0),
    }
    TK = 25.0 + 273.15
    cNaHCO3 = 1e-3
    cCaCl2 = 5e-3
    carbone_total = cNaHCO3  # 1.34e+01 * 1e-3
    args = (
        np.array([cNaHCO3, cCaCl2]),
        TK,
        carbone_total,
    )  # Instead of pCO2->DIC

    feed_compounds = ["NaHCO3", "CaCl2"]
    initial_feed_mass_balance = ["Cl-"]
    # closing_equation_type = ClosingEquationType.CARBONE_TOTAL

    sys_eq = create_equilibrium(
        feed_compounds, initial_feed_mass_balance=initial_feed_mass_balance
    )

    x_guess = np.full(sys_eq.idx_control.idx["size"], -1e-4)

    solution = solve_equilibrium(
        sys_eq,
        x_guess=x_guess,
        args=args,
        # jac=nahco3_residual_jacobian
    )
    pyequion.print_solution(solution)
    assert_solution_result(solution, EXPECTED)


def test_two_irreversible_cacl2_bacl2():
    EXPECTED = {
        "sc": (3417.7, -1),
        "I": (43.02e-3, 3.0),
        "pH": (7.91, 1.0),
    }
    feed_compounds = ["NaHCO3", "CaCl2", "BaCl2"]
    initial_feed_mass_balance = ["Cl-"]
    # closing_eq = ClosingEquationType.CARBON_TOTAL

    sys_eq = create_equilibrium(
        feed_compounds, initial_feed_mass_balance=initial_feed_mass_balance
    )

    TK = 25.0 + 273.15
    cNaHCO3 = 15e-3
    cCaCl2 = 5e-3
    cBaCl2 = 5e-3
    carbone_total = cNaHCO3
    args = (np.array([cNaHCO3, cCaCl2, cBaCl2]), TK, carbone_total)
    solution = solve_equilibrium(sys_eq, args=args)
    pyequion.print_solution(solution)

    assert_solution_result(solution, EXPECTED)


def test_engine_mix_solve_1_and_5_closed_precipitation():
    EXPECTED = {
        "sc": (6977.5, -1),
        "I": (99.779e-3, 1.0),
        "pH": (8.85, 1.0),
        "SI": {
            "Calcite": (0.00, 1.0),
            "Aragonite": (-0.14, 1.0),
            "Vaterite": (-0.57, 1.0),
        },
        "sat-conc": {
            "Calcite": (80.0e-3, 1.0),
            "Aragonite": (0.0, -1.0),
            "Vaterite": (0.0, -1.0),
        },
    }
    TK = 25.0 + 273.15
    cCaCO3 = 80e-3
    cCaCl2 = 33.25e-3
    carbone_total = cCaCO3  # 1.34e+01 * 1e-3
    args = (
        np.array([cCaCO3, cCaCl2]),
        TK,
        carbone_total,
    )  # Instead of pCO2->DIC

    feed_compounds = ["CaCO3", "CaCl2"]
    initial_feed_mass_balance = ["Cl-"]
    allow_precipitation = True
    closing_equation_type = ClosingEquationType.CARBON_TOTAL

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        initial_feed_mass_balance=initial_feed_mass_balance,
    )

    # This was with an old API, the create_equilibrium is created without solid reaction and
    #  only in the solution function the code is adjust to treat the precipitation
    # x_guess = np.full(sys_eq.idx_control.idx['size'], -1e-1)

    solution = solve_equilibrium(
        sys_eq,
        args=args,
        # jac=nahco3_residual_jacobian
        allow_precipitation=allow_precipitation,
    )
    pyequion.print_solution(solution)
    assert_solution_result(solution, EXPECTED)
    for tag in EXPECTED["SI"]:
        assert compare_with_expected_perc_tuple(
            solution.saturation_index[tag], EXPECTED["SI"][tag]
        )
        assert compare_with_expected_perc_tuple(
            solution.preciptation_conc[tag], EXPECTED["sat-conc"][tag]
        )


def test_engine_activity_external_ideal():
    EXPECTED = {
        "pH": (8.2806817914915, 10.0),  # IDEAL
    }
    TK = 25.0 + 273.15
    cNaHCO3 = 15e-3
    cCaCl2 = 0.02e-3
    carbone_total = cNaHCO3
    c_feed = np.array([cNaHCO3, cCaCl2])
    args = (c_feed, TK, carbone_total)  # Instead of pCO2->DIC

    feed_compounds = ["NaHCO3", "CaCl2"]
    initial_feed_mass_balance = ["Cl-"]
    element_mass_balance = ["Na", "Ca"]
    closing_equation_type = ClosingEquationType.CARBON_TOTAL

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        element_mass_balance,
        initial_feed_mass_balance,
    )

    # setup_log_gamma(sys_eq, TK, c_feed)

    x_guess = np.full(sys_eq.idx_control.idx["size"], -1e-4)

    solution = solve_equilibrium(
        sys_eq,
        x_guess=x_guess,
        args=args,
        setup_log_gamma_func=pyequion.setup_log_gamma_ideal,
        calc_log_gamma=pyequion.calc_log_gamma_ideal,
        # jac=nahco3_residual_jacobian
    )
    pyequion.print_solution(solution)
    assert np.isclose(EXPECTED["pH"][0], solution.pH)


def test_engine_activity_external_debye_huckel():
    EXPECTED = {
        "pH": (8.2, 1.0),
    }
    TK = 25.0 + 273.15
    cNaHCO3 = 15e-3
    cCaCl2 = 0.02e-3
    carbone_total = cNaHCO3
    c_feed = np.array([cNaHCO3, cCaCl2])
    args = (c_feed, TK, carbone_total)  # Instead of pCO2->DIC

    feed_compounds = ["NaHCO3", "CaCl2"]
    initial_feed_mass_balance = ["Cl-"]
    element_mass_balance = ["Na", "Ca"]
    closing_equation_type = ClosingEquationType.CARBON_TOTAL

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        element_mass_balance,
        initial_feed_mass_balance,
    )

    # setup_log_gamma(sys_eq, TK, c_feed)

    x_guess = np.full(sys_eq.idx_control.idx["size"], -1e-4)

    solution = solve_equilibrium(
        sys_eq,
        x_guess=x_guess,
        args=args
        # jac=nahco3_residual_jacobian
    )
    pyequion.print_solution(solution)
    assert np.isclose(EXPECTED["pH"][0], solution.pH, 1e-2)


def test_engine_activity_external_debye_huckel_mean_coef_for_neutral():
    EXPECTED = {
        "pH": (8.2, 1.0),
    }
    TK = 25.0 + 273.15
    cNaHCO3 = 15e-3
    cCaCl2 = 0.02e-3
    carbone_total = cNaHCO3
    c_feed = np.array([cNaHCO3, cCaCl2])
    args = (c_feed, TK, carbone_total)  # Instead of pCO2->DIC

    feed_compounds = ["NaHCO3", "CaCl2"]
    initial_feed_mass_balance = ["Cl-"]
    element_mass_balance = ["Na", "Ca"]
    closing_equation_type = ClosingEquationType.CARBON_TOTAL

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        element_mass_balance,
        initial_feed_mass_balance,
    )

    # setup_log_gamma(sys_eq, TK, c_feed)

    x_guess = np.full(sys_eq.idx_control.idx["size"], -1e-4)

    solution = solve_equilibrium(
        sys_eq,
        x_guess=x_guess,
        args=args,
        # setup_log_gamma_func=pyequion.act.setup_log_gamma_bdot_mean_activity_neutral,
        # calc_log_gamma=pyequion.act.calc_log_gamma_dh_bdot_mean_activity_neutral
        activity_model_type=pyequion.TypeActivityCalculation.DEBYE_MEAN
        # jac=nahco3_residual_jacobian
    )
    pyequion.print_solution(solution)
    assert np.isclose(EXPECTED["pH"][0], solution.pH, 1e-2)


def test_engine_caco3_cacl2_pitzer():
    TK = 25.0 + 273.15
    cNaHCO3 = 80e-3
    cCaCl2 = 33.25e-3
    args = (np.array([cNaHCO3, cCaCl2]), TK, pyequion.pCO2_ref)

    feed_compounds = ["CaCO3", "CaCl2"]
    initial_feed_mass_balance = ["Cl-"]
    closing_equation_type = ClosingEquationType.OPEN

    sys_eq = create_equilibrium(
        feed_compounds,
        closing_equation_type,
        initial_feed_mass_balance=initial_feed_mass_balance,
    )

    x_guess = np.full(sys_eq.idx_control.idx["size"], -1e-2)

    solution = solve_equilibrium(
        sys_eq,
        x_guess=x_guess,
        args=args,
        # jac=nahco3_residual_jacobian
        setup_log_gamma_func=pyequion.setup_log_gamma_pitzer,
        calc_log_gamma=pyequion.calc_log_gamma_pitzer,
    )
    pyequion.print_solution(solution)
    print(solution.specie_names)
    assert np.isclose(
        9.327, solution.pH, 1e-2
    )  # 9.45 is D.H, check for Pitzer with phreeqc


# def test_dynamically_compilation():
#     EXPECTED = {
#         'sc': (1272.0, 10),
#         'I': (15.885e-3, 1.0),
#         'pH': (7.92, 1.0),
#     }
#     TK = 25.0 + 273.15
#     cNaHCO3 = 1e-3
#     cCaCl2 = 5e-3
#     carbone_total = cNaHCO3 # 1.34e+01 * 1e-3
#     args = (np.array([cNaHCO3, cCaCl2]), TK, carbone_total) #Instead of pCO2->DIC

#     feed_compounds = ['NaHCO3', 'CaCl2']
#     initial_feed_mass_balance = ['Cl-']
#     closing_equation_type = ClosingEquationType.CARBON_TOTAL

#     compilation.dynamically_compile()

#     sys_eq = create_equilibrium(feed_compounds,
#         closing_equation_type,
#         initial_feed_mass_balance=initial_feed_mass_balance)

#     x_guess = np.full(sys_eq.idx_control.idx['size'], -1e-4)

#     solution = solve_equilibrium(sys_eq, args=args)
#     pyequion.print_solution(solution)
#     assert_solution_result(solution, EXPECTED)

#     assert True

# --------------------------------------------
# 	MINOR TESTS
# --------------------------------------------
def test_get_species_of_reaction():
    reac_species = rbuilder.get_reactions_species(EXAMPLE_REACTIONS[0])
    assert "H+" in reac_species
    assert "OH-" in reac_species
    assert "H2O" in reac_species
    assert "type" not in reac_species


def test_converted_signals_plus_minus():
    comps = [
        "H2O",
        "NaHCO3",
        "CaCl2",
        "H+",
        "OH-",
        "Na+",
        "NaOH",
        "HCO3-",
        "CO2",
        "CO2(g)",
        "CO3--",
        "NaCO3-",
        "Na2CO3",
        "Ca++",
        "CaHCO3+",
        "CaCO3",
        "CaOH+",
        "Cl-",
    ]
    species_conv = rbuilder.convert_species_tag_for_engine(comps)
    for s in species_conv:
        assert "+" not in s
        assert "-" not in s
        assert "(g)" not in s


def test_converted_signals_plus_minus_in_reactions():
    reacs_conv = rbuilder.convert_species_tags_for_reactions(EXAMPLE_REACTIONS)
    for reac in reacs_conv:
        for sp in reac:
            assert "+" not in sp
            assert "-" not in sp
            assert "(g)" not in sp


# def test_print_solution_results():
#     sol = pyequion.SolutionResult(np.array([1.0]), np.array([1.0]), 7.0, 16.0, 1400.0, 1.0, np.array([0.0]), np.array([0.0]), [''], np.array([0.0]))
#     pyequion.print_solution(sol)
#     assert True


def test_feed_indexes_for_mb():
    feed_compounds = ["NaHCO3", "CaCl2"]
    element = "Na"
    # tags_species_collected = ['Na+', 'NaOH', 'NaCO3-', 'NaHCO3', 'Na2CO3']
    idxs_coefs_feed = rbuilder.get_indexes_feed_for_mass_balance(
        feed_compounds, element
    )
    assert idxs_coefs_feed[0][0] == 0
    assert idxs_coefs_feed[0][1] == 1.0
    assert len(idxs_coefs_feed) == 1


# #--------------------------------------------
# #	Symbolic Residual
# #--------------------------------------------
# @pytest.fixture()
# def fxtr_mix():
#     feed_compounds = ['NaHCO3', 'CaCl2']
#     initial_feed_mass_balance = ['Cl-']
#     element_mass_balance = ['Na', 'Ca']
#     is_open_system = False

#     sys_eq = create_equilibrium(feed_compounds,
#         is_open_system, element_mass_balance,
#         initial_feed_mass_balance)
#     return sys_eq

# def test_engine_mix_residual_symbolic():

#     feed_compounds = ['NaHCO3', 'CaCl2']
#     initial_feed_mass_balance = ['Cl-']
#     element_mass_balance = ['Na', 'Ca']
#     is_open_system = False

#     sys_eq = create_equilibrium(feed_compounds,
#         is_open_system, element_mass_balance,
#         initial_feed_mass_balance)

#     x = sympy.symbols('x0:{}'.format(sys_eq.idx_control.idx['size']))
#     args_symbols = sympy.symbols('args0:4')
#     args = ([args_symbols[0], args_symbols[1]],
#             args_symbols[2], args_symbols[3])
#     mod_sym.prepare_for_sympy_substituting_numpy()

#     res = sys_eq.residual(x, args)

#     # J = mod_sym.obtain_symbolic_jacobian(res, x)

#     # s = mod_sym.string_lambdastr_as_function(
#     #     J, x, args, 'engine_mix_residual_jacobian',
#     #     use_numpy=True, include_imports=True
#     # )
#     # s = mod_sym.numbafy_function_string(s, numba_kwargs_string='cache=True')
#     # fname = 'engine_mix_residual_jacobian.py'
#     # loc_path = os.path.join(os.path.dirname(__file__), fname)
#     # mod_sym.save_function_string_to_file(s, loc_path)

#     pass

# --------------------------------------------
# 	ASSERT HELPERS
# --------------------------------------------
def assert_list_in_numba(list_, el):
    for item in list_:
        if el == item:
            return True
    return False


if __name__ == "__main__":
    # test_engine_activity_external_debye_huckel()
    test_engine_cacl2_solve_open_0p01()
    pass
