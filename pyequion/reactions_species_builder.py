import logging
import os
import re
from enum import Enum

# import json
from functools import reduce

import numpy as np
from ordered_set import OrderedSet

from . import core, utils
from .activity_coefficients import TypeActivityCalculation

# if os.getenv('NUMBA_DISABLE_JIT') != "1":
#     from numba.typed import List, Dict
# else:
#     List = list
#     Dict = dict #parei aqui
from .utils_for_numba import create_nb_Dict, create_nb_List
from .core import DEFAULT_DB_FILES, Dict, List, SolutionResult
from .properties_utils import solve_with_exception
from .utils import ClosingEquationType

# --------------------------------------------
# 	REACTIONS LIST DEFINITIONS DATABASE
# --------------------------------------------


# FIXME: Irreversible Equations
# Necessary for the correct element identification
# Also Including CO2 = CO2(g) Henry constant, but should be treat later as a gas FIXME
reactionsListIrreversible = [
    # FIXME Automatic irreversible for: Cl-
    {"CaCl2": -1, "Ca++": 1, "Cl-": 2, "type": "irrev", "id_db": -1},
    {"NaCl": -1, "Na+": 1, "Cl-": 1, "type": "irrev", "id_db": -1},
    {"KCl": -1, "K+": 1, "Cl-": 1, "type": "irrev", "id_db": -1},
    {"KOH": -1, "K+": 1, "OH-": 1, "type": "irrev", "id_db": -1},
    {"MgCl2": -1, "Mg++": 1, "Cl-": 2, "type": "irrev", "id_db": -1},
    {"K2SO4": -1, "K+": 2, "SO4--": 1, "type": "irrev", "id_db": -1},
    {"BaCl2": -1, "Ba++": 1, "Cl-": 2, "type": "irrev", "id_db": -1},
    {"NaHSO4": -1, "Na+": 1, "HSO4-": 1, "type": "irrev", "id_db": -1},
    {"Al2(SO4)3": -1, "Al+++": 2, "SO4--": 3, "type": "irrev", "id_db": -1},
    # {'CO2': 1, 'CO2(g)': -1, 'type': 'henry', 'log_K25': -1.468, 'log_K_coefs': ''}, #FIXME
]

reactionsListVapourPhase = [
    {
        "CO2": 1,
        "CO2(g)": -1,
        "type": "henry",
        "log_K25": -1.468,
        "log_K_coefs": "",
    },  # FIXME,  but logK(T) is hardcoded for CO2(g)
    # {
    #     "CO2-": 1.0,
    #     "CO2(g)": -1.0,
    #     "log_K25": "",
    #     "deltah": "",
    #     "log_K_coefs": [ #enter here
    #         108.3865,
    #         0.01985076,
    #         -6919.53,
    #         -40.45154,
    #         -669365
    #     ],
    #     "type": "henry"
    # },
]


reactionsListAqueous = [
    {"H2O": -1, "H+": +1, "OH-": +1, "type": "rev", "id_db": 10},
    {"HCO3-": -1, "H+": +1, "CO3--": +1, "type": "rev", "id_db": 9},
    {"HCO3-": 1, "H+": 1, "H2O": -1, "CO2": -1, "type": "rev", "id_db": 8},
    {"CO2": 1, "CO2(g)": -1, "type": "henry", "id_db": 7},
    {"NaHCO3": 1, "Na+": -1, "HCO3-": -1, "type": "rev", "id_db": 2},
    {"NaCO3-": 1, "Na+": -1, "CO3--": -1, "type": "rev", "id_db": 1},
    {"Na2CO3": 1, "Na+": -2, "CO3--": -1, "type": "rev", "id_db": 3},
    {"NaOH": 1, "H+": 1, "Na+": -1, "H2O": -1, "type": "rev", "id_db": 0},
    {"CaCl2": -1, "Ca++": 1, "Cl-": 2, "type": "irrev", "id_db": -1},
    {"CaHCO3+": 1, "HCO3-": -1, "Ca++": -1, "type": "rev", "id_db": 11},
    {"CaCO3": 1, "CO3--": -1, "Ca++": -1, "type": "rev", "id_db": 12},
    {"CaOH+": 1, "H+": 1, "H2O": -1, "Ca++": -1, "type": "rev", "id_db": 4},
    {"BaCO3": 1, "Ba++": -1, "CO3--": -1, "type": "rev", "id_db": 16},
    {"BaHCO3+": 1, "Ba++": -1, "HCO3-": -1, "type": "rev", "id_db": 17},
    {"BaOH+": 1, "H+": 1, "H2O": -1, "Ba++": -1, "type": "rev", "id_db": 18},
    {"BaCl2": -1, "Ba++": 1, "Cl-": 2, "type": "irrev", "id_db": -1},
    {"NaCl": -1, "Na+": 1, "Cl-": 1, "type": "irrev", "id_db": -1},
]

reactionsListPreciptation = [
    {"CaCO3(s)": -1, "CO3--": +1, "Ca++": +1, "type": "rev", "id_db": 13},
    {"BaCO3(s)": -1, "CO3--": +1, "Ba++": +1, "type": "rev", "id_db": 19},
]

RX_CASE = r"[A-Z][^A-Z]*"
RX_NO_SIGNAL = r"[+-]"
RX_PRNTHS = r"(\(\w+\)\d?)"


def disable_logging():
    logging.basicConfig(level=logging.CRITICAL)


def create_equilibrium_from_reaction_and_species(
    reactions,
    species,
    known_tags,
    element_mass_balance,
    feed_compounds,
    initial_feed_mass_balance,
    closing_equation_type,
    return_intermediaries=False,
    # activity_model_type: TypeActivityCalculation=TypeActivityCalculation.DEBYE,
    fixed_elements=None,
    database_files=DEFAULT_DB_FILES,
    possible_solid_reactions_in=None,
    closing_equation_element=None,  # TODO...
    allow_precipitation=False,
    solid_reactions_in=None,  # just to save for easier serialization
):
    reacs_irrev = get_reacs_irrev_list(reactions, species)

    species = re_organize_species_to_non_variables_at_end(species, known_tags)

    "Species"
    engine_species = create_list_of_species_engine(
        species
    )  # , species_activity_db)

    "Fill with Conductivity Parameters"
    fill_conductivity_in_species_from_db(
        engine_species, database_files["species"]
    )

    engine_idxs = create_Indexes_instance(species, len(known_tags))

    "Mass Balances"
    mb_list_engine = create_list_of_mass_balances_engine(
        species,
        element_mass_balance,
        engine_idxs,
        feed_compounds,
        closing_equation_type,
        initial_feed_mass_balance,
        fixed_elements,
    )  # FIXME: remove mass balance detection logic to another function

    "Store the number of feeds"
    num_of_feeds = len(feed_compounds)

    "Known specie from mass balance due to irreversible reaction -> How to automate it?"
    known_specie = obtain_know_specie_list_from_feed(
        initial_feed_mass_balance, engine_idxs.idx, reacs_irrev, feed_compounds
    )

    "Fixed Specie"
    if fixed_elements is not None:
        known_specie_constant = get_fixed_specie_as_known_specie(
            fixed_elements, engine_idxs.idx, feed_compounds
        )  # FIXME
        if len(known_specie_constant) > 0:
            known_specie += known_specie_constant

    "Reactions"
    engine_reactions = create_list_of_reactions_engine(
        reactions, engine_idxs.idx
    )

    if closing_equation_type == ClosingEquationType.CARBON_TOTAL:
        known_specie = adjust_known_specie_list_when_closed_system(
            species, engine_idxs.idx, known_specie
        )

    # DIC Tuple
    dic_tuple = get_dic_tuple(species, engine_idxs.idx)
    if not dic_tuple:  # len(dic_tuple) == 0:
        dic_tuple = [
            (-1, np.nan)
        ]  # DIC reference -> Remember Carbone total is just a mass balance, FIXME

    if not mb_list_engine:  # len(mb_list_engine) == 0:
        mb_list_engine = [core.DUMMY_MB]

    if not known_specie:  # len(known_specie) == 0:
        known_specie = [core.DUMMY_MB]

    if possible_solid_reactions_in is None:
        reactionsListPreciptationDB = utils.load_from_db(
            database_files["phases"]
        )
    else:
        reactionsListPreciptationDB = possible_solid_reactions_in

    "Possible Solid Formation: for S and IS calculation"
    _, solid_reactions = get_species_reactions_from_compounds(
        species, reactionsListPreciptationDB
    )
    if len(solid_reactions) > 0:
        solid_reactions_ = create_list_of_reactions_engine(
            solid_reactions, engine_idxs.idx
        )
    else:
        solid_reactions_ = core.List()
        solid_reactions_.append(core.DUMMY_EqREACTION)
        logging.info("No solid phase detected.")

    # Database as dict to store the info:
    feed_compounds_List = create_nb_List(feed_compounds)
    element_mass_balance_List = create_nb_List(element_mass_balance)
    initial_feed_mass_balance_List = create_nb_List(initial_feed_mass_balance)
    fixed_elements_List = create_nb_List(fixed_elements)

    reactionsStorage = numbafy_list_dict_str_float_reaction_storage(reactions)
    for item in reacs_irrev:
        d_mod = {
            k: float(v) for k, v in item.items() if k[0].isupper()
        }  # WIERD CHECK FOR UPPERCASE
        d_mod["irrev"] = 1.0
        aux = create_nb_Dict(d_mod)
        reactionsStorage.append(aux)

    # For Serialization: numba blinding:
    if solid_reactions_in is None or solid_reactions_in == []:
        d_aux_del = {"dummy": -1.0}
        aux = create_nb_Dict(d_aux_del)
        solid_reactions_in_ = core.List()
        solid_reactions_in_.append(aux)
        del solid_reactions_in_[0]
    else:
        # solid_reactions_in = core.create_numba_list_of_dict()
        solid_reactions_in_ = numbafy_list_dict_str_float_reaction_storage(
            solid_reactions_in
        )

    sys_eq = core.EquilibriumSystem(
        engine_species,
        engine_idxs,
        engine_reactions,
        mb_list_engine,
        known_specie,
        dic_tuple,
        solid_reactions_,
        num_of_feeds,  # activity_modeltype removed
        feed_compounds_List,
        closing_equation_type,
        element_mass_balance_List,
        initial_feed_mass_balance_List,
        fixed_elements_List,
        # database_files_Dict,
        reactionsStorage,
        allow_precipitation,  # Just to save this information (not used)
        solid_reactions_in_,  # reactionsStorage, #solid_reactions_in_, #Just to save this information (not used)
        # -1
    )
    if return_intermediaries:
        return (
            sys_eq,
            species,
            reactions,
            engine_idxs.idx,
            engine_idxs,
            mb_list_engine,
            known_specie,
            dic_tuple,
        )
    else:
        return sys_eq


def numbafy_list_dict_str_float_reaction_storage(list_in):
    r = core.List()
    for item in list_in:
        d_mod = {
            k: float(v) for k, v in item.items() if k[0].isupper()
        }  # WIERD CHECK FOR UPPERCASE
        aux = create_nb_Dict(d_mod)
        r.append(aux)
    return r


def numbafy_list_dict_str_float_regular(list_in):
    r = core.List()
    for item in list_in:
        d_mod = {
            k: float(v) for k, v in item.items() if k[0].isupper()
        }  # WIERD CHECK FOR UPPERCASE
        aux = create_nb_Dict(d_mod)
        r.append(aux)
    return r


def fill_conductivity_in_species_from_db(species_engine, db_species):

    # Access db file
    db = utils.load_from_db(db_species)
    db_cond = db["conductivity"]
    for sp in species_engine:
        try:
            cond_molar = db_cond[sp.name]
        except KeyError:
            cond_molar = 0.0
        sp.set_cond_molar(cond_molar)

    return


def get_fixed_specie_as_known_specie(
    fixed_elements, dict_indexes_species_conv, feed_compounds
):
    known_species_constants = []
    for fixed_element in fixed_elements:
        idx_of_feed = [
            i for i, s in enumerate(feed_compounds) if fixed_element == s
        ][0]

        ini_known_idxs_mass_list = [dict_indexes_species_conv[fixed_element]]

        known_species_constants += [
            core.MassBalance(
                np.array(ini_known_idxs_mass_list),
                np.array([1.0]),
                [(idx_of_feed, 1.0)],
                use_constant_value=True,
            )
        ]
    return known_species_constants


def get_reactions_species_from_initial_configs(
    allow_precipitation, initial_comp, closing_equation_type, reactionList_
):

    species, reactions = get_species_reactions_from_compounds(
        initial_comp, reactionList_
    )

    return reactions, species


def get_initial_comp_and_known_tags_from_ini_config(
    feed_compounds,
    initial_feed_mass_balance,
    closing_equation_type,
    fixed_elements=None,
):
    initial_comp = ["H2O"]  # numba-issue
    for v in feed_compounds:
        initial_comp.append(v)
    known_tags = ["H2O"]
    # initial_comp = ['H2O'] + feed_compounds

    if initial_feed_mass_balance:
        known_tags += initial_feed_mass_balance

    if fixed_elements:
        known_tags += fixed_elements

    if closing_equation_type == ClosingEquationType.OPEN:
        initial_comp += ["CO2(g)"]
        known_tags += ["CO2(g)"]
    return initial_comp, known_tags


def adjust_known_specie_list_when_closed_system(
    species, dict_indexes_species_conv, known_specie
):
    tags_coefs_carbone_total = get_species_tags_with_an_element(species, "C")
    (
        species_indexes_carbone_total,
        species_coefs_carbone_total,
    ) = get_idx_coef_from_species_tag_list(
        tags_coefs_carbone_total, dict_indexes_species_conv
    )
    known_specie_carbone_total_RHS = [
        core.MassBalance(
            species_indexes_carbone_total,
            species_coefs_carbone_total,
            [(-1, -1.0)],  # DUMMY VALS NOT USED carbone_total
            use_constant_value=True,
        )
    ]
    known_specie = known_specie_carbone_total_RHS + known_specie
    return known_specie


def get_dic_tuple(species, dict_indexes_species_conv):
    tags_coefs_carbone_total = get_species_tags_with_an_element(species, "C")
    tags_coefs_carbone_total_no_solid = {
        key: val
        for key, val in tags_coefs_carbone_total.items()
        if "(s)" not in key
    }
    (
        species_indexes_carbone_total,
        species_coefs_carbone_total,
    ) = get_idx_coef_from_species_tag_list(
        tags_coefs_carbone_total_no_solid, dict_indexes_species_conv
    )
    dic_tuple = [
        (i, coef)
        for i, coef in zip(
            species_indexes_carbone_total, species_coefs_carbone_total
        )
    ]
    return dic_tuple


def get_reacs_irrev_list(reactions, species):
    reacs_irrev = []
    idx_rm = []
    for i, reaction in enumerate(reactions):
        if reaction["type"] == "irrev":
            _, reacs = utils.get_tags_of_prods_reacts(
                reaction
            )  # Irreversivible alaways considering the prod
            [species.remove(reac) for reac in reacs]
            # reactions.remove(reaction)
            idx_rm += [i]
            reacs_irrev += [reaction]
    [reactions.remove(r) for r in reacs_irrev]
    return reacs_irrev


def obtain_know_specie_list_from_feed(
    initial_feed_mass_balance,
    dict_indexes_species_conv,
    reacs_irrev,
    feed_compounds,
):
    if initial_feed_mass_balance:
        # ini_tag_mb_conv = convert_species_tag_for_engine(initial_feed_mass_balance)
        ini_known_idxs_mass_list = [
            dict_indexes_species_conv[tag] for tag in initial_feed_mass_balance
        ]

        ini_feed_mb_item = initial_feed_mass_balance[0]
        reac_irrev_present = [r for r in reacs_irrev if ini_feed_mb_item in r]
        # reac_irrev = reacs_irrev[0]
        tt = []
        for reac_irrev in reac_irrev_present:
            #
            # #START-FIXME
            # this logic is not completely specified:
            # - Does not loop for all irreversible equation
            # - Does not loop for the prods, only for the reacts
            # TODO: Return here with a new test case with more than a irreversible equation and
            # The prods is kind right, since the irrevesible equation will only give a side of the reaction
            #
            # #END-FIXME
            for i, comp in enumerate(feed_compounds):
                prods, reacs = utils.get_tags_of_prods_reacts(reac_irrev)
                if comp in reacs:
                    coef = reac_irrev[initial_feed_mass_balance[0]]
                    tt += [(i, float(coef))]
        if len(tt) > 0:
            known_specie = [
                core.MassBalance(
                    np.array(ini_known_idxs_mass_list),
                    np.array([1.0]),
                    tt,
                    False,
                )
            ]
        else:
            pass  # FIXME
            # known_specie = [core.MassBalance(
            #     np.array(ini_known_idxs_mass_list),
            #     np.array([1.0]),
            #     [(-1, -1.0)], #DUMMY VALS NOT USED carbone_total
            #     use_constant_value=True
            # )] #FAILED #FIXME PAREI AQUI - WAS WORKING BEFORE WAS JUST A NUMBA ISSUE, THUS FIX JUST THE NUMBA CONFLICT...
    else:
        known_specie = []
    return known_specie


def get_idx_coef_from_species_tag_list(
    tags_coefs_carbone_total, dict_indexes_species_conv
):
    species_indexes_carbone_total = np.array(
        [dict_indexes_species_conv[tag] for tag in tags_coefs_carbone_total]
    )
    species_coefs_carbone_total = np.array(
        list(tags_coefs_carbone_total.values()), dtype=np.float64
    )
    return species_indexes_carbone_total, species_coefs_carbone_total


def get_species_reactions_from_compounds(
    compounds, reactionList=reactionsListAqueous
):
    species = OrderedSet([c for c in compounds])
    reactions = []
    for c in compounds:
        walk_in_species_reactions(c, species, reactions, reactionList)
    return species, reactions


def are_others_in_side_of_reaction_kown(elements, species):
    tester = [e for e in elements if e in species]
    return len(tester) == len(elements)


def walk_in_species_reactions(c, species, reactions, reactionsList):
    for r in reactionsList:
        # DEBUG
        if r["type"] == "electronic":
            continue  # skipping electronic in this evaluation
        if r in reactions:
            continue
        if c not in r:
            continue
        # if "phase_name" in r:
        #     if "Calcite" == r["phase_name"]:
        #         pass
        prods, reacs = utils.get_tags_of_prods_reacts(r)
        # THIS WAY: SOLID ALLWAYS (-1 ; NEGATIVE)
        if (
            c in prods
            and are_others_in_side_of_reaction_kown(prods, species)
            and r["type"] != "irrev"
        ):  # IRREV ALWAYS SHOULD BE react -> prod
            reactions += [r]
            for in_element in reacs:
                if "phase_name" in r and "(s)" in in_element:
                    tag_add = in_element + "__" + r["phase_name"]
                    r[tag_add] = r.pop(in_element)  # update reaction tag!
                else:
                    tag_add = in_element
                species.add(tag_add)
                walk_in_species_reactions(
                    in_element, species, reactions, reactionsList
                )
        if c in reacs and are_others_in_side_of_reaction_kown(reacs, species):
            reactions += [r]
            for in_element in prods:
                species.add(in_element)
                walk_in_species_reactions(
                    in_element, species, reactions, reactionsList
                )


def replace_strings_for_engine_compatibility(string):
    rpl = string.replace("+", "p")
    rpl = rpl.replace("-", "m")
    rpl = rpl.replace("(g)", "g")
    rpl = rpl.replace("(s)", "s")
    return rpl


def convert_species_tag_for_engine(species):
    # new = set()
    new = []
    for el in species:
        new += [(replace_strings_for_engine_compatibility(el))]
    return new


def convert_species_tags_for_reactions(reactions_list):

    new_rections = []
    for reac in reactions_list:
        d = {}
        if "phase_name" in reac:
            for k, v in reac.items():
                if utils.check_validity_specie_tag_in_reaction_dict(k):
                    ele = replace_strings_for_engine_compatibility(k)
                else:
                    ele = k
                if "(s)" in k:
                    ele += "__" + reac["phase_name"]
                d[ele] = v
        else:
            for k, v in reac.items():
                if utils.check_validity_specie_tag_in_reaction_dict(k):
                    ele = replace_strings_for_engine_compatibility(k)
                else:
                    ele = k
                d[ele] = v
        new_rections += [d]
    return new_rections


# def get_dict_indexes_of_species_to_variable_position(species_tags_conv):
#     non_numba_indexes_species_dict = {}
#     for tag_db in tags_db:  # db position -> variable position
#         idxDB_match = [i for i, tag in enumerate(species_tags_conv) if tag == tag_db]
#         if len(idxDB_match) == 0:
#             ret = -1
#         else:
#             ret = idxDB_match[0]
#         non_numba_indexes_species_dict[tag_db] = ret
#     return non_numba_indexes_species_dict


# def get_Indexes_of_species_to_variable_position(species_tags):
#     idexes_species = []
#     for j, tag_db in enumerate(tags_db):  # db position -> variable position
#         idxDB_match = [i for i, tag in enumerate(species_tags) if tag == tag_db]
#         if len(idxDB_match) == 0:
#             idexes_species.append(-1)
#         else:
#             idexes_species.append(idxDB_match[0])

#     return idexes_species


def re_organize_species_to_non_variables_at_end(species_tags, known_tags=[]):
    reorganized_tags_to_move_non_vars_to_end = [
        tag for tag in species_tags if tag not in known_tags
    ]
    reorganized_tags_to_move_non_vars_to_end += known_tags
    return reorganized_tags_to_move_non_vars_to_end


def get_charge_of_specie(tag):
    num_plus = len([c for c in tag if c == "+"])
    num_minus = len([c for c in tag if c == "-"])
    if num_plus > 0:
        return +num_plus
    elif num_minus > 0:
        return -num_minus
    else:
        return 0


def create_single_specie(tag):
    z = int(core.act.get_charge_of_specie(tag))
    phase = get_phase_from__species_tag(tag)
    return core.Specie(z, phase, tag)


def get_phase_from__species_tag(tag):
    if "(s)" in tag:
        phase = 2
    elif "(g)" in tag:
        phase = 1
    elif tag == "H2O":
        phase = 3
    else:
        phase = 0
    return phase


def create_list_of_species_engine(species_tag):  # , species_activity_db):
    species = core.List()
    for tag in species_tag:
        sp = create_single_specie(tag)
        species.append(sp)
    return species


def create_Indexes_instance(species_tag, known_specie_size=0):
    # idexes_species = get_Indexes_of_species_to_variable_position(species_tag_conv)

    size = len(species_tag) - known_specie_size
    idx_ctrl = core.IndexController(species_tag, size)
    return idx_ctrl


def create_list_of_reactions_engine(reactions_list, dict_of_indexes_species):
    reactions = core.List()
    for reac in reactions_list:
        if reac["type"] not in ["rev"]:
            continue  # , 'henry']:
        reac_species = get_reactions_species(reac)
        reac_species_List = create_nb_List(reac_species)
        reac_species_indexes = []
        for tag in reac_species:
            if tag in dict_of_indexes_species:
                val = dict_of_indexes_species[tag]
            else:
                # val = np.nan
                val = -1
            reac_species_indexes += [val]

        coefs = np.array([reac[k] for k in reac_species], dtype=np.float64)

        if not reac["log_K25"]:
            logK25 = np.nan
        else:
            logK25 = float(reac["log_K25"])

        if not reac["log_K_coefs"]:
            log_coefs = np.array([np.nan])
        else:
            # logK = a + b*x + c/x +d*log10(x) + e/(x^2) + f*(x^2)
            log_coefs = np.zeros(6)
            log_coefs[0 : len(reac["log_K_coefs"])] = np.array(
                reac["log_K_coefs"]
            )

        if "phase_name" in reac:
            reaction_type = reac["phase_name"]
        elif reac["type"] == "henry":
            reaction_type = "henry"
        else:
            reaction_type = "aqueous"

        try:
            delta_h = float(reac["deltah"])
            delta_h *= 4184.0  # kcal -> J
        except (ValueError, KeyError):
            delta_h = np.nan

        reac_species_indexes = np.array(reac_species_indexes, dtype=np.int64)
        reac_new = core.EqReaction(
            reac_species_indexes,
            coefs,
            logK25,
            log_coefs,
            reaction_type,
            reac_species_List,
            delta_h,
        )
        reactions.append(reac_new)

    return reactions


def create_list_of_mass_balances_engine(
    list_species,
    element_list,
    idx_control,
    feed_list,
    closing_equation_type,
    initial_feed_mass_balance,
    fixed_elements,
):
    """
    Gives the list of mass balances



    Possible equations (besides reactions and charge):
    - Mass balance Cations
    - Mass balance Anions
    - pH
    - For instance: CO2, HCO3-, CO3-- fixing pH should not create any mass balance, but it will in this current logic...
    """
    ele_set = OrderedSet(element_list)

    if not element_list:
        if closing_equation_type in [
            ClosingEquationType.NONE,
            ClosingEquationType.CARBON_TOTAL,
            ClosingEquationType.OPEN,
        ]:  # CARBON_TOTAL STAYS OR NOT?
            # Can perform automatic mass balance detection
            list_elements_in_tags = get_elements_and_their_coefs(feed_list)
            aux_ele_as_list = [
                [sub[0] for sub in item] for item in list_elements_in_tags
            ]
            aux_ele_flat = [sub for item in aux_ele_as_list for sub in item]
            ele_set = OrderedSet(aux_ele_flat)
            ele_set.discard("O")
            ele_set.discard("H")
            # element_list = ele_set #ERROR When Using Al(OH)3 !!! Parenthesis
            logging.info(f"Element mass balances detectec: {element_list}")

    if not ele_set:  # FIXME
        return None

    # [for el in ele_set: if ]

    if initial_feed_mass_balance:
        ini_ele_no_signal = [
            re.sub(RX_NO_SIGNAL, "", tag) for tag in initial_feed_mass_balance
        ]
        [ele_set.discard(item) for item in ini_ele_no_signal]
    if fixed_elements:
        ini_ele_no_signal = [
            re.sub(RX_NO_SIGNAL, "", tag) for tag in fixed_elements
        ]
        [ele_set.discard(item) for item in ini_ele_no_signal]
    if closing_equation_type in [
        ClosingEquationType.CARBON_TOTAL,
        ClosingEquationType.OPEN,
    ]:
        ele_set.discard("C")  # Will be handle differently

    infos_el = []
    mass_balances = []
    for el in ele_set:
        tags_coefs, species_indexes = get_species_indexes_matching_element(
            list_species, el, idx_control.idx
        )

        species_coefs = np.array(list(tags_coefs.values()), dtype=np.float64)
        i_c_feed = get_indexes_feed_for_mass_balance(feed_list, el)
        infos_el += [(species_indexes, species_coefs, i_c_feed)]
        mass_balances += [
            core.MassBalance(species_indexes, species_coefs, i_c_feed, False)
        ]

    return mass_balances


def get_species_indexes_matching_element(list_species, el, idx_species_dict):
    tags_coefs = get_species_tags_with_an_element(idx_species_dict.keys(), el)
    species_indexes = np.array([idx_species_dict[tag] for tag in tags_coefs])
    # species_indexes = get_species_indexes_from_tags_coefs(idx_species_dict, tags_coefs)
    return tags_coefs, species_indexes


def get_species_indexes_from_tags_coefs(dict_of_indexes_species, tags_coefs):
    species_indexes = np.array(
        [
            dict_of_indexes_species[
                replace_strings_for_engine_compatibility(tag)
            ]
            for tag in tags_coefs
        ]
    )
    return species_indexes


def get_reactions_species(reac):
    return [
        k
        for k in reac.keys()
        if utils.check_validity_specie_tag_in_reaction_dict(k)
    ]


def get_elements_and_their_coefs(list_of_species):
    # Removing signals
    tags_no_signals = [
        re.sub(RX_NO_SIGNAL, "", tag) for tag in list_of_species
    ]
    # Clean-up phase name if it has it
    tags_no_signals = [tag.split("__")[0] for tag in tags_no_signals]

    elements_with_coefs = [
        separate_elements_coefs(item) for item in tags_no_signals
    ]

    return elements_with_coefs


def get_species_tags_with_an_element(list_of_species, element):
    elements = list(list_of_species)
    try:
        elements.remove(
            "size"
        )  # FIXME FIXME PUT SIZE IN THE IDX CONTROL CLASS, NOT IN IDX DICT
    except ValueError:
        pass
    elements_with_coefs = tag_list_to_element_coef(elements)

    tags_with_coefs = {}
    for els_coefs, tag in zip(elements_with_coefs, elements):
        for el_coef in els_coefs:
            el, coef = el_coef
            if el == element:
                tags_with_coefs[tag] = int(coef)
    return tags_with_coefs


def get_indexes_feed_for_mass_balance(feeds: list, element: str):
    tags_with_coefs = get_species_tags_with_an_element(feeds, element)
    idxs_in_feed = [
        i for i, feed in enumerate(feeds) if feed in tags_with_coefs
    ]
    feeds_coefs = [float(tags_with_coefs[feeds[idx]]) for idx in idxs_in_feed]
    return_as_list_of_tuple = [
        (i, c) for i, c in zip(idxs_in_feed, feeds_coefs)
    ]
    return return_as_list_of_tuple


def setup_log_gamma(
    reaction_sys,
    T=25 + 298.15,
    c_feed=None,
    setup_func=None,
    db_activities=DEFAULT_DB_FILES["species"],
):

    species_activity_db = utils.load_from_db(db_activities)

    if setup_func is None:
        setup_func = core.act.setup_log_gamma_bdot

    setup_func(reaction_sys, T, species_activity_db, c_feed)

    return


def get_guess_from_ideal_solution(
    reaction_system, args, jac=None, activities_db_file_name=None
):
    x_guess0 = np.full(reaction_system.idx_control.idx["size"], -1e-3)
    _, fsol = solve_equilibrium_existing_activity_model(
        reaction_system,
        x_guess0,
        args,
        jac,
        TypeActivityCalculation.IDEAL,
        activities_db_file_name,
    )
    x_guess = fsol.x
    return x_guess


def solve_equilibrium_existing_activity_model(
    reaction_system,
    x_guess,
    args=None,
    jac=None,
    activity_model_type=TypeActivityCalculation.DEBYE,
    activities_db_file_name=None,
):

    if not activities_db_file_name:
        activities_db_file_name = DEFAULT_DB_FILES["species"]

    if activity_model_type == TypeActivityCalculation.DEBYE:
        setup_log_gamma_func = core.act.setup_log_gamma_bdot
        calc_log_gamma = core.act.calc_log_gamma_dh_bdot
    elif activity_model_type == TypeActivityCalculation.IDEAL:
        setup_log_gamma_func = core.act.setup_log_gamma_ideal
        calc_log_gamma = core.act.calc_log_gamma_ideal
    elif activity_model_type == TypeActivityCalculation.DEBYE_MEAN:
        setup_log_gamma_func = (
            core.act.setup_log_gamma_bdot_mean_activity_neutral
        )
        calc_log_gamma = core.act.calc_log_gamma_dh_bdot_mean_activity_neutral

    setup_log_gamma(
        reaction_system,
        args[1],
        args[0],
        setup_log_gamma_func,
        activities_db_file_name,
    )  # args: TK, c_feed

    args_calc_gamma = (args, calc_log_gamma)

    fsol = solve_with_exception(
        reaction_system.residual, x_guess, args_calc_gamma, jac=jac
    )
    solution = reaction_system.calculate_properties(fsol.success)

    return solution, fsol


def check_polymorphs_in_reaction(reac_solid_precip, dict_saturation_index):
    d_solids = {}
    for i, reac in enumerate(reac_solid_precip):
        tag_full = [tag for tag in reac.species_tags if "(s)" in tag][0]
        formula = tag_full.split("__")[0]
        phase_name = reac.type
        si = dict_saturation_index[phase_name]
        if formula not in d_solids:
            d_solids[formula] = (si, i)
        else:
            si_prev, i_prev = d_solids[formula]
            if si > si_prev:
                d_solids[formula] = (si, i)
    return d_solids


def conv_reaction_engine_to_db_like(reac_solid_precip_no_polymorph):
    reacs_conv_solid_precip = []
    for reac in reac_solid_precip_no_polymorph:
        d = {
            tag.split("__")[0]: coef
            for tag, coef in zip(reac.species_tags, reac.stoic_coefs)
        }
        d["log_K25"] = reac.log_K25
        d["log_K_coefs"] = reac.constant_T_coefs.tolist()
        d["type"] = "rev"
        if reac.type != "aqueous":
            d["phase_name"] = reac.type
        reacs_conv_solid_precip += [d]
    return reacs_conv_solid_precip


def match_indexes_of_a_solution_with_another(
    dict_idx_map_helper,
    dict_indexes_species_conv,
    sys_eq,
    solution_no_precipitation,
    x_guess,
):
    for tag_no_precip, val in dict_idx_map_helper.items():
        if val > -1:
            if dict_indexes_species_conv[tag_no_precip] >= sys_eq.idx.size:
                continue
            x_guess[
                dict_indexes_species_conv[tag_no_precip]
            ] = solution_no_precipitation.c_molal[val]
    x_guess[sys_eq.idx.CaCO3s] = -1e-5


def format_reaction_list_as_latex_mhchem(reaction_list):
    formatted = []
    is_irrev = False
    for reac in reaction_list:
        if "irrev" in reac:
            is_irrev = True
            reac.pop("irrev")
        prods, reags = utils.get_tags_of_prods_reacts(reac)
        s = "\\ce{ "
        for r in reags:
            s += convert_as_latex_specie(reac, r)
        s = s[0:-2]
        s += " <=> " if not is_irrev else "->"
        for p in prods:
            # coefstr = abs(reac[r]) if abs(reac[r]) > 1 else ''
            s += convert_as_latex_specie(reac, p)
            # s += '{}{{{}}} +'.format(coefstr, p)
        s = s[0:-2]
        s += "}"
        formatted.append(s)
    return formatted


def convert_as_latex_specie(reac, r):
    if np.isclose(reac[r] % 1, 0):
        coefstr = abs(int(reac[r])) if abs(int(reac[r])) > 1 else ""
    else:
        coefstr = abs(reac[r]) if abs(reac[r]) > 1 else ""
    tag_no_signal = re.sub(RX_NO_SIGNAL, "", r)
    charge = get_charge_of_specie(r)
    signal = "+" if charge > 0 else "-"
    signal = "" if charge == 0 else signal
    abs_charge = abs(charge)
    charge_str = "" if abs_charge < 2 else str(abs(charge))
    ss = "{}{{{}^{{{}{}}}}} +".format(
        coefstr, tag_no_signal, charge_str, signal
    )
    return ss


def format_reaction_formatted_as_itemize(formatted_reactions):
    s = "\\begin{{itemize}}{}\\end{{itemize}}"
    items = ""
    for reac in formatted_reactions:
        items += "\\item " + reac + "\n"
    return s.format(items)


def get_itemized_latex_from_compounds(initial_comp):
    species, reactions = get_species_reactions_from_compounds(initial_comp)
    reacs_formatted = format_reaction_list_as_latex_mhchem(reactions)
    itemized = format_reaction_formatted_as_itemize(reacs_formatted)
    return itemized


def get_list_of_reactions_latex(
    initial_comp, allow_precipitation=False, database_files=DEFAULT_DB_FILES
):
    reactionsListSolutions = utils.load_from_db(database_files["solutions"])
    # reactionList_ = list(reactionsListAqueous)
    reactionList_ = reactionsListSolutions
    # if allow_precipitation:
    #     reactionList_ += reactionsListPreciptation
    if allow_precipitation:
        reactionsListPhase = utils.load_from_db(database_files["phases"])
        reactionList_ += reactionsListPhase

    species, reactions = get_species_reactions_from_compounds(
        initial_comp, reactionList_
    )
    reacs_formatted = format_reaction_list_as_latex_mhchem(reactions)
    return reacs_formatted


def display_reactions(sys_eq, show_possible_solid=False):
    latex_reactions = format_reaction_list_as_latex_mhchem(
        sys_eq.reactionsStorage
    )
    if show_possible_solid:
        solid_possible_reactions = conv_reaction_engine_to_db_like(
            sys_eq.solid_reactions_but_not_equation
        )
        fill_reactions_with_solid_name_underscore(solid_possible_reactions)
        solid_possible_reactions_latex = format_reaction_list_as_latex_mhchem(
            solid_possible_reactions
        )
        latex_reactions += solid_possible_reactions_latex
    try:
        from IPython.display import display, Math, Latex

        [display(Math(r)) for r in latex_reactions]
    except (ImportError, ModuleNotFoundError):
        # print('Error to display the reactions in latex format: module Ipython not found')
        # raise error
        [print(eq) for eq in latex_reactions]
    return


def ipython_display_reactions(sol: SolutionResult):
    try:
        from IPython.display import display, Math, Latex
    except ImportError as error:
        print(
            "Error to display the reactions in latex format: module Ipython not found"
        )
        raise error
    latex_reactions = format_reaction_list_as_latex_mhchem(sol.reactions)
    [display(Math(r)) for r in latex_reactions]
    return


def fill_reactions_with_solid_name_underscore(solid_possible_reactions):
    for r_solid in solid_possible_reactions:
        prev_keys = list(r_solid.keys())
        for in_element in prev_keys:
            if "phase_name" in r_solid and "(s)" in in_element:
                tag_add = in_element + "__" + r_solid["phase_name"]
                r_solid[tag_add] = r_solid.pop(in_element)


#####################################
# CONVERTING TAG TO ELEMENT VS COEFICIENT
#####################################


def tag_list_to_element_coef(tags_elements):
    return [tag_to_element_coef(tag) for tag in tags_elements]


def tag_to_element_coef(tag):
    tag = re.sub(RX_NO_SIGNAL, "", tag)
    tag = tag.split("__")[0]
    tag_split_hydration = tag.split(":")
    tag = tag_split_hydration[0]
    elements_with_coefs = separate_elements_coefs(tag)
    if len(tag_split_hydration) > 1:
        tag_hydration = ":" + tag_split_hydration[-1]
        elements_with_coefs_hydr = separate_elements_coefs(tag_hydration)
        elements_with_coefs = elements_with_coefs + elements_with_coefs_hydr

    elements = [v for v, c in elements_with_coefs]
    dupls = list_duplicates(elements)
    if len(dupls) > 0:
        dupl_cases = [
            [v_c for v_c in elements_with_coefs if v_c[0] in dupl]
            for dupl in dupls
        ]
        summed_duplicated = [
            reduce(lambda cum, t_c: (t_c[0], cum[1] + t_c[1]), dupl)
            for dupl in dupl_cases
        ]
        elements_coef_comb = [
            v_c for v_c in elements_with_coefs if v_c[0] not in dupls
        ]
        elements_coef_comb += summed_duplicated
        elements_with_coefs = elements_coef_comb
    return elements_with_coefs


def separate_letter_digit_old(el_tag):
    return [
        el_tag[0:-1] if el_tag[-1].isdigit() else el_tag,
        int(el_tag[-1]) if el_tag[-1].isdigit() else 1,
    ]


RX_DIGIT = "[A-Z][^A-Z]*"


def separate_letter_digit(el_tag):
    match_dig = re.match(r"([aA-zZ]+)([0-9]+)", el_tag)
    if match_dig:
        el, dig = match_dig.groups()
    else:
        el, dig = el_tag, 1
    return [el, int(dig)]


#     return [el_tag[0:-1] if el_tag[-1].isdigit() else el_tag, int(el_tag[-1]) if el_tag[-1].isdigit() else 1]


def get_tag_el_coef(tag):
    by_case = re.findall(RX_CASE, tag)
    case_coefs = [separate_letter_digit(e) for e in by_case]
    return case_coefs


def separate_elements_coefs(tag):
    separated_parens = re.split(RX_PRNTHS, tag)
    separated_parens = [val for val in separated_parens if val != ""]
    elements_with_coefs = []
    for v in separated_parens:
        if "(s)" in v or "(g)" in v:
            continue
        if "(" in v and ")" in v:
            intr_prh = v[1:-2]
            d = int(v[-1])
            case_coefs = get_tag_el_coef(intr_prh)
            for el_coef in case_coefs:
                el_coef[1] *= d
        elif ":" in v:
            d_first = int(v[1]) if v[1].isdigit() else 1
            case_coefs = get_tag_el_coef(v[1:])
            for el_coef in case_coefs:
                el_coef[1] *= d_first
        else:
            # by_case = re.findall(RX_CASE, v)
            case_coefs = get_tag_el_coef(v)
        elements_with_coefs += case_coefs
    return elements_with_coefs


def list_duplicates(seq):
    seen = set()
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = set(x for x in seq if x in seen or seen.add(x))
    # turn the set into a list (as requested)
    return list(seen_twice)
