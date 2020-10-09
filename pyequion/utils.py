import commentjson as json
import numpy as np
import json as json_base
import copy

class ClosingEquationType():
    """Type of closing equation

    - OPEN: Open system - equilibrium with CO2
    - CARBON_TOTAL: Special case of a mass balance with element C: allow the setting of total carbone (different from the input compounds)
    - PH: fix the pH
    - NONE: Let the package provide all the equations (obatined from the input file)
    """
    OPEN = 0
    CARBON_TOTAL = 1
    PH = 2
    NONE = 3

def get_dissociating_ions(tag, reaction_list):

    for reac in reaction_list:
        if tag in reac.species_tags:
            prod, reacs = get_tags_prod_reacts_from_eq_reaction(reac)
            if tag in prod[0] and len(prod) == 1:
                return reacs
            if tag in reacs[0] and len(reacs) == 1:
                return prod
    return None

def get_dissociating_ions_plain_reactions(tag, reaction_list):

    for reac in reaction_list:
        if tag in reac:
            prod, reacs = get_tags_prod_reacts_plain_reaction(reac)
            if tag in prod[0] and len(prod) == 1:
                return reacs
            if tag in reacs[0] and len(reacs) == 1:
                return prod
    return None

def get_tags_of_prods_reacts(r: dict):
    """Get Reaction Species tags for products and reactions
    """
    prods = [k for k, v in r.items() if check_validity_specie_tag_in_reaction_dict(k) if v > 0]
    reacs = [k for k, v in r.items() if check_validity_specie_tag_in_reaction_dict(k) if v < 0]
    return prods, reacs

def get_tags_prod_reacts_from_eq_reaction(eq_reaction):
    """Get Reaction Species tags for products and reactions from Reaction Engine
    """
    prods_t_s = [(tag, stoic) for tag, stoic in zip(eq_reaction.species_tags, eq_reaction.stoic_coefs)
        if stoic > 0]
    reacs_t_s = [(tag, stoic) for tag, stoic in zip(eq_reaction.species_tags, eq_reaction.stoic_coefs)
        if stoic < 0]

    return prods_t_s, reacs_t_s

def get_tags_prod_reacts_plain_reaction(eq_reaction):
    """Get Reaction Species tags for products and reactions from Reaction Engine
    """
    prods_t_s = [(tag, stoic) for tag, stoic in eq_reaction.items()
        if stoic > 0]
    reacs_t_s = [(tag, stoic) for tag, stoic in eq_reaction.items()
        if stoic < 0]

    return prods_t_s, reacs_t_s

def check_validity_specie_tag_in_reaction_dict(k):
    """Validate key in database reaction entry
    """
    return (not (k == 'type' or \
        k == 'id_db' or \
        k == 'log_K25' or \
        k == 'log_K_coefs' or \
        k == 'deltah' or \
        k == 'phase_name'))

def load_from_db(fname):
    if not isinstance(fname, str):
        return copy.deepcopy(fname)
    "Open a json file as dict"
    with open(fname, 'r') as json_file:
        db = json.load(json_file)
    return db

