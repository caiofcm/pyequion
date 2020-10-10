import os

os.environ["NUMBA_DISABLE_JIT"] = "1"
import pytest
from pyequion import reactions_species_builder as rbuilder


def test_formatted_single():
    # initial_comp = ['H2O', 'NaHCO3', 'CaCl2']
    # species, reactions = rbuilder.get_species_reactions_from_compounds(initial_comp)
    reac = [{"HCO3-": -1, "Na+": -1, "NaHCO3": 1, "id_db": 2, "type": "rev"}]
    reacs_formatted = rbuilder.format_reaction_list_as_latex_mhchem(reac)
    assert type(reacs_formatted) == list
    assert r"\ce" in reacs_formatted[0]


def test_formatted_reacs():
    initial_comp = ["H2O", "NaHCO3", "CaCl2"]
    species, reactions = rbuilder.get_species_reactions_from_compounds(
        initial_comp
    )
    reacs_formatted = rbuilder.format_reaction_list_as_latex_mhchem(reactions)
    print(reacs_formatted)
    assert type(reacs_formatted) == list
    assert len(reacs_formatted) == 12
    assert r"\ce" in reacs_formatted[0]


def test_formatted_as_itemize():
    initial_comp = ["H2O", "NaHCO3", "CaCl2"]
    species, reactions = rbuilder.get_species_reactions_from_compounds(
        initial_comp
    )
    reacs_formatted = rbuilder.format_reaction_list_as_latex_mhchem(reactions)
    itemized = rbuilder.format_reaction_formatted_as_itemize(reacs_formatted)
    print(itemized)
    assert type(itemized) == str
    assert "\\begin{itemize}" in itemized


def test_formatted_from_reaction_storage():
    reac_listing = [
        {"H+": 1.0, "H2O": -1.0, "OH-": 1.0},
        {"H+": 1.0, "H2O": -1.0, "Na+": -1.0, "NaOH": 1.0},
        {"Cl-": 1.0, "Na+": 1.0, "NaCl": -1.0, "irrev": 1.0},
    ]
    _ = rbuilder.format_reaction_list_as_latex_mhchem(reac_listing)
    assert True


def test_formatted_as_itemize_function():
    initial_comp = ["H2O", "NaHCO3", "CaCl2"]
    itemized = rbuilder.get_itemized_latex_from_compounds(initial_comp)
    print(itemized)
    assert type(itemized) == str
    assert "\\begin{itemize}" in itemized


if __name__ == "__main__":
    test_formatted_from_reaction_storage()
