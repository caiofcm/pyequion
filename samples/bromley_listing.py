"""
Pyequion example

Using a user provided function for activity coefficient calculation.
"""
# pylint: disable=invalid-name
import os

import numpy as np

import pyequion

os.environ["NUMBA_DISABLE_JIT"] = "1"


bromleyDB = {
    "K+": {"Cl-": 0.024},
    "Na+": {"Cl-": 0.0574},
    "Mg++": {"SO4--": -0.0153},
}


def bromley_model_ion(I, Bi_j, zi, zj, mj):
    "Bromley gamma model"
    TK = 25.0 + 273.15
    A, _ = pyequion.activity_coefficients.debye_huckel_constant(TK)

    e = 4.8029e-10  # erg
    k = 1.38045e-16  # erg
    Na = 6.02214076e23
    d0 = pyequion.properties_utils.density_water(TK)
    D = pyequion.properties_utils.dieletricconstant_water(TK)
    A = (
        1
        / 2.303
        * (e / np.sqrt(D * k * TK)) ** 3
        * np.sqrt(2 * np.pi * d0 * Na / 1000.0)
    )

    sqI = np.sqrt(I)
    zz = np.abs(zi * zj)
    nBij = (0.06 + 0.6 * Bi_j) * zz
    dBij = (1.0 + (1.5 / (zz)) * I) ** 2
    dotBij = nBij / dBij + Bi_j
    Zij = (np.abs(zi) + np.abs(zj)) / 2.0
    Fi = np.sum(dotBij * Zij ** 2 * mj)

    loggi = -A * zi ** 2 * sqI / (1.0 + sqI) + Fi

    return loggi


def setup_bromley_single_electrlyt(reaction_sys, T, db_species, c_feed):
    "Setup bromley single electrolyte"
    anions = [sp for sp in reaction_sys.species if sp.z < 0]
    cations = [sp for sp in reaction_sys.species if sp.z > 0]
    for c in cations:
        for a in anions:
            try:
                c.p_scalar[a.name] = bromleyDB[c.name][a.name]
            except KeyError:
                c.p_scalar[a.name] = 0.0


def calc_bromley_single_electrlyt(idx_ctrl, species, I, T):
    "Calc bromley method for single electrolyte"
    anions = [sp for sp in species if sp.z < 0]
    cations = [sp for sp in species if sp.z > 0]

    for c in cations:
        Bi_j = np.array([c.p_scalar[a.name] for a in anions])
        z_j = np.array([a.z for a in anions])
        mj = np.power(10, np.array([a.logc for a in anions]))
        loggC = bromley_model_ion(I, Bi_j, c.z, z_j, mj)
        c.set_log_gamma(loggC)

    for a in anions:
        Bi_j = np.array([c.p_scalar[a.name] for c in cations])
        z_j = np.array([c.z for c in cations])
        mj = np.power(10, np.array([c.logc for c in cations]))
        loggA = bromley_model_ion(I, Bi_j, a.z, z_j, mj)
        a.set_log_gamma(loggA)


sys_eq = pyequion.create_equilibrium(["KCl"])

solution = pyequion.solve_solution(
    {"KCl": 2e3},
    sys_eq,
    setup_log_gamma_func=setup_bromley_single_electrlyt,
    calc_log_gamma=calc_bromley_single_electrlyt,
)
gamma_mean = pyequion.get_mean_activity_coeff(solution, "KCl")

pyequion.print_solution(solution)
print("gamma_mean", gamma_mean)


sys_eq = pyequion.create_equilibrium(["NaCl"])

solution = pyequion.solve_solution(
    {"NaCl": 2e3},
    sys_eq,
    setup_log_gamma_func=setup_bromley_single_electrlyt,
    calc_log_gamma=calc_bromley_single_electrlyt,
)
gamma_mean = pyequion.get_mean_activity_coeff(solution, "NaCl")

pyequion.print_solution(solution)
print("gamma_mean", gamma_mean)
