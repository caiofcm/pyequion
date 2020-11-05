#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa
__doc__ = """
This sample demonstrate the coupling of aqueous equilibrium equations generated from pyequion
with the simulation using the DAETOOLs framework.

This sample was based on the daetools's  tutorial: `tutorial_adv_2.py`
"""

import sys
import os
from time import localtime, strftime

import numpy as np
from daetools.pyDAE import *
from daetools.solvers.trilinos import pyTrilinos
# from res_nahco3_cacl2_reduced_T_2 import res as res_speciation
from res_nahco3_cacl2_T_2 import res as res_speciation
import pyequion
import aux_create_dynsim
# import daetools_cm

# import pyequion
# Standard variable types are defined in variable_types.py
from pyUnits import J, K, g, kg, kmol, m, mol, s, um

no_small_positive_t = daeVariableType(
    "no_small_positive_t", dimless, 0.0, 1.0, 0.1, 1e-5
)

eq_idx_regular = {
    "size": 14,
    "Na+": 0,
    "HCO3-": 1,
    "Ca++": 2,
    "OH-": 3,
    "H+": 4,
    "CO2": 5,
    "CaOH+": 6,
    "NaOH": 7,
    "NaHCO3": 8,
    "CO3--": 9,
    "CaCO3": 10,
    "NaCO3-": 11,
    "Na2CO3": 12,
    "CaHCO3+": 13,
    "H2O": 14,
    "Cl-": 15,
}
eq_idx_reduced2 = {
    "size": 11,  # REDUCEC
    "Na+": 0,
    "HCO3-": 1,
    "Ca++": 2,
    "OH-": 3,
    "H+": 4,
    "CO2": 5,
    "NaHCO3": 6,
    "CO3--": 7,
    "CaCO3": 8,
    "Na2CO3": 9,  # WRONG, fixing
    "CaHCO3+": 10,
    "H2O": 11,
    "Cl-": 12,
}
# eq_idx = eq_idx_reduced2
eq_idx = eq_idx_regular
iNa = 0
iC = 1
iCa = 2
iCl = 3

sys_eq_aux = aux_create_dynsim.get_caco3_nahco3_equilibrium()

less_factor = 0.1
CONCS = np.array(
    [
        0.5 * less_factor * 150.0e-3,
        0.5 * less_factor * 150.0e-3,
        0.5 * less_factor * 50.0e-3,
        0.5 * less_factor * 2 * 50.0e-3,
    ]
)
comps_aux = {
    "Na+": CONCS[0] * 1e3,
    "HCO3-": CONCS[1] * 1e3,
    "Ca++": CONCS[2] * 1e3,
    "Cl-": CONCS[3] * 1e3,
}
solution_aux = pyequion.solve_solution(comps_aux, sys_eq_aux)
guess_speciation = 10 ** (solution_aux.x)
# guess_speciation = np.array([7.33051995e-02, 6.43325906e-02, 1.82951191e-02, 7.76126264e-07,
#        2.23776785e-08, 1.90732600e-03, 8.07985599e-08, 2.03684279e-08,
#        1.51672652e-03, 3.62336563e-04, 1.36791361e-03, 1.74388571e-04,
#        1.83235834e-06, 5.33688654e-03])

sol_eq_full = pyequion.solve_solution(comps_aux)

rhoc = 2.709997e3  # kg/m3
kv = 1.0
MW_C = 12.0107
MW_Na = 22.98977
MW_Ca = 40.078
MW_CaCO3 = 100.0869
MW_Cl = 35.453
f_cryst_mol = [0, 1, 1, 0]


def calc_B(S):
    # Verdoes 92
    Eb = 12.8
    Ks = 1.4e18
    in_exp = -Eb / ((np.log(S)) ** 2)
    # if in_exp < -300.0:
    #     return 0.0
    B = Ks * S * np.exp(in_exp)
    # B *= (self.V*1e-6) * 60.0 #to #/min

    return B * 1e-3  # : #/m^3/s -> #/kg/s


def calc_G(S):
    # Verdoes 92
    # Kg = 2.4e-12 #m/s
    Kg = 5.6e-10  # ref 22 in Verdoes
    g = 1.8
    G = Kg * (S - 1.0) ** g
    return G


class modelCaCO3Precip(daeModel):
    def __init__(self, Name, Parent=None, Description=""):
        daeModel.__init__(self, Name, Parent, Description)

        self.Nmoments = daeDomain(
            "Nmoments", self, dimless, "Number of Moments"
        )
        self.mus = daeVariable(
            "mus",
            no_t,
            self,
            "Particle Moment concentrations",
            [self.Nmoments],
        )
        self.lmin = daeParameter("lmin", dimless, self, "Minimal size")
        self.G = daeVariable("G", no_t, self, "Growth rate")
        self.B0 = daeVariable("B0", no_t, self, "Nucleation rate")

        self.NSpecies = daeDomain(
            "NSpecies", self, dimless, "Number of Species"
        )
        self.NElementTotal = daeDomain(
            "NElementTotal", self, dimless, "Number of Elements"
        )
        # self.conc_species = daeVariable("conc_species", no_t, self, "Conc Species", [self.NSpecies])
        self.x_species = daeVariable(
            "x_species", no_t, self, "Conc Species", [self.NSpecies]
        )
        self.conc_element_total = daeVariable(
            "conc_element_total",
            no_t,
            self,
            "Conc Element Total",
            [self.NElementTotal],
        )
        # self.iap = daeVariable("iap", no_t, self, "Ionic Activity Product")
        self.S = daeVariable("S", no_t, self, "Supersaturation")
        self.aCapp = daeVariable("aCapp", no_t, self, "Ca++ Activity")
        self.aCO3mm = daeVariable("aCO3mm", no_t, self, "CO3-- Activity")
        self.pH = daeVariable("pH", no_t, self, "pH")
        self.massConcCrystalBulk = daeVariable(
            "massConcCrystalBulk", no_t, self, "massConcCrystalBulk"
        )
        self.L10 = daeVariable("L10", no_t, self, "L10")
        self.TK = daeVariable("TK", no_t, self, "T")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        n_species = self.NSpecies.NumberOfPoints
        n_elements = self.NElementTotal.NumberOfPoints

        cNa = self.conc_element_total(iNa)
        cC = self.conc_element_total(iC)
        cCa = self.conc_element_total(iCa)
        cCl = self.conc_element_total(iCl)
        C = [cNa, cC, cCa, cCl]

        kappa = 1e-6  # Constant(1e-6 * m) #dimensionless auxiliary param
        mu0_ref = 1e20  # dimensionless auxiliary param
        # mu_refs = [mu0_ref*kappa**k for k in range(0,4)]

        "Element Conservation"
        rhoc_mol = rhoc / (MW_CaCO3 * 1e-3)  # kg/m^3 -> mol/m^3
        for j in range(0, n_elements):
            eq = self.CreateEquation("EleConservation({})".format(j), "")
            mu2_dim = self.mus(2) * mu0_ref * kappa ** 2
            G_dim = self.G() * kappa
            eq.Residual = dt(C[j]) - (
                -3.0 * f_cryst_mol[j] * kv * rhoc_mol * mu2_dim * G_dim
            )
            eq.CheckUnitsConsistency = False

        "Moment Balance"

        eq = self.CreateEquation("Moment({})".format(0), "")
        eq.Residual = dt(self.mus(0)) - self.B0()
        eq.CheckUnitsConsistency = False

        for j in range(1, 4):
            eq = self.CreateEquation("Moment({})".format(j), "")
            eq.Residual = dt(self.mus(j)) - (
                j * self.G() * self.mus(j - 1)  # + self.B0()*self.lmin()**j
            )
            eq.CheckUnitsConsistency = False

        "Nucleation Rate"
        # S = 2.0 #TO DO.
        S = self.S()

        eq = self.CreateEquation("NucleationRate", "")
        # eq.Residual = self.B0() - calc_B(S)*1e-8
        B0_calc = calc_B(S)
        eq.Residual = self.B0() - (B0_calc / mu0_ref)

        "Growth Rate"

        eq = self.CreateEquation("GrowthRate", "")
        # eq.Residual = self.G() - calc_G(S)
        eq.Residual = self.G() - (calc_G(S) / kappa)

        # "Temperature"

        # eq = self.CreateEquation("TK", "")
        # eq.Residual = self.TK() - (25.0 + 273.15)

        "Chemical Speciation"

        args_speciation = ([cNa, cC, cCa, cCl], self.TK(), np.nan)

        x = [self.x_species(j) for j in range(n_species)]
        res_species = res_speciation(x, args_speciation)
        for i_sp in range(0, n_species):
            eq = self.CreateEquation("c_species({})".format(i_sp))
            eq.Residual = res_species[i_sp]

        molal_species = [np.log10(self.x_species(j)) for j in range(n_species)]

        comps = {
            "Na+": cNa,
            "HCO3-": cC,
            "Ca++": cCa,
            "Cl-": cCl,
        }
        solution = pyequion.get_solution_from_x(sys_eq_aux, x, comps)

        "pH"
        eq = self.CreateEquation("pH", "")
        eq.Residual = self.pH() - solution.pH

        "Supersaturation"
        aCapp = pyequion.get_activity(solution, "Ca++")
        aCO3mm = pyequion.get_activity(solution, "CO3--")
        Ksp = 10 ** (solution.log_K_solubility["Calcite"])
        iap = solution.ionic_activity_prod["Calcite"]

        eq = self.CreateEquation("aCapp", "")
        eq.Residual = self.aCapp() - aCapp

        eq = self.CreateEquation("aCapp", "")
        eq.Residual = self.aCO3mm() - aCO3mm

        eq = self.CreateEquation("S", "")
        eq.Residual = self.S() - Sqrt(aCapp * aCO3mm / Ksp)

        "Mass of Crystal in Bulk"
        eq = self.CreateEquation("massConcCrystlBulk", "")
        eq.Residual = (
            self.massConcCrystalBulk()
            - (self.mus(3) * kappa ** 3 * mu0_ref) * rhoc * kv
        )

        eq = self.CreateEquation("L10", "")
        eq.Residual = self.L10() - (self.mus(1) * kappa) / (
            self.mus(0) + 1e-20
        )

        "Disturbing Temperature"
        self.IF(Time() < Constant(10 * 600 * s), eventTolerance=1e-5)
        eq = self.CreateEquation("TKIni", "")
        eq.Residual = self.TK() - (25.0 + 273.15)
        self.ELSE()
        eq = self.CreateEquation("TKMod", "")
        eq.Residual = self.TK() - (50.0 + 273.15)
        self.END_IF()

        pass


class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modelCaCO3Precip("modelCaCO3Precip")

    def SetUpParametersAndDomains(self):
        self.m.Nmoments.CreateArray(4)
        self.m.NSpecies.CreateArray(eq_idx["size"])
        self.m.NElementTotal.CreateArray(4)

        self.m.lmin.SetValue(0.0)
        pass

    def SetUpVariables(self):
        nMus = self.m.Nmoments.NumberOfPoints
        self.m.mus.SetInitialConditions(np.zeros(nMus))

        for i_sp in range(0, self.m.NSpecies.NumberOfPoints):
            # self.m.conc_species.SetInitialGuess(i_sp, guess_speciation[i_sp])
            self.m.x_species.SetInitialGuess(i_sp, solution_aux.x[i_sp])

        self.m.conc_element_total.SetInitialConditions(CONCS)

        # self.m.iap.SetInitialGuess(1e-6)
        guess_aCapp = pyequion.get_activity(solution_aux, "Ca++")
        guess_aCO3mm = pyequion.get_activity(solution_aux, "CO3--")
        S = (
            solution_aux.ionic_activity_prod["Calcite"]
            / 10 ** (solution_aux.log_K_solubility["Calcite"])
        ) ** 0.5
        self.m.pH.SetInitialGuess(solution_aux.pH)
        self.m.aCapp.SetInitialGuess(guess_aCapp)
        self.m.aCO3mm.SetInitialGuess(guess_aCO3mm)
        self.m.S.SetInitialGuess(S)
        self.m.TK.SetInitialGuess(25.0 + 273.15)

        pass


def run(**kwargs):
    simulation = simTutorial()
    print(
        "Supported Trilinos solvers: %s"
        % pyTrilinos.daeTrilinosSupportedSolvers()
    )
    # lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Klu", "")
    lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Umfpack", "")
    # lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Lapack", "")
    # lasolver = pyTrilinos.daeCreateTrilinosSolver("'AztecOO_ML", "")
    return daeActivity.simulate(
        simulation,
        reportingInterval=1,
        timeHorizon=20 * 60,
        lasolver=lasolver,
        **kwargs
    )


if __name__ == "__main__":

    guiRun = (
        False if (len(sys.argv) > 1 and sys.argv[1] == "console") else True
    )
    # run(guiRun = guiRun)
    run(guiRun=False)
