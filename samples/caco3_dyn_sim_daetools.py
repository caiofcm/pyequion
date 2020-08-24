#!/usr/bin/env python
# -*- coding: utf-8 -*-
__doc__ = """
This sample demonstrate the coupling of aqueous equilibrium equations generated from pyequion
with the simulation using the DAETOOLs framework.

This sample was based on the daetools's  tutorial: `tutorial_adv_2.py`
"""

import sys
from time import localtime, strftime

import numpy as np
from daetools.pyDAE import *
from daetools.solvers.trilinos import pyTrilinos
from res_nahco3_cacl2_reduced_T_2 import res as res_speciation

# import pyequion
# Standard variable types are defined in variable_types.py
from pyUnits import J, K, g, kg, kmol, m, mol, s, um

eq_idx_regular = {
    'size': 14,
    'Na+': 0,
    'HCO3-': 1,
    'Ca++': 2,
    'OH-': 3,
    'H+': 4,
    'CO2': 5,
    'CaOH+': 6,
    'NaOH': 7,
    'NaHCO3': 8,
    'CO3--': 9,
    'CaCO3': 10,
    'NaCO3-': 11,
    'Na2CO3': 12,
    'CaHCO3+': 13,
    'H2O': 14,
    'Cl-': 15,
}
eq_idx = eq_idx_regular
iNa = 0
iC = 1
iCa = 2
iCl = 3

guess_speciation = np.array([7.33051995e-02, 6.43325906e-02, 1.82951191e-02, 7.76126264e-07,
       2.23776785e-08, 1.90732600e-03, 8.07985599e-08, 2.03684279e-08,
       1.51672652e-03, 3.62336563e-04, 1.36791361e-03, 1.74388571e-04,
       1.83235834e-06, 5.33688654e-03])

rhoc = 2.709997e3 #kg/m3
kv = 1.0
MW_C = 12.0107
MW_Na = 22.98977
MW_Ca = 40.078
MW_CaCO3 = 100.0869
MW_Cl = 35.453
f_cryst_mol = [0,1,1,0]

def calc_B(S):
    # Verdoes 92
    Eb = 12.8
    Ks = 1.4e18
    in_exp = -Eb/(np.log(S))**2
    # if in_exp < -300.0:
    #     return 0.0
    B = Ks * S * np.exp(in_exp)
    # B *= (self.V*1e-6) * 60.0 #to #/min

    # MODIFY JUST FOR TEST
    # B *= 1e-6 #TOO HIGH VALUES...
    return B #* 0.0 #MOD

def calc_G(S):
    # Verdoes 92
    # Kg = 2.4e-12 #m/s
    Kg = 5.6e-10 #ref 22 in Verdoes
    g = 1.8
    G = Kg * (S - 1.0)**g
    return G

class modelCaCO3Precip(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.Nmoments = daeDomain("Nmoments", self, dimless, "Number of Moments")
        self.NSpecies = daeDomain("NSpecies", self, dimless, "Number of Species")
        self.NElementTotal = daeDomain("NElementTotal", self, dimless, "Number of Elements")

        self.mus = daeVariable("mus", no_t, self, "Particle Moment concentrations", [self.Nmoments])
        self.lmin = daeParameter("lmin", dimless, self, "Minimal size")
        self.G = daeVariable("G", no_t, self, "Growth rate")
        self.B0 = daeVariable("B0", no_t, self, "Nucleation rate")
        self.conc_species = daeVariable("conc_species", no_t, self, "Conc Species", [self.NSpecies])
        self.conc_element_total = daeVariable("conc_element_total", no_t, self, "Conc Element Total", [self.NElementTotal])
        self.S = daeVariable("S", no_t, self, "Supersaturation")


    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        n_species = self.NSpecies.NumberOfPoints
        n_elements = self.NElementTotal.NumberOfPoints

        cNa = self.conc_element_total(iNa)
        cC = self.conc_element_total(iC)
        cCa = self.conc_element_total(iCa)
        cCl = self.conc_element_total(iCl)
        C = [cNa, cC, cCa, cCl]

        "Element Conservation"

        for j in range(0, n_elements):
            eq = self.CreateEquation("EleConservation({})".format(j), "")
            eq.Residual = dt(C[j]) - (
                -3.0*f_cryst_mol[j]*kv*rhoc*self.mus(2)*self.G()
            )
            eq.CheckUnitsConsistency = False

        "Moment Balance"

        eq = self.CreateEquation("Moment({})".format(0), "")
        eq.Residual = dt(self.mus(0)) - self.B0()
        eq.CheckUnitsConsistency = False

        for j in range(1, 4):
            eq = self.CreateEquation("Moment({})".format(j), "")
            eq.Residual = dt(self.mus(j)) - (
                j*self.G()*self.mus(j-1) + self.B0()*self.lmin()**j
            )
            eq.CheckUnitsConsistency = False

        "Nucleation Rate"

        eq = self.CreateEquation("NucleationRate", "")
        eq.Residual = self.B0() - calc_B(S)

        "Growth Rate"

        eq = self.CreateEquation("GrowthRate", "")
        eq.Residual = self.G() - calc_G(S)

        "Chemical Speciation"

        x_speciation = [
            np.log10(self.conc_species(j))
            for j in range(n_species)
        ]
        T = 25.0 + 273.15 #K
        args_speciation = ([cNa, cC, cCa, cCl], T, np.nan)

        res_species = res_speciation(x_speciation, args_speciation)
        for i_sp in range(0, n_species):
            eq = self.CreateEquation("c_species({})".format(i_sp))
            eq.Residual = res_species[i_sp] #*dae.Constant(1*mol/kg)

        "Supersaturation"

        S = 2.0 #TO DO.

        pass

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modelCaCO3Precip("modelCaCO3Precip")

    def SetUpParametersAndDomains(self):
        self.m.Nmoments.CreateArray(4)
        self.m.NSpecies.CreateArray(eq_idx['size'])
        self.m.NElementTotal.CreateArray(4)

        self.m.lmin.SetValue(0.0)
        pass

    def SetUpVariables(self):
        nMus = self.m.Nmoments.NumberOfPoints
        self.m.mus.SetInitialConditions(np.zeros(nMus))

        for i_sp in range(0, self.m.NSpecies.NumberOfPoints):
            self.m.conc_species.SetInitialGuess(i_sp, guess_speciation[i_sp])

        self.m.conc_element_total.SetInitialConditions(np.array([
            0.5*150.0e-3,
            0.5*150.0e-3,
            0.5*50.0e-3,
            0.5*2*50.0e-3,
        ]))

        pass

def run(**kwargs):
    simulation = simTutorial()
    print('Supported Trilinos solvers: %s' % pyTrilinos.daeTrilinosSupportedSolvers())
    lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Klu", "")
    return daeActivity.simulate(simulation, reportingInterval = 600,
                                            timeHorizon       = 10*3600,
                                            lasolver          = lasolver,
                                            **kwargs)

if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    # run(guiRun = guiRun)
    run(guiRun = False)
