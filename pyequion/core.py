from enum import Enum

import numpy

from . import activity_coefficients as act
from . import properties_utils
from .properties_utils import pCO2_ref, solve_with_exception

global np
np = numpy
# import numba
import os
from collections import namedtuple

import sympy

from . import PengRobinson  # WILL FAIL WITH NUMBA
from . import conductivity
from . import symbolic_computations as mod_sym
from . import utils, utils_for_numba
from .data.reactions_solids import reactions_solids
from .data.reactions_solutions import reactions_solutions

# DATABASE IMPORTED
from .data.species import species
from .utils_for_numba import Dict, List, create_nb_Dict, create_nb_List

DIRNAME = os.path.dirname(__file__)
SOLUTIONS_DB = os.path.join(DIRNAME, "../data/reactions_solutions.json")
SOLIDS_DB = os.path.join(DIRNAME, "../data/reactions_solids.json")
ACTIVITIES_DB = os.path.join(DIRNAME, "../data/species.json")
DEFAULT_DB_FILES = {
    # 'solutions': os.path.abspath(SOLUTIONS_DB),
    # 'phases': os.path.abspath(SOLIDS_DB),
    # 'species': os.path.abspath(ACTIVITIES_DB)
    "solutions": reactions_solutions,
    "phases": reactions_solids,
    "species": species,
}

R = 8.314


class IndexController:
    def __init__(self, species_tags=None, size=0):
        self.idx = {"size": size}
        if not species_tags:
            return
        for i, tag in enumerate(species_tags):
            self.idx[tag] = i

        self.s = {"dummy": 0.0}
        pass


class Specie:
    "A chemical compound (ion, neutral, gas, solid)"

    def __init__(self, z, phase=0, name=""):
        self.z = z
        self.logc = -0.1
        self.logg = -0.1
        self.phase = phase
        self.name = name
        # self.I_factor = I_factor
        # self.dh_a = dh_a
        self.p_int = {"dummy": 0}
        self.p_scalar = {"dummy": 0.0}
        self.p_iarray = {"dummy": np.zeros(1, dtype=np.int64)}
        self.p_array = {"dummy": np.zeros(1)}
        self.p_matrix = {"dummy": np.zeros((1, 1))}
        self.d = {"dummy": {"dummy": np.zeros(1)}}
        pass

    def set_cond_molar(self, cond_molar):
        self.cond_molar = cond_molar

    def logact(self):
        "Calculates log activity"
        if self.phase == 0:
            return self.logc + self.logg
        elif self.phase == 1:
            # gas phase (usualy does not enter here) -> Improve it creating the henry equation type
            # but for now I will just consider for CO2! hardcoded!
            # return np.log10(pCO2_ref)
            # return np.log10(2.0)
            return -1  # FIXME
        elif self.phase == 2:  # Solid
            return 0.0  # activity is 1.0 -> log(a) = 0.0
        elif self.phase == 3:
            return self.logg

    def set_log_gamma(self, val):
        "Set the log of activity coefficient"
        self.logg = val


# --------------------------------------------
# 	REACTIONS DEFINITIONS
# --------------------------------------------


class EqReaction:
    "A Chemical Reaction representation"

    def __init__(
        self,
        idx_species_,
        stoic_coefs,
        log_K25,
        constant_T_coefs,
        type_reac,
        species_tags,
        delta_h,
    ):
        self.idx_species = idx_species_
        self.stoic_coefs = stoic_coefs
        self.log_K25 = log_K25
        self.constant_T_coefs = constant_T_coefs
        self.type = type_reac
        self.species_tags = species_tags
        self.delta_h = delta_h
        pass

    # def eq(self, species, logK_db):
    def eq(self, species, T):
        "Calculates the residual of the equilibrium equation"
        summed = np.sum(
            np.array(
                [
                    -species[self.idx_species[i]].logact()
                    * self.stoic_coefs[i]
                    for i in range(len(self.idx_species))
                ]
            )
        )
        # summed = 0.0
        logK = self.calc_reaction_constant(T)
        if self.type == "henry":
            summed -= logK  # FIXME
            return summed
        summed += logK  # logK_db[self.idx_reaction_db]
        return summed

    def calc_reaction_constant(self, T):
        "Calculates Reaction constant"
        # if np.any(np.isnan(np.array(self.constant_T_coefs))):
        # case 0 - log K25
        # case 1 - poly
        # case 2 - delta_h
        case = 0 if np.isnan(self.delta_h) else 2
        case = case if np.any(np.isnan(self.constant_T_coefs)) else 1

        if case == 0:
            logK = self.log_K25
        elif case == 1:
            t0 = self.constant_T_coefs[0]
            t1 = self.constant_T_coefs[1] * T
            t2 = self.constant_T_coefs[2] / T
            t3 = self.constant_T_coefs[3] * np.log10(T)
            t4 = self.constant_T_coefs[4] / (T ** 2)
            t5 = self.constant_T_coefs[5] * (T ** 2)
            logK = t0 + t1 + t2 + t3 + t4 + t5
        elif case == 2:
            dh = self.delta_h
            R = 8.314
            logK = self.log_K25 - dh / (2.303 * R) * (1 / T - 1 / (298.15))

        return logK


# --------------------------------------------
# 	REACTIONS DEFINITIONS
# --------------------------------------------


class MassBalance:
    "A mass balance representation"

    def __init__(
        self,
        idx_species,
        stoic_coefs,
        idx_feed,
        use_constant_value=False,
        feed_is_unknown=False,
    ):
        """Mass balance structure

        Parameters
        ----------
        idx_species :
            Indexes of species in the mass balance
        stoic_coefs : [type]
            Stoichiometric coefficient of species in the mass balance
        idx_feed : [type]
            Indexes element appears in the feed input
        use_constant_value : bool, optional
            Check if is to use a constant value (from residual argument), by default False
        """
        self.idx_species = idx_species
        self.stoic_coefs = stoic_coefs
        self.idx_feed = idx_feed
        self.use_constant_value = use_constant_value  # to enable DIC or forcing a value not necessary from feed
        self.feed_is_unknown = feed_is_unknown
        pass

    def known_specie_from_feed(self, species, c_feed):
        "Calculation for a compound obtained directly from feed"
        feed_summed = 0.0  # np.array([0.0])
        for i_f, coef in self.idx_feed:  # TypingError FIXME
            feed_summed += c_feed[i_f] * coef
        species[self.idx_species[0]].logc = np.log10(feed_summed)

    def mass_balance_just_summation(self, species):
        "Calculation for the sum of species"
        summed = np.sum(
            np.array(
                [
                    self.stoic_coefs[j]
                    * 10.0 ** (species[self.idx_species[j]].logc)
                    for j in range(len(self.idx_species))
                ]
            )
        )
        return summed

    def mass_balance(self, species, c_feed):
        "Calculation residual: feed - the sum of species"
        summed = np.sum(
            np.array(
                [
                    self.stoic_coefs[j]
                    * 10.0 ** (species[self.idx_species[j]].logc)
                    for j in range(len(self.idx_species))
                ]
            )
        )
        if self.feed_is_unknown:
            mb = 0.0 - summed
        else:
            feed_summed = 0.0  # np.array([0.0])
            for i_f, coef in self.idx_feed:
                feed_summed += c_feed[i_f] * coef
            # feed_summed = 1.0 #np.sum(np.array([c_feed[i_f]*coef for i_f, coef in self.idx_feed]))
            mb = feed_summed - summed
        return mb


DUMMY_MB = MassBalance(np.array([-1]), np.array([-1.0]), [(-1, -1.0)], False)

DUMMY_EqREACTION = EqReaction(
    np.array([-1, -1], dtype=np.int64),
    # np.array([-1]),
    np.array([np.nan], dtype=np.float64),
    -1.0,
    np.array([np.nan], dtype=np.float64),
    "dummy",
    create_nb_List([""]),
    np.nan,
)  # -1 IS DUMMY -> Numba issues

# --------------------------------------------
# 	SOLUTION RESULT
# --------------------------------------------


class SolutionResult:
    """Final equilibrium solution representation

    - c_molal: molal concentration
    - gamma: activity coefficient
    - pH: pH
    - I: Ionic Strength
    - sc: Electrical Conductivity
    - saturation_index: Saturation Index for potential solids
    - preciptation_conc: Precipitation concentration for solids
    - ionic_activity_prod: Ionic Activity Product for potential solids

    - FIXME:
        - Supersaturation?
        - Temperature depencency is not correct

    """

    def __init__(
        self,
        c_molal,
        gamma,
        pH,
        I,
        sc,
        DIC,
        solid_names,
        specie_names,
        saturation_index,
        preciptation_conc,
        ionic_activity_prod,
        log_K_solubility,
        idx,
        reactions,
        index_solubility_calculation,
        x,
        successfull,
    ):
        self.c_molal = c_molal
        self.gamma = gamma
        self.pH = pH
        self.I = I
        self.sc = sc
        self.DIC = DIC
        self.solid_names = solid_names
        self.specie_names = specie_names
        self.saturation_index = saturation_index
        self.preciptation_conc = preciptation_conc
        self.ionic_activity_prod = ionic_activity_prod
        self.log_K_solubility = log_K_solubility
        self.idx = idx
        self.reactions = reactions
        self.index_solubility_calculation = index_solubility_calculation
        if self.index_solubility_calculation > 0:
            self.calculated_solubility = {
                self.specie_names[index_solubility_calculation]: self.c_molal[
                    index_solubility_calculation
                ]
            }

        # Numba does not accept this:
        # self.concentrations = {
        #     k:v for k,v in zip(self.specie_names, self.c_molal)
        # }
        _concentrations = Dict()
        for k, v in zip(self.specie_names, self.c_molal):
            if "(g)" in k:
                continue
            _concentrations[k] = v
        self.concentrations = _concentrations
        self.x = x
        self.successfull = successfull

        pass


class EquilibriumSystem:
    "Equilibrium System - Main class for calculations"

    def __init__(
        self,
        species,
        idx,
        reactions,
        mass_balances,
        mass_balances_known=None,
        dic_idx_coef=None,
        solid_reactions_but_not_equation=None,
        num_of_feeds=None,
        feed_compounds=None,
        closing_equation_type=utils.ClosingEquationType.NONE,
        element_mass_balance=None,
        initial_feed_mass_balance=None,
        fixed_elements=None,
        # database_files=None,
        reactionsStorage=None,
        allow_precipitation=False,
        solid_reactions_in=None,
        # index_solubility_calculation=None,
    ):
        """Constructor EquilibriumSystem

        Parameters
        ----------
        species : list of species

        idx : IdxController
            FIXME: may be simplified
        reactions : list of reactions

        mass_balances : lif of mass balances

        mass_balances_known : list of mass balance from feed, optional
            by default None
        dic_idx_coef : dict, optional
            FIXME REMOVE, by default None
        solid_reactions_but_not_equation : list of reactions, optional
            Possible Solid reactions that are not included in computations, by default None
        num_of_feeds : int, optional
            REVIEW, by default None
        feed_compounds : list, optional
            List of input compounds, by default None
        closing_equation_type : ClosingEquationType, optional
            Set a closing equation, by default utils.ClosingEquationType.NONE
        element_mass_balance : list, optional
            List of elements to form the mass balances, by default None
        initial_feed_mass_balance : list, optional
            List of species that can be optained directily from the mass balance, by default None
        fixed_elements : list, optional
            List of chemical element that can be obtained directily from the mass balance, by default None
        database_files : dict, optional
            Contain the paths to database files, by default None, it is just for storage, NOT USING hence should be deprected is a dummy value
        """
        self.idx_control = idx
        self.species = species
        self.reactions = reactions

        # For linting
        self.ionic_strength = -1
        self.pH = -1
        self.sc = -1
        self.molar_conc = np.empty(0)
        self.gamma = np.empty(0)
        self.mass_balances = mass_balances
        if mass_balances_known[0].idx_species[0] == -1:
            self.is_there_known_mb = False
        else:
            self.is_there_known_mb = True
            self.mass_balances_known = mass_balances_known  # MassBalance(0) MODIFIED CAREFUL - Will save for ease of serialization

        self.dic_idx_coef = dic_idx_coef
        self.solid_reactions_but_not_equation = (
            solid_reactions_but_not_equation
        )
        self.num_of_feeds = num_of_feeds

        self.feed_compounds = feed_compounds
        self.closing_equation_type = closing_equation_type
        self.element_mass_balance = element_mass_balance
        self.initial_feed_mass_balance = initial_feed_mass_balance
        self.fixed_elements = fixed_elements
        # self.database_files = database_files
        self.reactionsStorage = reactionsStorage
        self.index_solubility_calculation = -1
        self.fugacity_calculation = ""  # WILL FAILT IN NUMBA FIXME
        # self.res = np.empty(idx.idx['size'])
        self.allow_precipitation = allow_precipitation
        self.solid_reactions_in = solid_reactions_in
        return

    def set_index_solubility_calculation(self, index_solubility_calculation):
        self.index_solubility_calculation = index_solubility_calculation

    # Use Reaction as a class -> each reaction will have its equilibrium implementation
    def residual(self, x, args, calc_log_gamma):
        """Calculate the residual for the nonlinear system equilibrium

        Parameters
        ----------
        x : float[:]
            Values of log molal
        args : tuple
            Arguments: (c_feed [float[:]], T [float])
        calc_log_gamma : callable
            Function for the nonideality calculation

        Returns
        -------
        float[:]
            Residual for the equilibrium system
        """
        cFeed = args[0]  # self.c_feed, self.TK, self.pCO2
        TK = args[1]  # self.c_feed, self.TK, self.pCO2
        self.TK = (
            TK  # Mutating: saving the last temperature for SolutionResult call
        )
        if self.closing_equation_type == 0:
            pCO2 = args[2]
            logPCO2 = np.log10(pCO2)
        elif self.closing_equation_type == 1:
            carbone_total = args[2]
        elif self.closing_equation_type == 2:
            pH = args[2]
        idx = self.idx_control.idx

        # FIXME: water concentration fixed:
        self.species[idx["H2O"]].logc = 0.0

        for i in range(idx["size"]):
            self.species[i].logc = x[
                i
            ]  # FIXME: engine not mapping && error for engine usage CO2g is not unknown variable, how to remove it?

        # for i in range(self.idx.size):
        #     if not self.species[i].is_known:
        #         self.species[i].logc = x[i]

        # Forced species:
        if self.is_there_known_mb:
            # self.species[idx.Clm].logc = np.log10(2*cFeed[0])
            for mb in self.mass_balances_known:
                if mb.use_constant_value:
                    if mb.idx_feed[0][0] > -1:
                        mb.known_specie_from_feed(self.species, cFeed)
                    continue
                mb.known_specie_from_feed(self.species, cFeed)

        I = self.get_I()

        calc_log_gamma(self.idx_control, self.species, I, TK)

        # FIXME: Remove from residual, put in another method used for initialization
        #  - Temperature will be fixed for a certain Equilibrium
        # logK = calculate_log10_equilibrium_constant(TK)
        # loggammaH20 = loggama_H20(I, TK)

        sp = self.species
        res = np.empty(idx["size"])
        idx_start = 1
        if self.closing_equation_type == 0:
            if self.fugacity_calculation == "pr":  # WILL FAIL WITH NUMBA
                # logfiCO2 = PengRobinson.fugacidade(TK, pCO2)
                # logfCO2 = np.log10(np.exp(logfiCO2)*pCO2)
                iCO2g = self.idx_control.idx["CO2(g)"]
                logfCO2 = self.species[iCO2g].logg
                # GAS FUGACITY IS BEING OBTAINED FROM logg, Consider to modify?
            else:
                logfCO2 = logPCO2
            # deltaVmCO2 = 0.0242675592650
            res[0] = (
                (sp[idx["CO2"]].logact()) - logfCO2 - logK_H(TK)
            )  # - deltaVmCO2*(pCO2*1e5-1e5)/(R*TK*2.302)
        elif self.closing_equation_type == 1:
            # IF WITH DIC, MassBalanceKownList Calculation ALWAYS the first element
            res[0] = carbone_total - self.mass_balances_known[
                0
            ].mass_balance_just_summation(self.species)
        elif self.closing_equation_type == 2:  # pH
            res[0] = self.species[idx["H+"]].logact() + pH
        elif self.closing_equation_type == 3:  # NONE
            idx_start = 0

        res[idx_start : idx_start + len(self.reactions)] = [
            reac.eq(sp, TK) for reac in self.reactions
        ]
        i_prev = idx_start + len(self.reactions)

        if self.mass_balances[0].idx_feed[0][0] != -1:
            for i_mb in range(len(self.mass_balances)):
                res[i_prev + i_mb] = self.mass_balances[i_mb].mass_balance(
                    self.species, cFeed
                )
            i_prev += len(self.mass_balances)
        res[i_prev] = self.charge_conservation()
        return res

    def residual_setup(self, x, args, calc_log_gamma):
        """Calculate the residual for the nonlinear system equilibrium

        Parameters
        ----------
        x : float[:]
            Values of log molal
        args : tuple
            Arguments: (c_feed [float[:]], T [float])
        calc_log_gamma : callable
            Function for the nonideality calculation

        Returns
        -------
        float[:]
            Residual for the equilibrium system
        """
        cFeed = args[0]  # self.c_feed, self.TK, self.pCO2
        TK = args[1]  # self.c_feed, self.TK, self.pCO2
        self.TK = (
            TK  # Mutating: saving the last temperature for SolutionResult call
        )
        # if self.closing_equation_type == 0:
        #     pCO2 = args[2]
        #     logPCO2 = np.log10(pCO2)
        # elif self.closing_equation_type == 1:
        #     carbone_total = args[2]
        # elif self.closing_equation_type == 2:
        #     pH = args[2] # noqa
        idx = self.idx_control.idx

        # FIXME: water concentration fixed:
        self.species[idx["H2O"]].logc = 0.0

        for i in range(idx["size"]):
            self.species[i].logc = x[
                i
            ]  # FIXME: engine not mapping && error for engine usage CO2g is not unknown variable, how to remove it?

        # Forced species:
        if self.is_there_known_mb:
            # self.species[idx.Clm].logc = np.log10(2*cFeed[0])
            for mb in self.mass_balances_known:
                if mb.use_constant_value:
                    if mb.idx_feed[0][0] > -1:
                        mb.known_specie_from_feed(self.species, cFeed)
                    continue
                mb.known_specie_from_feed(self.species, cFeed)

        I = self.get_I()

        calc_log_gamma(self.idx_control, self.species, I, TK)
        pass

    def charge_conservation(self):
        """Residual for charge conservation

        Returns
        -------
        float
        """
        aux = 0.0
        for sp in self.species:
            aux += sp.z * 10.0 ** sp.logc
        return aux

    def get_pH(self):
        """Calculate pH

        Returns
        -------
        float
        """
        return -self.species[self.idx_control.idx["H+"]].logact()

    def get_I(self):
        """Calculate Ionic Strength

        Returns
        -------
        float
        """
        I = 0.0
        for specie in self.species:
            if specie.z == 0:
                continue
            I += 10.0 ** specie.logc * specie.z ** 2
        I *= 0.5
        return I

    def get_gamma(self):
        """Get log gamma

        Returns
        -------
        float[:]
        """
        v = np.empty(len(self.species))
        for i, sp in enumerate(self.species):
            v[i] = 10.0 ** (sp.logg)
        return v

    def get_molal_conc(self):
        """Get molal concentration

        Returns
        -------
        float[:]
        """
        v = np.empty(len(self.species))
        for i, sp in enumerate(self.species):
            v[i] = 10.0 ** (sp.logc)
        return v

    def get_sc(self):
        """Get Conductivivty

        Returns
        -------
        float
        """
        I = self.get_I()
        gm = self.get_gamma()
        c = self.get_molal_conc()
        charges = np.empty(len(self.species))
        for i, sp in enumerate(self.species):
            charges[i] = sp.z
        # charges = np.array([sp.z for sp in self.species])
        cond_zero = np.zeros(len(self.species))
        for i, sp in enumerate(self.species):
            cond_zero[i] = sp.cond_molar
        return conductivity.solution_conductivity(I, gm, c, charges, cond_zero)
        # return -1

    def calc_DIC(self):
        """Get Dissolved Inorganic Carbon concentration content

        Returns
        -------
        float
        """
        sum_ = 0.0
        for i, coef in self.dic_idx_coef:
            if self.species[i].phase == 2 or self.species[i].phase == 1:
                continue
            sum_ += coef * 10.0 ** (self.species[i].logc)
        return sum_

    def get_specie_idx_in_list(self, specie_tag):
        match = [
            i for i, sp in enumerate(self.species) if specie_tag in sp.name
        ]
        if len(match) == 0:
            return -1
        return match[0]

    def calc_SI_and_IAP_precipitation(self):
        TK = (
            self.TK
        )  # 25.0+273.15 #FIXME: EqReaction should be aware of the temperature IMPORTANT FIX

        if self.solid_reactions_but_not_equation[0].type == "dummy":
            return (
                np.array([np.nan]),
                np.array([np.nan]),
                [""],
                np.array([np.nan]),
                np.array([np.nan]),
            )
        # if isinstance(self.species[0].logc, float):
        #     dtype_aux = np.float64
        # else:
        #     dtype_aux = object
        dtype_aux = np.float64
        SI = np.zeros(
            len(self.solid_reactions_but_not_equation), dtype=dtype_aux
        )
        arr_log_Ksp = np.zeros(
            len(self.solid_reactions_but_not_equation), dtype=dtype_aux
        )
        IAP = np.zeros(
            len(self.solid_reactions_but_not_equation), dtype=dtype_aux
        )
        solids_names = [""] * len(self.solid_reactions_but_not_equation)
        precipitation_molalities = np.zeros(
            len(self.solid_reactions_but_not_equation), dtype=dtype_aux
        )
        for i, solid_react in enumerate(self.solid_reactions_but_not_equation):
            # idxs_ions = [ii for ii in solid_react.idx_species if ii >= 0]
            idxs_ions = List()
            for tag, ii in zip(
                solid_react.species_tags, solid_react.idx_species
            ):
                # if not ('(s)' in tag) and (tag != 'H2O'): #FIXME
                check_s_in = "(s)" in tag
                if (
                    not check_s_in
                ):  # and (tag != 'H2O'):  why did I removed water?
                    idxs_ions.append(ii)

            solids_names[i] = solid_react.type
            solid_specie_tag = [
                tag for tag in solid_react.species_tags if "(s)" in tag
            ][0]
            idx_solids = self.get_specie_idx_in_list(solid_specie_tag)
            # solids = [self.species[ii] for ii in idxs_ions if abs(self.species[ii].z) == 0] #FIXME: Possible problem if any non charged specie
            # if len(solids) > 0:
            if (
                idx_solids > -1
            ):  # FIXME Order is not maching correctly the phases -> USE DICT IMPORTANT
                solid = self.species[idx_solids]
                # solids_names[i] = solids[0].name #is only one element anyway
                solid_logc = solid.logc
                precipitation_molalities[i] = 10.0 ** solid_logc

            aqueous_species = [
                self.species[ii]
                for ii in idxs_ions
                if self.species[ii].phase in [0, 3]
            ]  # Including Water? Checkme
            idx_species_in_reaction = [
                solid_react.species_tags.index(sp.name)
                for sp in aqueous_species
            ]
            coefs_in_reaction = [
                solid_react.stoic_coefs[idx] for idx in idx_species_in_reaction
            ]
            log_activities = [sp.logact() for sp in aqueous_species]

            # coef_ref = [abs(self.species[ii].z) for ii in idxs_ions if abs(self.species[ii].z) > 0]
            # coef_ref = coef_ref[0]
            # log_activities = np.array([self.species[ii].logact() for ii in idxs_ions])
            # log_Ksp = logK[solid_react.idx_reaction_db]
            log_Ksp = solid_react.calc_reaction_constant(TK)
            # SI[i] = 1/coef_ref * (np.sum(log_activities) - log_Ksp)
            sum_loga = 0.0
            for loga, coef in zip(log_activities, coefs_in_reaction):
                sum_loga += coef * loga  # Numba bug -> cannot understand
            # sum_logact = [ for loga in log_activities]
            # SI[i] = 1/1.0 * (np.sum(log_activities) - log_Ksp) #FIXME: confirm if SI uses the (.)**(1/eta)
            IAP[i] = 10.0 ** sum_loga
            SI[i] = 1 / 1.0 * (sum_loga - log_Ksp)
            arr_log_Ksp[i] = log_Ksp

            # SI[i] = solid_react.idx_reaction_db

        return SI, IAP, solids_names, precipitation_molalities, arr_log_Ksp

    def calculate_properties(self, successfull=True):
        """Calculate properties of solution

        Returns
        -------
        SolutionResult
        """

        self.molar_conc = self.get_molal_conc()
        self.gamma = self.get_gamma()

        # Reporting only the unknowns (why I'v added this?)
        # i_max = self.idx_control.idx["size"]
        # molal_conc = self.molar_conc[:i_max]
        # molal_conc = self.molar_conc
        # gamma = self.gamma #[:i_max]
        specie_names = List()
        for s in self.species:
            specie_names.append(s.name)
        # specie_names = specie_names[0:i_max]

        self.ionic_strength = self.get_I()
        self.pH = self.get_pH()
        self.sc = self.get_sc()
        dic = self.calc_DIC()
        (
            SI,
            IAP,
            solids_names,
            precipitation_molalities,
            logKsps,
        ) = self.calc_SI_and_IAP_precipitation()

        x = np.array(
            [
                sp.logc
                for i, sp in enumerate(self.species)
                if i < self.idx_control.idx["size"]
            ]
        )

        sat_index = Dict()
        iap = Dict()
        precip_conc = Dict()
        logKsps_dict = Dict()
        for i, name in enumerate(solids_names):
            sat_index[name] = SI[i]
            iap[name] = IAP[i]
            precip_conc[name] = precipitation_molalities[i]
            logKsps_dict[name] = logKsps[i]
        # sat_index = {name: SI[i] for i, name in enumerate(solids_names)}
        # iap = {name: IAP[i] for i, name in enumerate(solids_names)}
        # precip_conc = {name: precipitation_molalities[i] for i, name in enumerate(solids_names)}
        return SolutionResult(
            self.molar_conc,
            self.gamma,
            self.pH,
            self.ionic_strength,
            self.sc,
            dic,
            solids_names,
            specie_names,
            sat_index,
            precip_conc,
            iap,
            logKsps_dict,
            self.idx_control.idx,
            self.reactionsStorage,
            self.index_solubility_calculation,
            x,
            successfull,
        )

    def set_specie_logc(self, idx, val):
        self.species[idx].logc = val

    def numerical_jac(self, x, args):
        J = utils_for_numba.numeric_jacobian(self.residual, x, 1e-8, args)
        return J

    # END EquilibriumSystem


def logK_H(TK):
    logK = (
        108.3865
        + 0.01985076 * TK
        - 6919.53 / TK
        - 40.45154 * np.log10(TK)
        + 669365.0 / (TK ** 2)
    )
    return logK


# --------------------------------------------
# 	Symbolic Generations
# --------------------------------------------
def generate_symbolic_residual(
    sys_eq,
    return_symbols=True,
    setup_log_gamma=None,
    calc_log_gamma=None,
    fixed_temperature=None,
    activities_db_file_name=None,
    activity_model_type=act.TypeActivityCalculation.DEBYE,
):
    """Generate Symbolic Residual

    Parameters
    ----------
    sys_eq : EquilibriumSystem
        [description]
    return_symbols : bool, optional
        Also return symbols or only the residual, by default True
    setup_log_gamma : callable, optional
        Setup loggamma function, by default None
    calc_log_gamma : callable, optional
        Calculate loggamma function, by default None
    fixed_temperature : float, optional
        Fix temperature to remove from symbolic generation, by default None
    activities_db_file_name : str, optional
        Database file name of species, by default DEFAULT_DB_FILES['species']
    activity_model_type : Enumeration, optional
        Use a provided activity model instead of passing setup and calc function for activity, by default act.TypeActivityCalculation.DEBYE

    Returns
    -------
    Symbolic Expression
    """
    if not activities_db_file_name:
        activities_db_file_name = DEFAULT_DB_FILES["species"]
    x = sympy.symbols("x0:{}".format(sys_eq.idx_control.idx["size"]))
    num_args = sys_eq.num_of_feeds + 2  # FIXME: IS ERROR PRONE
    args_symbols = sympy.symbols("args0:{}".format(num_args))
    args = (
        args_symbols[0 : sys_eq.num_of_feeds],
        args_symbols[-2],
        args_symbols[-1],
    )
    mod_sym.prepare_for_sympy_substituting_numpy()

    setup_log_gamma, calc_log_gamma = default_activity_logic(
        activity_model_type
    )
    species_activity_db = utils.load_from_db(activities_db_file_name)
    c_feed, TK = args[0:2]
    if fixed_temperature:
        TK = fixed_temperature
        args = (args_symbols[0 : sys_eq.num_of_feeds], TK, args_symbols[-1])
    setup_log_gamma(sys_eq, TK, species_activity_db, c_feed)

    res = sys_eq.residual(x, args, calc_log_gamma)
    if return_symbols:
        return res, x, args
    return res


#####
#####
def default_activity_logic(
    activity_model_type,
    setup_log_gamma_func=None,
    calc_log_gamma=None,
    fugacity_calculation="ideal",
):
    if setup_log_gamma_func is not None or calc_log_gamma is not None:
        return setup_log_gamma_func, calc_log_gamma
    if activity_model_type == act.TypeActivityCalculation.DEBYE:
        setup_log_gamma_func = act.setup_log_gamma_bdot
        calc_log_gamma = act.calc_log_gamma_dh_bdot
    elif activity_model_type == act.TypeActivityCalculation.IDEAL:
        setup_log_gamma_func = act.setup_log_gamma_ideal
        calc_log_gamma = act.calc_log_gamma_ideal
    elif activity_model_type == act.TypeActivityCalculation.DEBYE_MEAN:
        setup_log_gamma_func = act.setup_log_gamma_bdot_mean_activity_neutral
        calc_log_gamma = act.calc_log_gamma_dh_bdot_mean_activity_neutral
    elif activity_model_type == act.TypeActivityCalculation.PITZER:
        setup_log_gamma_func = act.setup_log_gamma_pitzer
        calc_log_gamma = act.calc_log_gamma_pitzer
    elif activity_model_type == act.TypeActivityCalculation.BROMLEY:
        setup_log_gamma_func = act.setup_bromley_method_Bindividual
        calc_log_gamma = act.calc_bromley_method
    elif activity_model_type == act.TypeActivityCalculation.SIT:
        setup_log_gamma_func = act.setup_SIT_model
        calc_log_gamma = act.calc_sit_method
    if fugacity_calculation == "pr":  # DEFAULTING TO dh-bdot
        # setup_log_gamma_func = act.setup_log_gamma_bdot
        if activity_model_type == act.TypeActivityCalculation.DEBYE:
            calc_log_gamma = act.calc_log_gamma_dh_bdot_with_pengrobinson
        elif activity_model_type == act.TypeActivityCalculation.PITZER:
            calc_log_gamma = act.calc_log_gamma_pitzer_pengrobinson
        else:
            ValueError(
                "Pengrobinson for CO2(g) is only adjusted for PITZER and B-DOT, contact the development team for extension."
            )
    return setup_log_gamma_func, calc_log_gamma
