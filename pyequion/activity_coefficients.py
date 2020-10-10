# flake8: noqa #FIXME

from pyequion import properties_utils
from . import utils
from .utils_for_numba import List, Dict
from pyequion import pitzer
from .wateractivity import activitywater
import enum
import numpy

global np
np = numpy
from . import PengRobinson  # TODO Implement numbafied version


class TypeActivityCalculation(
    enum.Enum
):  # TODO: add enum and fix numba enum.Enum
    IDEAL = "IDEAL"
    DEBYE = "DEBYE"
    DEBYE_MEAN = "DEBYE_MEAN"
    PITZER = "PITZER"
    BROMLEY = "BROMLEY"
    SIT = "SIT"


################################
# Activity Models
################################

# -------------------------------
# Ideal Case
# -------------------------------


def setup_log_gamma_ideal(reaction_sys, T, db_species, c_feed=None):
    "Setup activity calculation for ideal model"
    return


# @numba.njit
def calc_log_gamma_ideal(idx_ctrl, species, I, T):
    "Activity Coefifcient calculation for ideal model"
    [sp.set_log_gamma(0.0) for sp in species]
    pass


# -------------------------------
# Extended Debye-Huchel with Davies and Constant Fallback
# -------------------------------


def setup_log_gamma_bdot(reaction_sys, T, db_species, c_feed=None):
    """Setup activity calculation for:

    Extended Debye Huckel (B-dot) when there is those parameters

    Fallback to:

    - Davies for charged species
    - :math:`\\log{\\gamma} = 0.1I` for neutral species

    Parameters
    ----------
    reaction_sys : EquilibriumSystem

    T : Temperature
        [description]
    db_species : dict
        Database of species with their activities
    c_feed : float[:]
        Input concentrations
    """
    db_species = db_species["debye"]
    # A, B = debye_huckel_constant(T)
    # idx_cntrl = reaction_sys.idx_control
    # idx_cntrl.s['A'] =  A
    # idx_cntrl.s['B'] =  B #I will not assume constant T for the setup, but is an option
    for sp in reaction_sys.species:
        I_factor, dh_a, dh_b = species_definition_dh_model(sp.name, db_species)
        sp.p_scalar["I_factor"] = I_factor
        sp.p_scalar["dh_a"] = dh_a
        sp.p_scalar["dh_b"] = dh_b
    return


# @numba.njit
def calc_log_gamma_dh_bdot(
    idx_ctrl, species, I: float, T: float
):  # FIXME: idx_ctrl to reaction_sys ?
    """Calculation as in PHREEQC

    Extended Debye Huckel (B-dot) when there is those parameters

    Fallback to:

    - Davies for charged species
    - :math:`\\log{\\gamma} = 0.1I` for neutral species

    Parameters
    ----------
    idx_ctrl : FIXME
        Auxilar structure for getting index of species
    species : list
        List of species
    I : float
        Ionic Strength
    T : float
        Temperature
    """

    # A = idx_ctrl.s['A']
    # B = idx_ctrl.s['B']
    A, B = debye_huckel_constant(T)
    for sp in species:
        if np.isfinite(sp.p_scalar["dh_a"]):
            # Debye-Huckel Modified
            a = sp.p_scalar["dh_a"]
            b = sp.p_scalar["dh_b"]
            logg = (
                -A * sp.z ** 2 * np.sqrt(I) / (1 + B * a * np.sqrt(I)) + b * I
            )
        elif np.isfinite(sp.p_scalar["I_factor"]):
            logg = sp.p_scalar["I_factor"] * I
        else:
            # Davies
            logg = log10gamma_davies(I, sp.z, A)
        sp.set_log_gamma(logg)
    pass


# @numba.njit WITH PR
def calc_log_gamma_dh_bdot_with_pengrobinson(
    idx_ctrl, species, I: float, T: float
):  # FIXME: idx_ctrl to reaction_sys ?
    """Calculation B-Dot with **PengRobinson** for CO2

    CANNOT BE NUMBAFIED FIXME

    Extended Debye Huckel (B-dot) when there is those parameters

    Fallback to:

    - Davies for charged species
    - :math:`\\log{\\gamma} = 0.1I` for neutral species

    Parameters
    ----------
    idx_ctrl : FIXME
        Auxilar structure for getting index of species
    species : list
        List of species
    I : float
        Ionic Strength
    T : float
        Temperature
    """

    # A = idx_ctrl.s['A']
    # B = idx_ctrl.s['B']
    A, B = debye_huckel_constant(T)
    for sp in species:
        if "(g)" in sp.name:  # gas phase
            pCO2 = sp.p_scalar["P"]
            logfiCO2 = PengRobinson.fugacidade(T, pCO2)
            logfCO2 = np.log10(np.exp(logfiCO2) * pCO2)
            logg = logfCO2  # Is the log fugacity for gas...
        elif np.isfinite(sp.p_scalar["dh_a"]):
            # Debye-Huckel Modified
            a = sp.p_scalar["dh_a"]
            b = sp.p_scalar["dh_b"]
            logg = (
                -A * sp.z ** 2 * np.sqrt(I) / (1 + B * a * np.sqrt(I)) + b * I
            )
        elif np.isfinite(sp.p_scalar["I_factor"]):
            logg = sp.p_scalar["I_factor"] * I
        else:
            # Davies
            logg = log10gamma_davies(I, sp.z, A)
        sp.set_log_gamma(logg)
    pass


def b_dot_equation(I, A, B, a, b, z):
    logg = -A * z ** 2 * np.sqrt(I) / (1 + B * a * np.sqrt(I)) + b * I
    return logg


# -------------------------------
# Extended Debye-Huchel with Mean Coefficient for Neutral Dissociating Species
# -------------------------------


def setup_log_gamma_bdot_mean_activity_neutral(
    reaction_sys, T, db_species, c_feed=None
):
    """Setup activity coef. calculation as in PHREEQC but with mean coefficient for neutral dissociating

    Extended Debye Huckel (B-dot) using mean coefficient for neutral species

    When missing parameters:

    Fallback to:

    - Davies for charged species
    - :math:`\\log{\\gamma} = 0.1I` for neutral species

    Parameters
    ----------
    reaction_sys : EquilibriumSystem

    T : Temperature
        [description]
    db_species : dict
        Database of species with their activities
    c_feed : float[:]
        Input concentrations
    """
    db_species = db_species["debye"]
    A, B = debye_huckel_constant(T)
    idx_cntrl = reaction_sys.idx_control
    idx_cntrl.s["A"] = A
    idx_cntrl.s["B"] = B
    # def logic_dh_fallback(sp, db_species):
    #     I_factor, dh_a, dh_b = species_definition_dh_model(sp.name, db_species)
    #     sp.p_scalar['I_factor'] = I_factor
    #     sp.p_scalar['dh_a'] = dh_a
    #     sp.p_scalar['dh_b'] = dh_b
    #     return

    for sp in reaction_sys.species:
        if sp.z != 0:
            logic_dh_fallback(sp, db_species)
            sp.p_int["use_mean_act"] = False
        else:
            reac_match = utils.get_dissociating_ions(
                sp.name, reaction_sys.reactions
            )
            if reac_match is None:
                logic_dh_fallback(sp, db_species)
                sp.p_int["use_mean_act"] = False
            else:
                tags = [item[0] for item in reac_match]
                stoics = [item[1] for item in reac_match]
                # idxs = [idx_cntrl.idx[tag] for tag in tags]
                # for tag, stoic in zip(tags, stoics):
                #     sp.p_scalar[tag] = stoic
                sp.d["mean_coefs"] = {
                    tag: stoic for tag, stoic in zip(tags, stoics)
                }
                sp.p_int["use_mean_act"] = True
    return


# @numba.njit
def calc_log_gamma_dh_bdot_mean_activity_neutral(
    idx_ctrl, species, I, T
):  # FIXME: idx_ctrl to reaction_sys ?
    r"""Calculation as in PHREEQC but with mean coefficient for neutral dissociating

    Extended Debye Huckel (B-dot) using mean coefficient for neutral species

    When missing parameters:

    - Fallback to:
        - Davies for charged species
        - :math:`\log{\gamma} = 0.1I` for neutral species

    Parameters
    ----------
    idx_ctrl : FIXME
        Auxilar structure for getting index of species
    species : list
        List of species
    I : float
        Ionic Strength
    T : float
        Temperature
    """
    A = idx_ctrl.s["A"]
    B = idx_ctrl.s["B"]
    for sp in species:
        if sp.p_int["use_mean_act"]:
            continue
        if np.isfinite(sp.p_scalar["dh_a"]):
            # Debye-Huckel Modified
            a = sp.p_scalar["dh_a"]
            b = sp.p_scalar["dh_b"]
            logg = (
                -A * sp.z ** 2 * np.sqrt(I) / (1 + B * a * np.sqrt(I)) + b * I
            )
        elif np.isfinite(sp.p_scalar["I_factor"]):
            logg = sp.p_scalar["I_factor"] * I
        else:
            # Davies
            logg = log10gamma_davies(I, sp.z, A)
        sp.set_log_gamma(logg)

    for sp in species:
        if not sp.p_int["use_mean_act"]:
            continue
        aux = 0.0
        stoic_sum = 0.0
        for tag, stoic in sp.d["mean_coefs"].items():
            sp_ion = species[idx_ctrl.idx[tag]]
            aux += stoic * sp_ion.logg
            stoic_sum += stoic
        log_g_mean = aux / stoic_sum
        sp.set_log_gamma(log_g_mean)
    pass


# -------------------------------
# Pitzer Model
# -------------------------------


def setup_log_gamma_pitzer(reaction_sys, T, db_species, c_feed=None):
    """Setup activity calculation for Pitzer Method:

    Parameters
    ----------
    reaction_sys : EquilibriumSystem

    T : Temperature
        [description]
    db_species : dict
        Database of species with their activities
    c_feed : float[:]
        Input concentrations
    """
    species = reaction_sys.species
    db_species = db_species["pitzer"]
    neutrals = db_species["neutrals"]
    try:
        i_neutrals = [
            [i for i, sp in enumerate(species) if sp.name == n][0]
            for n in neutrals
        ]
        n_n = len(i_neutrals)
    except IndexError:
        i_neutrals = []  # -9999 #?
        n_n = 0

    cations = [sp for sp in species if sp.z > 0]
    n_c = len(cations)
    anions = [sp for sp in species if sp.z < 0]
    n_a = len(anions)

    # Neutrals Matrix
    Lambda_c = np.zeros((n_n, n_c))
    Lambda_a = np.zeros((n_n, n_a))
    for i_n in range(n_n):
        neutral = neutrals[i_n]  # CHECKME
        vals = db_species[neutral]
        for i_c, sp_c in enumerate(cations):
            try:
                Lambda_c[i_n, i_c] = vals["n-c"][sp_c.name]
            except KeyError:
                continue
        for i_a, sp_a in enumerate(anions):
            try:
                Lambda_a[i_n, i_a] = vals["n-a"][sp_a.name]
            except KeyError:
                continue

    # Binary Cations Iteractions
    Theta_c = np.zeros((n_c, n_c))
    Theta_a = np.zeros((n_a, n_a))
    for i, sp_i in enumerate(
        cations
    ):  # FIXME: symmetric matrix, take advantage of it and the db can have less information
        try:
            vals_db = db_species[sp_i.name]
        except KeyError:
            continue
        # vals_db = db_species[sp_i.name]
        for j, sp_j in enumerate(cations):
            try:
                Theta_c[i, j] = vals_db["c-c"][sp_j.name]
            except KeyError:
                continue

    for i, sp_i in enumerate(anions):
        try:
            vals_db = db_species[sp_i.name]
        except KeyError:
            # Theta_a[i, :] = 0.0
            continue
        for j, sp_j in enumerate(anions):
            try:
                Theta_a[i, j] = vals_db["a-a"][sp_j.name]
            except KeyError:
                continue
        pass

    # Cation-Anion Interactions
    Cf = np.zeros((n_c, n_a))
    for i_c, sp_c in enumerate(cations):
        try:
            db_c = db_species[sp_c.name]["c-a"]
        except KeyError:
            # Cf[i_c, :] = 0.0
            continue
        for i_a, sp_a in enumerate(anions):
            try:
                Cf[i_c, i_a] = db_c[sp_a.name]
            except KeyError:
                continue

    # Beta Interactions
    Beta = np.zeros((n_c * n_a, 3))
    for i_c, sp_c in enumerate(cations):
        try:
            db_c = db_species[sp_c.name]["beta"]
        except KeyError:
            # Beta[i_c*n_a:i_c*n_a + n_a, :] = 0.0
            continue
        for i_a, sp_a in enumerate(anions):
            try:
                beta_params = db_c[sp_a.name]
                Beta[(i_c * n_a) + i_a, :] = beta_params
            except KeyError:
                continue

    # Mean Activity species:
    for sp in reaction_sys.species:
        if sp.z != 0:
            sp.p_int["use_mean_act"] = False
        else:
            reac_match = utils.get_dissociating_ions(
                sp.name, reaction_sys.reactions
            )
            if reac_match is None:
                sp.p_int["use_mean_act"] = False
            else:
                tags = [item[0] for item in reac_match]
                stoics = [item[1] for item in reac_match]
                # sp.d['mean_coefs'] = Dict()
                # {'n': {'i': np.empty((1))}}
                d_aux = Dict()
                for tag, stoic in zip(tags, stoics):
                    d_aux[tag] = np.array([stoic])
                sp.d["mean_coefs"] = d_aux
                sp.p_int["use_mean_act"] = True

    # I will make in future idx_controller as simple Dict: string-int
    # thus, I will use the zero sp to store systemwide parameters -> FIXME ?
    sp0 = species[0]
    sp0.p_matrix["Lambda_c"] = Lambda_c
    sp0.p_matrix["Lambda_a"] = Lambda_a
    sp0.p_matrix["Theta_c"] = Theta_c
    sp0.p_matrix["Theta_a"] = Theta_a
    sp0.p_matrix["Cf"] = Cf
    sp0.p_matrix["Beta"] = Beta
    sp0.p_iarray["z_c"] = np.array([c.z for c in cations])
    sp0.p_iarray["z_a"] = np.array([a.z for a in anions])
    sp0.p_iarray["i_neutrals"] = np.array(i_neutrals, dtype=np.int64)

    return


# fmt: off

# # @numba.njit
def calc_log_gamma_pitzer(idx_ctrl, species, I, T): #FIXME: idx_ctrl to reaction_sys ?
    """ Activity Coef. Calculation with Pitzer

    - H2O as log mean
    - Non-neutral species such as CaOH+, CaHCO3+ are not treated from mean approach, but from the pitzer log cation/anion


    Parameters
    ----------
    idx_ctrl : FIXME
        Auxilar structure for getting index of species
    species : list
        List of species
    I : float
        Ionic Strength
    T : float
        Temperature
    """
    sp0 = species[0]
    i_neutrals = sp0.p_iarray['i_neutrals']
    cations = [sp for sp in species if sp.z > 0]
    anions = [sp for sp in species if sp.z < 0]
    m_c = np.array([sp.logc for sp in cations])
    m_c = 10**m_c
    m_a = np.array([sp.logc for sp in anions])
    m_a = 10**m_a
    neutrals = [species[i] for i in i_neutrals]
    m_n = np.array([sp.logc for sp in neutrals])
    m_n = 10**m_n
    z_c = sp0.p_iarray['z_c']
    z_a = sp0.p_iarray['z_a']
    Lambda_c = sp0.p_matrix['Lambda_c']
    Lambda_a = sp0.p_matrix['Lambda_a']
    Theta_c = sp0.p_matrix['Theta_c']
    Theta_a = sp0.p_matrix['Theta_a']
    Cf = sp0.p_matrix['Cf']
    Beta = sp0.p_matrix['Beta']

    skip_tern = True
    anan = np.array([np.nan])
    a2nan = np.array([[np.nan], [np.nan]])
    for i, sp in enumerate(cations):
        logg = pitzer.loggammam(m_c, m_a, m_n, z_c, z_a, Beta, Cf,
            Theta_c, Theta_a, anan, anan, Lambda_c, T, i, skip_tern)/np.log(10)
        sp.set_log_gamma(logg)

    for i, sp in enumerate(anions):
        logg = pitzer.loggammax(m_c, m_a, m_n, z_c, z_a, Beta, Cf,
            Theta_c, Theta_a, anan, anan, Lambda_a, T, i, skip_tern)/np.log(10)
        sp.set_log_gamma(logg)

    for i, sp in enumerate(neutrals): #FIXME: epnca-ternary-remove; why m_n not used?
        logg = pitzer.logneutro(m_c, m_a, Lambda_c, Lambda_a, a2nan, i, skip_tern)/np.log(10)
        sp.set_log_gamma(logg)

    # Mean Activity species:
    for sp in species:
        if not sp.p_int['use_mean_act']:
            continue
        aux = 0.0
        stoic_sum = 0.0
        for tag, stoic in sp.d['mean_coefs'].items():
            sp_ion = species[idx_ctrl.idx[tag]]
            aux += stoic[0]*sp_ion.logg
            stoic_sum += stoic[0]
        log_g_mean = aux / stoic_sum
        sp.set_log_gamma(log_g_mean)

    spH2O = species[idx_ctrl.idx['H2O']]
    # sum_molal = np.array([10**sp.logc for sp in species if sp.name != 'H2O']).sum()
    ln_ac_water = pitzer.activitywater(m_c, m_a, m_n, z_c, z_a, Beta, Cf, Theta_c, Theta_a, Lambda_c, Lambda_a, T)
    logac_water = ln_ac_water/np.log(10)
    spH2O.set_log_gamma(logac_water)

    pass

# ## improve this!!!
# ## equation of state inclusion is not good, get a better way

def calc_log_gamma_pitzer_pengrobinson(idx_ctrl, species, I, T): #FIXME: idx_ctrl to reaction_sys ?
    """ Activity Coef. Calculation with Pitzer

    - H2O as log mean
    - Non-neutral species such as CaOH+, CaHCO3+ are not treated from mean approach, but from the pitzer log cation/anion


    Parameters
    ----------
    idx_ctrl : FIXME
        Auxilar structure for getting index of species
    species : list
        List of species
    I : float
        Ionic Strength
    T : float
        Temperature
    """
    sp0 = species[0]
    i_neutrals = sp0.p_iarray['i_neutrals']
    cations = [sp for sp in species if sp.z > 0]
    anions = [sp for sp in species if sp.z < 0]
    m_c = np.array([sp.logc for sp in cations])
    m_c = 10**m_c
    m_a = np.array([sp.logc for sp in anions])
    m_a = 10**m_a
    neutrals = [species[i] for i in i_neutrals]
    m_n = np.array([sp.logc for sp in neutrals])
    m_n = 10**m_n
    z_c = sp0.p_iarray['z_c']
    z_a = sp0.p_iarray['z_a']
    Lambda_c = sp0.p_matrix['Lambda_c']
    Lambda_a = sp0.p_matrix['Lambda_a']
    Theta_c = sp0.p_matrix['Theta_c']
    Theta_a = sp0.p_matrix['Theta_a']
    Cf = sp0.p_matrix['Cf']
    Beta = sp0.p_matrix['Beta']

    skip_tern = True
    anan = np.array([np.nan])
    a2nan = np.array([[np.nan], [np.nan]])
    for i, sp in enumerate(cations):
        logg = pitzer.loggammam(m_c, m_a, m_n, z_c, z_a, Beta, Cf,
            Theta_c, Theta_a, anan, anan, Lambda_c, T, i, skip_tern)/np.log(10)
        sp.set_log_gamma(logg)

    for i, sp in enumerate(anions):
        logg = pitzer.loggammax(m_c, m_a, m_n, z_c, z_a, Beta, Cf,
            Theta_c, Theta_a, anan, anan, Lambda_a, T, i, skip_tern)/np.log(10)
        sp.set_log_gamma(logg)

    for i, sp in enumerate(neutrals): #FIXME: epnca-ternary-remove; why m_n not used?
        logg = pitzer.logneutro(m_c, m_a, Lambda_c, Lambda_a, a2nan, i, skip_tern)/np.log(10)
        sp.set_log_gamma(logg)

    # Mean Activity species:
    for sp in species:
        if not sp.p_int['use_mean_act']:
            continue
        aux = 0.0
        stoic_sum = 0.0
        for tag, stoic in sp.d['mean_coefs'].items():
            sp_ion = species[idx_ctrl.idx[tag]]
            aux += stoic[0]*sp_ion.logg
            stoic_sum += stoic[0]
        log_g_mean = aux / stoic_sum
        sp.set_log_gamma(log_g_mean)

    for sp in species:
        if '(g)' in sp.name: #gas phase
            pCO2 = sp.p_scalar['P']
            logfiCO2 = PengRobinson.fugacidade(T, pCO2)
            logfCO2 = np.log10(np.exp(logfiCO2)*pCO2)
            logg = logfCO2 #Is the log fugacity for gas...
            sp.set_log_gamma(logg)

    spH2O = species[idx_ctrl.idx['H2O']]
    # sum_molal = np.array([10**sp.logc for sp in species if sp.name != 'H2O']).sum()
    ln_ac_water = pitzer.activitywater(
        m_c,
        m_a,
        m_n,
        z_c,
        z_a,
        Beta,
        Cf,
        Theta_c,
        Theta_a,
        Lambda_c,
        Lambda_a,
        T
    )
    logac_water = ln_ac_water/np.log(10)
    spH2O.set_log_gamma(logac_water)

    pass

# fmt: on

# -------------------------------
# Bromley Model
# -------------------------------


def setup_bromley_method_Binteration(reaction_sys, T, db_species, c_feed=None):
    """
    From interation B12 - Halted
    """
    anions = [sp for sp in reaction_sys.species if sp.z < 0]
    cations = [sp for sp in reaction_sys.species if sp.z > 0]
    for c in cations:
        for a in anions:
            try:
                c.p_scalar[a.name] = db_species["bromley"][c.name][a.name]
            except KeyError:
                c.p_scalar[a.name] = 0.0
    return


def setup_bromley_method_Bindividual(reaction_sys, T, db_species, c_feed=None):
    """
    From individual ions B values
    """
    db = db_species["bromley-individuals"]
    anions = [sp for sp in reaction_sys.species if sp.z < 0]
    cations = [sp for sp in reaction_sys.species if sp.z > 0]
    for c in cations:
        for a in anions:
            try:
                Bplus = db[c.name]["B"]
                Bminus = db[a.name]["B"]
                dplus = db[c.name]["d"]
                dminus = db[a.name]["d"]
                Bca = Bplus + Bminus + dplus * dminus
            except KeyError:
                Bca = 0.0
            c.p_scalar[a.name] = Bca
    return


# @numba.njit
def bromley_model_ion(I, Bi_j, zi, zj, mj):
    TK = 25.0 + 273.15
    A, _ = debye_huckel_constant(TK)

    e = 4.8029e-10  # erg
    k = 1.38045e-16  # erg
    Na = 6.02214076e23
    d0 = properties_utils.density_water(TK)
    D = properties_utils.dieletricconstant_water(TK)
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


# @numba.njit
def calc_bromley_method(idx_ctrl, species, I, T):
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

    # FIXME!!
    # species[idx_ctrl.idx['H2O']].set_log_gamma(-1.609437912)
    pass


# -------------------------------
# Specific Ion Interaction Theory (SIT) Model
# Same reference as PHREEQC:
# Grenthe et al_1997_Modelling in aquatic chemistry.pdf
# -------------------------------


def setup_SIT_model(reaction_sys, T, db_species, c_feed=None):
    """
    SIT Method as in Ref: Grenthe et al_1997_Modelling in aquatic chemistry.pdf
    """
    db_species = db_species["sit"]
    species = reaction_sys.species
    all_eps_names = [sp.name for sp in species]

    for sp_i in species:
        epslon_i = np.empty(len(species))
        for k, sp_name_k in enumerate(all_eps_names):
            try:
                e_ik = db_species[sp_i.name][sp_name_k]
            except KeyError:
                e_ik = 0.0
            epslon_i[k] = e_ik

        sp_i.p_array["eik"] = epslon_i
    return


def calc_sit_method(idx_ctrl, species, I, T):
    A, _ = debye_huckel_constant(T)
    Ba = 1.5
    mj = np.power(
        10, np.array([sp.logc for sp in species])
    )  # DO NOT NEED TO BE CALCULATED FOR EACH SPECIES, FIXME
    for sp in species:
        eik = sp.p_array["eik"]
        sum_interactions = eik @ mj
        logg = (
            -A * sp.z ** 2 * np.sqrt(I) / (1 + Ba * np.sqrt(I))
            + sum_interactions
        )

        sp.set_log_gamma(logg)
    pass


# -------------------------------
# e-NRTL
# Chen
# title = {A local composition model for the excess {Gibbs} energy of aqueous electrolyte systems},
# -------------------------------

db_enrtl_debug = {
    "Na": {
        "tau": {
            "H2O": -4.5916,
        }
    },
    "H2O": {
        "tau": {
            "H2O": 9.0234,
        }
    },
}


def setup_eNRTL_model(reaction_sys, T, db_species, c_feed=None):
    """
    eNRTL Method Chen (1986)
    """
    db_species = db_enrtl_debug
    species = reaction_sys.species
    all_eps_names = [sp.name for sp in species]

    for sp_i in species:
        epslon_i = np.empty(len(species))
        for k, sp_name_k in enumerate(all_eps_names):
            try:
                e_ik = db_species[sp_i.name][sp_name_k]
            except KeyError:
                e_ik = 0.0
            epslon_i[k] = e_ik

        sp_i.p_array["eik"] = epslon_i
    return


# @numba.njit
def calc_eNRTL_method(idx_ctrl, species, I, T):
    A, _ = debye_huckel_constant(T)
    Ba = 1.5
    mj = np.power(
        10, np.array([sp.logc for sp in species])
    )  # DO NOT NEED TO BE CALCULATED FOR EACH SPECIES, FIXME
    for sp in species:
        eik = sp.p_array["eik"]
        sum_interactions = eik @ mj
        logg = (
            -A * sp.z ** 2 * np.sqrt(I) / (1 + Ba * np.sqrt(I))
            + sum_interactions
        )

        sp.set_log_gamma(logg)
    pass


## AUXILIARIES


def logic_dh_fallback(sp, db_species):
    I_factor, dh_a, dh_b = species_definition_dh_model(sp.name, db_species)
    sp.p_scalar["I_factor"] = I_factor
    sp.p_scalar["dh_a"] = dh_a
    sp.p_scalar["dh_b"] = dh_b
    return


def get_charge_of_specie(tag):
    num_plus = len([c for c in tag if c == "+"])
    num_minus = len([c for c in tag if c == "-"])
    if num_plus > 0:
        return +num_plus
    elif num_minus > 0:
        return -num_minus
    else:
        return 0


def species_definition_dh_model(tag, species_activity_db):
    z = get_charge_of_specie(tag)
    if tag not in species_activity_db:
        if z == 0:
            I_factor = 0.1
            dh_a = np.nan
            dh_b = np.nan
        else:  # Else should use davies
            I_factor = np.nan
            dh_a = np.nan
            dh_b = np.nan
    else:
        db_specie = species_activity_db[tag]
        try:
            if "I_factor" in db_specie:
                I_factor = db_specie["I_factor"]
                dh_a = np.nan
                dh_b = np.nan
            else:
                I_factor = np.nan
                dh_a = db_specie["dh"]["phreeqc"]["a"]
                dh_b = db_specie["dh"]["phreeqc"]["b"]
        except KeyError as e:
            print("Error getting activity of specie = {}".format(tag))
            raise e
    return I_factor, dh_a, dh_b


# @numba.njit
def debye_huckel_constant(TK):
    epsilon = properties_utils.dieletricconstant_water(TK)
    rho = properties_utils.density_water(TK)
    A = 1.82483e6 * np.sqrt(rho) / (epsilon * TK) ** 1.5  # (L/mol)^1/2
    B = 50.2916 * np.sqrt(rho / (epsilon * TK))  # Angstrom^-1 . (L/mol)^1/2
    return A, B


# @numba.njit
def log10gamma_davies(I, z, A):
    sqI = np.sqrt(I)
    logGamma = -A * z ** 2 * (sqI / (1.0 + sqI) - 0.3 * I)
    return logGamma
