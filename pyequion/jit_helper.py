import numpy as np
from . import utils_for_numba
from . import core
from . import conductivity
from . import activity_coefficients as act
from . import properties_utils

import numba
try:
    from numba.experimental import jitclass
except:
    from numba import jitclass
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False



import warnings #FIXME Check Those Numba warnings
warnings.filterwarnings("ignore", category=numba.NumbaPendingDeprecationWarning)

@numba.njit()
def initialize_dict_numbafied():
    d_int = {'size': 0}
    d_scalar = {'v': 0.0}
    d_iarray = {'a': np.empty(1, dtype=np.int64)}
    d_array = {'a': np.empty(1)}
    d_matrix = {'m': np.empty((1,1))}
    d_nested = {'n': {'i': np.empty((1))}}
    d_string = {'n': 'H+'}
    return d_int, d_scalar, d_iarray, d_array, d_matrix, d_nested, d_string

d_int, d_scalar, d_iarray, d_array, d_matrix, d_nested, d_string = initialize_dict_numbafied()

@numba.njit
def create_typed_lists():
    l_species = numba.typed.List()
    s = core.Specie(0, 0, 'dummy')
    l_species.append(s)
    l_string = numba.typed.List()
    l_string.append('H+(s)')
    # l_l_string = numba.typed.List()
    # l_l_string.append(l_string)
    return l_species, l_string

@numba.njit
def create_numba_list_of_dict():
    d_scalar = {'H+': 0.0}
    l_d = numba.typed.List()
    l_d.append(d_scalar)
    return l_d

def set_list_type_for_jit():
    # print('in set_list_type_for_jit')
    #global List, Dict #from .utils_for_numba import List, Dict
    core.List = numba.typed.List
    core.Dict = numba.typed.Dict
    return

def jit_compile_functions():
    # utils_for_numba.USE_JIT = True
    utils_for_numba.set_list_type_for_jit()
    set_list_type_for_jit()

    d_int, d_scalar, d_iarray, d_array, d_matrix, d_nested, d_string = initialize_dict_numbafied()

    specs_specie = [
        ('logc', numba.float64),
        ('logg', numba.float64),
        ('z', numba.int64),
        ('phase', numba.int64), #Improve, but See Phase
        ('name', numba.types.string),
        # ('I_factor', numba.float64),
        # ('dh_a', numba.float64),
        # ('dh_b', numba.float64),
        ('cond_molar', numba.float64),
        ('p_int', numba.typeof(d_int)), #scalar parameters
        ('p_scalar', numba.typeof(d_scalar)), #scalar parameters
        ('p_iarray', numba.typeof(d_iarray)), #int array parameters
        ('p_array', numba.typeof(d_array)),  #array parameters
        ('p_matrix', numba.typeof(d_matrix)),  #matrix parameters
        ('d', numba.typeof(d_nested)),  #nested dict float parameters
    ]

    # global Specie
    core.Specie = jitclass(specs_specie)(core.Specie)

    l_species, l_string = create_typed_lists()
    l_d_string_float = create_numba_list_of_dict()
    type_list_specie = numba.typeof(l_species)

    spec_result = [
        ('c_molal', numba.float64[:]),
        ('gamma', numba.float64[:]),
        ('pH', numba.float64),
        ('I', numba.float64),
        ('DIC', numba.float64), #Only the dissolved Carbon (solid not counted)
        ('sc', numba.float64),
        # ('SI', numba.typeof(SI_Dict)),
        # ('SI', numba.float64[:]),
        ('IAP', numba.float64[:]),
        ('solid_names', numba.types.List(numba.types.unicode_type)),
        ('precipitation_molalities', numba.float64[:]),
        # ('specie_names', numba.types.List(numba.types.unicode_type)),
        ('specie_names', numba.typeof(l_string)),
        ('saturation_index', numba.typeof(d_scalar)),
        ('preciptation_conc', numba.typeof(d_scalar)),
        ('ionic_activity_prod', numba.typeof(d_scalar)),
        ('log_K_solubility', numba.typeof(d_scalar)),
        ('idx', numba.typeof(d_int)),
        ('reactions', numba.typeof(l_d_string_float)),
        ('index_solubility_calculation', numba.int64),
        ('calculated_solubility', numba.typeof(d_scalar)),
        ('concentrations', numba.typeof(d_scalar)),
        ('x', numba.float64[:]), #Numerical solution
        ('successfull', numba.boolean),
    ]

    # global core.SolutionResult
    core.SolutionResult = jitclass(spec_result)(core.SolutionResult)

    specs_idx = [
        ('idx', numba.typeof(d_int)),
        ('s', numba.typeof(d_scalar)),
        ('a', numba.typeof(d_array)),
        ('m', numba.typeof(d_matrix)),
    ]

    # global core.IndexController
    core.IndexController = jitclass(specs_idx)(core.IndexController)


    specs_reacs = [
        # ('idx_species__', numba.int64),
        # ('stoic_coefs', numba.float64),
        # ('constant_T_coefs', numba.float64),

        ('idx_species', numba.int64[:]),
        ('stoic_coefs', numba.float64[:]),
        # ('stoic_coefs', numba.types.List(numba.int64)),
        ('idx_reaction_db', numba.int64),
        ('constant_T_coefs', numba.float64[:]),
        ('log_K25', numba.float64),
        ('type', numba.types.unicode_type),

        ('delta_h', numba.float64),
        # ('species_tags', numba.types.List(numba.types.unicode_type)),
        ('species_tags', numba.typeof(l_string)),
    ]

    # global EqReaction
    core.EqReaction = jitclass(specs_reacs)(core.EqReaction)

    # global DUMMY_EqREACTION
    core.DUMMY_EqREACTION = core.EqReaction(
            np.array([-1, -1], dtype=np.int64),
            # np.array([-1]),
            np.array([np.nan], dtype=np.float64),
            -1.0, np.array([np.nan], dtype=np.float64), 'dummy', utils_for_numba.create_nb_List(['']), np.nan
        ) #-1 IS DUMMY -> Numba issues

    # if os.getenv('NUMBA_DISABLE_JIT') != "1":
    l_reactions = numba.typed.List()
    l_reactions.append(core.DUMMY_EqREACTION)

    spec_mass_balance = [
        ('idx_species', numba.int64[:]),
        ('stoic_coefs', numba.float64[:]),
        ('idx_feed', numba.types.List(numba.types.Tuple( (numba.int64, numba.float64))) ),
        ('use_constant_value', numba.boolean),
        ('feed_is_unknown', numba.boolean),
    ]

    # global MassBalance
    core.MassBalance = jitclass(spec_mass_balance)(core.MassBalance)

    # global DUMMY_MB
    core.DUMMY_MB = core.MassBalance(np.array([-1]), np.array([-1.0]), [(-1, -1.0)], False)

    specs_eqsys = [
        ('c_feed', numba.float64),
        # Extra variables for equilibrium specific cases: to be more generic is a list of np.ndarray (it has to be an array)
        # Arguments in function is necessary for the jacobian, otherwise the jacobian would need to be generated for each change in args
        # This field args is just for convenience to have it stored (OR NOT MAY BE REMOVED)
        ('args', numba.types.List(numba.float64[:])),
        ('res', numba.float64[:]),
        ('TK', numba.float64),
        # ('idx', Indexes.class_type.instance_type),
        ('idx_control', core.IndexController.class_type.instance_type),
        # ('species', numba.types.List(Specie.class_type.instance_type)),
        ('species', type_list_specie),
        # ('reactions', numba.types.List(EqReaction.class_type.instance_type)),
        ('reactions', numba.typeof(l_reactions)),
        ('ionic_strength', numba.float64),
        ('pH', numba.float64),
        ('sc', numba.float64),
        ('molar_conc', numba.float64[:]),
        ('gamma', numba.float64[:]),
        # ('activity_model_type', numba.typeof(TypeActivityCalculation)),
        # ('activity_model_type', numba.types.EnumMember(TypeActivityCalculation, numba.int64)),
        ('mass_balances', numba.types.List(core.MassBalance.class_type.instance_type)),
        ('mass_balances_known', numba.types.List(core.MassBalance.class_type.instance_type)),
        ('is_there_known_mb', numba.boolean),
        ('dic_idx_coef', numba.types.List(numba.types.Tuple( (numba.int64, numba.float64))) ),
        # ('solid_reactions_but_not_equation', numba.types.List(EqReaction.class_type.instance_type)),
        ('solid_reactions_but_not_equation', numba.typeof(l_reactions)),
        ('num_of_feeds', numba.int64),

        # System creation related
        ('feed_compounds', numba.typeof(l_string)),
        ('closing_equation_type', numba.int64),
        ('element_mass_balance', numba.typeof(l_string)),
        ('initial_feed_mass_balance', numba.typeof(l_string)),
        ('fixed_elements', numba.typeof(l_string)),
        # ('database_files', numba.typeof(d_string)),
        ('reactionsStorage', numba.typeof(l_d_string_float)),
        ('index_solubility_calculation', numba.int64),
        ('fugacity_calculation', numba.types.unicode_type), #TEST: will fail in numba

        ('allow_precipitation', numba.boolean),
        ('solid_reactions_in', numba.typeof(l_d_string_float)),
        # ('known_tags', numba.typeof(l_string)),
    ]

    # global EquilibriumSystem
    core.EquilibriumSystem = jitclass(specs_eqsys)(core.EquilibriumSystem)


    # Activity coefficients model compilation
    conductivity.solution_conductivity = numba.njit()(conductivity.solution_conductivity)
    act.log10gamma_davies = numba.njit()(act.log10gamma_davies)
    act.debye_huckel_constant = numba.njit()(act.debye_huckel_constant)
    act.b_dot_equation = numba.njit()(act.b_dot_equation)
    act.calc_log_gamma_ideal = numba.njit()(act.calc_log_gamma_ideal)
    act.calc_log_gamma_dh_bdot = numba.njit()(act.calc_log_gamma_dh_bdot)
    act.bromley_model_ion = numba.njit()(act.bromley_model_ion)
    act.calc_bromley_method = numba.njit()(act.calc_bromley_method)
    act.calc_sit_method = numba.njit()(act.calc_sit_method)
    act.calc_log_gamma_pitzer = numba.njit()(act.calc_log_gamma_pitzer)
    act.calc_log_gamma_dh_bdot_mean_activity_neutral = numba.njit()(act.calc_log_gamma_dh_bdot_mean_activity_neutral)
    properties_utils.dieletricconstant_water = numba.njit()(properties_utils.dieletricconstant_water)
    properties_utils.density_water = numba.njit()(properties_utils.density_water)
    core.logK_H = numba.njit()(core.logK_H)

    print('End JIT compilation settings')
    return


#-------------------------------------------------
#-------------------------------------------------
#    NOT ON USE..
#-------------------------------------------------
#-------------------------------------------------

# @numba.njit
def solve_equilibrium_numbafied(reaction_system, args, jac, x_guess=None):
    "TODO.. also"

    if x_guess is None:
        x_guess = np.full(reaction_system.idx.size, -1.0)

    # jac_numerical = utils_for_numba.create_jacobian(reaction_system.residual)
    x, iteration_counter = utils_for_numba.root_finding_newton(
        reaction_system.residual,
        jac, x_guess, 1e-7, 200, args)
    solution = reaction_system.calculate_properties(True)
    return solution, x

# @numba.njit
def solve_equilibrium_numerical_jac_numbafied(reaction_system, args, x_guess=None):

    if x_guess is None:
        x_guess = np.full(reaction_system.idx.size, -1.0)

    # jac_numerical = utils_for_numba.create_jacobian(reaction_system.residual)
    x, iteration_counter = utils_for_numba.root_finding_newton(
        reaction_system.residual,
        reaction_system.numerical_jac,
        x_guess, 1e-7, 200, args)
    solution = reaction_system.calculate_properties(True)
    return solution, x
