
# import commentjson as json
import numpy as np
import json
from . import core
from . import pyequion
from . import utils_for_numba
import numba

#####
#####
# Serialization Helpers
#####
#####

class SimpleEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return SimpleEncoder.default(self, obj)

def cnv_reaction_from_json_serialized(r):
    return core.EqReaction(np.array(r['idx_species']),
                             np.array(r['stoic_coefs']),
                             r['log_K25'],
                             np.array(r['constant_T_coefs']),
                             r['type'], r['species_tags'],
                             r['delta_h']
    )

def create_eq_sys_from_serialized(obj_json):
    feed_compounds = obj_json['feed_compounds']
    closing_equation_type = obj_json['closing_equation_type']
    element_mass_balance = obj_json['element_mass_balance']
    initial_feed_mass_balance = obj_json['initial_feed_mass_balance']
    allow_precipitation = obj_json['allow_precipitation']
    fixed_elements = obj_json['fixed_elements']
    # database_files = obj_json['database_files']
    database_files = {'dummy': 'dummy'}
    solid_reactions_in = obj_json['solid_reactions_in']
    reactions_str = obj_json['reactions']
    reactions = [cnv_reaction_from_json_serialized(r) for r in reactions_str]
    possible_aqueous_reactions_in = pyequion.rbuilder.conv_reaction_engine_to_db_like(reactions)
    reactions_solid_not_eq_str = obj_json['solid_reactions_but_not_equation']
    reactions_solid_but_not_eq = [cnv_reaction_from_json_serialized(r) for r in reactions_solid_not_eq_str]
    possible_solid_reactions_in = pyequion.rbuilder.conv_reaction_engine_to_db_like(reactions_solid_but_not_eq)

    new_sys_eq = pyequion.create_equilibrium(
        feed_compounds,
        closing_equation_type,
        element_mass_balance,
        initial_feed_mass_balance,
        allow_precipitation,
        False,
        fixed_elements,
        # database_files,
        solid_reactions_in=solid_reactions_in,
        possible_aqueous_reactions_in=possible_aqueous_reactions_in,
        possible_solid_reactions_in=possible_solid_reactions_in
    )
    return new_sys_eq

def cnv_reaction_from_json_serialized_numbafy(r):
    species_tag_nb = utils_for_numba.create_nb_List(r['species_tags'])
    return core.EqReaction(np.array(r['idx_species']),
                             np.array(r['stoic_coefs']),
                             r['log_K25'],
                             np.array(r['constant_T_coefs']),
                             r['type'], species_tag_nb,
                             r['delta_h']
    )

def create_empty_dict(type_):
    mappin_types = {
        'float': numba.types.float64,
        'array': numba.types.float64[:],
        'matrix': numba.types.float64[:,:],
        'int': numba.types.int64,
        'intArray': numba.types.int64[:],
    }
    d = numba.typed.Dict.empty(
        key_type=numba.types.unicode_type,
        value_type=mappin_types[type_],
    )
    return d

def convert_dict_str_float_array(d):
    value_use = create_empty_dict('array')
    for k, v in d.items():
        value_use[k] = v
    return value_use

def convert_dict_str_int(d):
    value_use = create_empty_dict('int')
    for k, v in d.items():
        value_use[k] = v
    return value_use

def convert_dict_str_float(d):
    value_use = create_empty_dict('float')
    for k, v in d.items():
        value_use[k] = v
    return value_use

def convert_dict_str_int_array(d):
    value_use = create_empty_dict('intArray')
    for k, v in d.items():
        value_use[k] = v
    return value_use

def convert_dict_str_float_matrix(d):
    value_use = create_empty_dict('matrix')
    for k, v in d.items():
        value_use[k] = v
    return value_use

def create_eq_sys_from_serialized_numbafy(obj_json):
    # feed_compounds = obj_json['feed_compounds']
    # closing_equation_type = obj_json['closing_equation_type']
    # element_mass_balance = obj_json['element_mass_balance']
    # initial_feed_mass_balance = obj_json['initial_feed_mass_balance']
    # allow_precipitation = obj_json['allow_precipitation']
    # fixed_elements = obj_json['fixed_elements']
    # # database_files = obj_json['database_files']
    # database_files = {'dummy': 'dummy'}
    # solid_reactions_in = obj_json['solid_reactions_in']
    # possible_aqueous_reactions_in = pyequion.rbuilder.conv_reaction_engine_to_db_like(reactions)
    # reactions_solid_not_eq_str = obj_json['solid_reactions_but_not_equation']
    # reactions_solid_but_not_eq = [cnv_reaction_from_json_serialized_numbafy(r) for r in reactions_solid_not_eq_str]
    # possible_solid_reactions_in = pyequion.rbuilder.conv_reaction_engine_to_db_like(reactions_solid_but_not_eq)

    "Species"
    o_species = numba.typed.List()
    for s_specie in obj_json['species']:
        o_specie = pyequion.core.Specie(s_specie['z'], s_specie['phase'], s_specie['name'])

        for key, value in s_specie['p_array'].items():
            o_specie.p_array[key] = np.array(value)
        for key, value in s_specie['p_int'].items():
            o_specie.p_int[key] = value
        for key, value in s_specie['p_scalar'].items():
            o_specie.p_scalar[key] = value
        for key, value in s_specie['p_iarray'].items():
            o_specie.p_iarray[key] = np.array(value)
        for key, value in s_specie['p_matrix'].items():
            o_specie.p_matrix[key] = np.array(value)

        o_species.append(o_specie)

    "Index"
    species_tags = [k for k in obj_json['idx_control']['idx'].keys() if k != 'size']
    idx_ctrl = core.IndexController(species_tags, obj_json['idx_control']['idx']['size'])
    for key, value in obj_json['idx_control']['idx'].items():
        idx_ctrl.idx[key] = value
    for key, value in obj_json['idx_control']['s'].items():
        idx_ctrl.s[key] = value

    "Reactions"
    reactions_str = obj_json['reactions']
    reaction_list = numba.typed.List()
    for r in reactions_str:
        rnew = cnv_reaction_from_json_serialized_numbafy(r)
        reaction_list.append(rnew)
    # reactions = [cnv_reaction_from_json_serialized_numbafy(r) for r in reactions_str]

    "Mass Balances"
    # mb_list = numba.typed.List()
    mb_list = []
    for mb_dic in obj_json['mass_balances']:
        idx_species = np.array(mb_dic['idx_species'])
        stoic_coefs = np.array(mb_dic['stoic_coefs'])
        idx_feed = [tuple(item) for item in mb_dic['idx_feed']]
        use_constant_value = mb_dic['use_constant_value']
        feed_is_unknown = mb_dic['feed_is_unknown']
        mb_cnv = core.MassBalance(
            idx_species,
            stoic_coefs,
            idx_feed,
            use_constant_value,
            feed_is_unknown
        )
        mb_list.append(mb_cnv)

    "Known Specie"
    # mb_known_list = numba.typed.List()
    mb_known_list = []
    if 'mass_balances_known' in obj_json:
        for mb_dic in obj_json['mass_balances_known']:
            idx_species = np.array(mb_dic['idx_species'])
            stoic_coefs = np.array(mb_dic['stoic_coefs'])
            idx_feed = [tuple(item) for item in mb_dic['idx_feed']]
            use_constant_value = mb_dic['use_constant_value']
            feed_is_unknown = mb_dic['feed_is_unknown']
            mb_cnv = core.MassBalance(
                idx_species,
                stoic_coefs,
                idx_feed,
                use_constant_value,
                feed_is_unknown
            )
            mb_known_list.append(mb_cnv)
    else:
        mb_known_list.append(core.DUMMY_MB)

    "Dict Index Tuple"
    s_dic_tuple = obj_json['dic_idx_coef']
    dic_tuple = [tuple(dt) for dt in s_dic_tuple]

    "Solid reactions In"
    s_reactions_solid_ne = obj_json['solid_reactions_but_not_equation']
    solid_ne_reaction_list = numba.typed.List()
    for r in s_reactions_solid_ne:
        rnew = cnv_reaction_from_json_serialized_numbafy(r)
        solid_ne_reaction_list.append(rnew)

    num_of_feeds = obj_json['num_of_feeds']

    s_feed_compounds = obj_json['feed_compounds']
    feed_compounds = numba.typed.List()
    for item in s_feed_compounds:
        feed_compounds.append(item)

    closing_equation_type = obj_json['closing_equation_type']

    s_element_mass_balance = obj_json['element_mass_balance']
    element_mass_balance = numba.typed.List()
    if len(s_element_mass_balance) == 0:
        element_mass_balance.append('a')
        element_mass_balance.pop()
    for item in s_element_mass_balance:
        element_mass_balance.append(item)

    s_initial_feed_mass_balance = obj_json['initial_feed_mass_balance']
    initial_feed_mass_balance = numba.typed.List()
    if len(s_initial_feed_mass_balance) == 0:
        initial_feed_mass_balance.append('a')
        initial_feed_mass_balance.pop()
    else:
        for item in s_initial_feed_mass_balance:
            initial_feed_mass_balance.append(item)

    s_fixed_elements = obj_json['fixed_elements']
    fixed_elements = numba.typed.List()
    for item in s_fixed_elements:
        fixed_elements.append(item)

    s_fixed_elements = obj_json['fixed_elements']
    fixed_elements = numba.typed.List()
    if len(s_fixed_elements) == 0:
        fixed_elements.append('a')
        fixed_elements.pop()
    else:
        for item in s_fixed_elements:
            fixed_elements.append(item)

    s_reactionsStorage = obj_json['reactionsStorage']
    List_reactionsStg = numba.typed.List()
    for rec_stg in s_reactionsStorage:
        Dict_reactionsStg = numba.typed.Dict.empty(
            key_type=numba.types.unicode_type,
            value_type=numba.types.float64,
        )
        for key,val in rec_stg.items():
            Dict_reactionsStg[key] = val
        List_reactionsStg.append(Dict_reactionsStg)

    allow_precipitation = obj_json['allow_precipitation']

    s_solid_reactions_in = obj_json['solid_reactions_in']
    List_solid_reactions_in = numba.typed.List()
    for rec_stg in s_solid_reactions_in:
        Dict_List_solid_reactions_in = numba.typed.Dict.empty(
            key_type=numba.types.unicode_type,
            value_type=numba.types.float64,
        )
        for key,val in rec_stg.items():
            Dict_List_solid_reactions_in[key] = val
        List_solid_reactions_in.append(Dict_List_solid_reactions_in)

    # p_array = convert_dict_str_float_array(o_specie.p_array)
    # p_int = convert_dict_str_int(o_specie.p_int)
    # p_scalar = convert_dict_str_float(o_specie.p_scalar)
    # p_iarray = convert_dict_str_int_array(o_specie.p_iarray)
    # p_matrix = convert_dict_str_float_matrix(o_specie.p_matrix)
    # o_specie.p_array = p_array
    # o_specie.p_int = p_int
    # o_specie.p_scalar = p_scalar
    # o_specie.p_iarray = p_iarray
    # o_specie.p_matrix = p_matrix
    # for key, value in s_specie.items():
    #     if isinstance(value, dict):
    #         typeOfValue = value['dummy']
    #         value_use = create_empty_dict()
    #     setattr(o_specie, key, value)


    new_sys_eq = core.EquilibriumSystem(
        o_species,
        idx_ctrl,
        reaction_list,
        mb_list,
        mb_known_list,
        dic_tuple,
        solid_ne_reaction_list,
        num_of_feeds,
        feed_compounds,
        closing_equation_type,
        element_mass_balance,
        initial_feed_mass_balance,
        fixed_elements,
        List_reactionsStg,
        allow_precipitation,
        List_solid_reactions_in
    )

    # new_sys_eq = pyequion.create_equilibrium(
    #     feed_compounds,
    #     closing_equation_type,
    #     element_mass_balance,
    #     initial_feed_mass_balance,
    #     allow_precipitation,
    #     False,
    #     fixed_elements,
    #     # database_files,
    #     solid_reactions_in=solid_reactions_in,
    #     possible_aqueous_reactions_in=possible_aqueous_reactions_in,
    #     possible_solid_reactions_in=possible_solid_reactions_in
    # )
    return new_sys_eq

"""" Serialize Aux """
def serialize_sys_eq(sys_eq):
    # if sys_eq.solid_reactions_in:
    #     aux_solid_reactions = [
    #         {
    #             **r,

    #         }
    #         for r in sys_eq.solid_reactions_in
    #     ]
    stringfied_sys_eq = json.dumps(sys_eq, cls=pyequion.utils_api.NpEncoder)
    return
