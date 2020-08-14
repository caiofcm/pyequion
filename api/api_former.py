import os
os.environ['NUMBA_DISABLE_JIT'] = '1' #WHAT A FUCK IS HAPPENING?
import pyequion
from flask import Flask
from flask_jsonrpc import JSONRPC
from flask_cors import CORS
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import numpy as np
import simplejson
import json

# from werkzeug.contrib.cache import SimpleCache
# cache = SimpleCache()

# Flask application
app = Flask(__name__)
CORS(app)

# Flask-JSONRPC
jsonrpc = JSONRPC(app, '/api', enable_web_browsable_api=True)

@dataclass_json
@dataclass
class EquilibriumModel:
    reactions: list
    reactionsLatex: list
    solidReactionsLatex: list

@dataclass_json
@dataclass
class SolutionResult():
    c_molal: list
    gamma: list
    pH: float
    I: float
    sc: float
    DIC: float
    solid_names: list
    specie_names: list
    saturation_index: dict
    preciptation_conc: dict
    ionic_activity_prod: dict
    log_K_solubility: dict
    idx: list
    reactions: list

# GLOBAL FOR DEV, Improv this
sys_eq = None
# sys_eq = pyequion.create_equilibrium( #DELETE ME
#         feed_compounds=['NaCl'],
#         initial_feed_mass_balance=['Cl-']
# )

#(String, String, String, Array, Array) -> Object', validate=True
@jsonrpc.method('App.create_equilibrium')
def create_equilibrium(compounds, closingEqType,
    initial_feed_mass_balance
    ):
    """
    Output: {
        'reactions': list of reactions in the system
    }
    """
    global sys_eq
    sys_eq = pyequion.create_equilibrium( ## PASSING ALLOW_PRECIPITATION IS WRONG! Such as polymorph formation, cannot precipitation all phases
        feed_compounds=compounds,
        # allow_precipitation=allowPrecipitation,
        closing_equation_type=closingEqType,
        initial_feed_mass_balance=initial_feed_mass_balance #REMOVE ME
    )
    if sys_eq.solid_reactions_but_not_equation is not None:
        solid_possible_reactions = pyequion.rbuilder.conv_reaction_engine_to_db_like(sys_eq.solid_reactions_but_not_equation)
        for r_solid in solid_possible_reactions:
            prev_keys = list(r_solid.keys())
            for in_element in prev_keys:
                if 'phase_name' in r_solid and '(s)' in in_element:
                    tag_add = in_element + '__' + r_solid['phase_name']
                    r_solid[tag_add] = r_solid.pop(in_element)
        # for item in reactions:
        # solid_possible_reactions = [{k:float(v) for k,v in item.items() if k[0].isupper()} for item in reactions]
        solid_possible_reactions_latex = pyequion.rbuilder.format_reaction_list_as_latex_mhchem(solid_possible_reactions)
    else:
        solid_possible_reactions_latex = []
    latex_reactions = pyequion.rbuilder.format_reaction_list_as_latex_mhchem(sys_eq.reactionsStorage)
    resp = EquilibriumModel(
        reactions=sys_eq.reactionsStorage,
        reactionsLatex=latex_reactions,
        solidReactionsLatex=solid_possible_reactions_latex,
    )
    # cache.set('eq_sys', resp, timeout=5 * 60)
    print(resp)

    # FIX THIS: Stupid way to remove NaN:
    as_json_str = simplejson.dumps(resp.to_dict(), ignore_nan=True)
    back_to_dict = json.loads(as_json_str)

    return back_to_dict
    # return resp.to_dict()


@jsonrpc.method('App.solve_equilibrium')
def solve_equilibrium(concentrations, temperature, extraParameter, allowPrecipitation, nonidealityType):
    """
    Output: {
        'reactions': list of reactions in the system
    }
    """
    # rv = cache.get('eq_sys')
    if not sys_eq:
        raise ValueError('Equilibrium System not defined')

    TK = temperature + 273.15
    extra_param = extraParameter if extraParameter else np.nan
    if sys_eq.closing_equation_type == pyequion.ClosingEquationType.CARBON_TOTAL:
        extra_param *= 1e-3
    args = (np.array(concentrations)*1e-3, TK, extra_param)
    xguess = None #np.ones(sys_eq.idx_control.idx['size'])*1e-1
    solResult_pyEq = pyequion.solve_equilibrium(sys_eq,
        args=args, x_guess=xguess,
        allow_precipitation=allowPrecipitation,
        activity_model_type=pyequion.TypeActivityCalculation[nonidealityType])
    print('DONE')
    print(solResult_pyEq)
    solResult = SolutionResult(
        solResult_pyEq.c_molal,
        solResult_pyEq.gamma,
        solResult_pyEq.pH,
        solResult_pyEq.I,
        solResult_pyEq.sc,
        solResult_pyEq.DIC,
        # 2.0,
        solResult_pyEq.solid_names,
        solResult_pyEq.specie_names,
        solResult_pyEq.saturation_index,
        solResult_pyEq.preciptation_conc,
        solResult_pyEq.ionic_activity_prod,
        solResult_pyEq.log_K_solubility,
        solResult_pyEq.idx,
        solResult_pyEq.reactions,
    )

    print(solResult)

    # FIX THIS: Stupid way to remove NaN:
    as_json_str = simplejson.dumps(solResult.to_dict(), ignore_nan=True)
    back_to_dict = json.loads(as_json_str)
    return back_to_dict

@jsonrpc.method('App.hello')
def hello():
    return 'Hello!'

if __name__ == '__main__':
    app.run(debug=True) #host='0.0.0.0',

a = 1
