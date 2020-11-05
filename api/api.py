import os

os.environ["NUMBA_DISABLE_JIT"] = "1"  # reconsider
import pyequion
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import numpy as np
import simplejson
import json

IS_LOCAL = False
# Flask application - For local debugging
if IS_LOCAL:
    from flask import Flask
    from flask_jsonrpc import JSONRPC
    from flask_cors import CORS
    from flask import request  # TEST

    app = Flask(__name__)

    CORS(app)

"""
Testing:

{
  "endpoint": "App.create_equilibrium",
  "params": {
        "compounds": ["NaCl"],
        "closingEqType": 0,
        "initial_feed_mass_balance": ["Cl-"]
    }
}


{
    "concentrations": [10],
    "temperature": 25.0,
    "extraParameter": 0.0034,
    "allowPrecipitation": false,
    "nonidealityType": 0,
}

"""


@dataclass_json
@dataclass
class EquilibriumModel:
    reactions: list
    reactionsLatex: list
    solidReactionsLatex: list
    sys_eq: dict


@dataclass_json
@dataclass
class SolutionResult:
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


# sys_eq = None

# @conditional_decorator(app.route('/api', methods = ['POST']), IS_LOCAL)
# @app.route('/api', methods = ['POST'])
def pyequion_api(request):
    # def pyequion_api():
    """
    Output: {
        'reactions': list of reactions in the system
    }
    """
    # request = #TESTING
    # global sys_eq

    # Set CORS headers for the preflight request
    if request.method == "OPTIONS":
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            "Access-Control-Allow-Origin": "*",  # https://caiofcm.github.io/ todo
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }

        return ("", 204, headers)

    # Set CORS headers for the main request
    headers = {"Access-Control-Allow-Origin": "*"}

    request_json = request.get_json()
    print(request_json)
    if "method" not in request_json:
        json_resp = json.dumps(
            {
                "status": "fail",
                "error": {
                    "message": "Request endpoint should be <App.create_equilibrium> or <App.solve_equilibrium> "
                },
            }
        )
        return (json_resp, 200, headers)

    endpoint = request_json["method"]

    as_json_str = ""

    if endpoint == "App.startup":
        return (json.dumps({"result": "Started OK"}), 200, headers)
    elif endpoint == "App.create_equilibrium":
        params = request_json["params"]
        compounds = params["compounds"]
        closingEqType = params["closingEqType"]
        initial_feed_mass_balance = params["initial_feed_mass_balance"]

        # Creating the Equilibrium System
        sys_eq = pyequion.create_equilibrium(  # PASSING ALLOW_PRECIPITATION IS WRONG! Such as polymorph formation, cannot precipitation all phases
            feed_compounds=compounds,
            # allow_precipitation=allowPrecipitation,
            closing_equation_type=closingEqType,
            initial_feed_mass_balance=initial_feed_mass_balance,  # REMOVE ME
        )
        # Serializing EquilibriumSystem
        stringfied_sys_eq = json.dumps(
            sys_eq, cls=pyequion.utils_api.NpEncoder
        )
        as_dict_sys_eq = json.loads(stringfied_sys_eq)

        # Getting Solid Reactions
        if sys_eq.solid_reactions_but_not_equation is not None:
            solid_possible_reactions = (
                pyequion.rbuilder.conv_reaction_engine_to_db_like(
                    sys_eq.solid_reactions_but_not_equation
                )
            )
            pyequion.rbuilder.fill_reactions_with_solid_name_underscore(
                solid_possible_reactions
            )
            # for item in reactions:
            # solid_possible_reactions = [{k:float(v) for k,v in item.items() if k[0].isupper()} for item in reactions]
            solid_possible_reactions_latex = (
                pyequion.rbuilder.format_reaction_list_as_latex_mhchem(
                    solid_possible_reactions
                )
            )
        else:
            solid_possible_reactions_latex = []

        # Getting Aqueous Reactions
        latex_reactions = (
            pyequion.rbuilder.format_reaction_list_as_latex_mhchem(
                sys_eq.reactionsStorage
            )
        )

        # Creating API Response
        resp = EquilibriumModel(
            reactions=sys_eq.reactionsStorage,
            reactionsLatex=latex_reactions,
            solidReactionsLatex=solid_possible_reactions_latex,
            sys_eq=as_dict_sys_eq,
        )
        # cache.set('eq_sys', resp, timeout=5 * 60)
        print(resp)

        eq_out = {"result": resp.to_dict()}

        # FIX THIS: Stupid way to remove NaN:
        as_json_str = simplejson.dumps(eq_out, ignore_nan=True)
    # back_to_dict = json.loads(as_json_str)

    elif endpoint == "App.solve_equilibrium":
        params = request_json["params"]
        concentrations = params["concentrations"]
        temperature = params["temperature"]
        extraParameter = params["extraParameter"]
        allowPrecipitation = params["allowPrecipitation"]
        nonidealityType = params["nonidealityType"]
        sys_eq_serialized = params["sys_eq"]
        sys_eq_deslrd = pyequion.utils_api.create_eq_sys_from_serialized(
            sys_eq_serialized
        )
        solResult = solve_equilibrium(
            sys_eq_deslrd,
            concentrations,
            temperature,
            extraParameter,
            allowPrecipitation,
            nonidealityType,
        )
        eq_sol_out = {"result": solResult.to_dict()}
        as_json_str = simplejson.dumps(eq_sol_out, ignore_nan=True)
    else:
        # raise ValueError('Unknown method')
        as_json_str = json.dumps(
            {
                "status": "fail",
                "error": {
                    "message": "Request endpoint should be <App.create_equilibrium> or <App.solve_equilibrium> "
                },
            }
        )

    return (as_json_str, 200, headers)


def solve_equilibrium(
    sys_eq,
    concentrations,
    temperature,
    extraParameter,
    allowPrecipitation,
    nonidealityType,
):
    """
    Output: {
        'reactions': list of reactions in the system
    }
    """
    # rv = cache.get('eq_sys')
    if not sys_eq:
        raise ValueError("Equilibrium System not defined")

    TK = temperature + 273.15
    extra_param = extraParameter if extraParameter else np.nan
    if (
        sys_eq.closing_equation_type
        == pyequion.ClosingEquationType.CARBON_TOTAL
    ):
        extra_param *= 1e-3
    args = (np.array(concentrations) * 1e-3, TK, extra_param)
    xguess = None  # np.ones(sys_eq.idx_control.idx['size'])*1e-1
    fugacity_calc = (
        "pr"
        if sys_eq.closing_equation_type == pyequion.ClosingEquationType.OPEN
        else "ideal"
    )
    solResult_pyEq = pyequion.solve_equilibrium(
        sys_eq,
        args=args,
        x_guess=xguess,
        allow_precipitation=allowPrecipitation,
        activity_model_type=pyequion.TypeActivityCalculation[nonidealityType],
        fugacity_calculation=fugacity_calc,
    )
    print("DONE")
    # print(solResult_pyEq)
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

    return solResult  # .to_dict()


def pyequion_api_local_flask():
    return pyequion_api(request)


if IS_LOCAL:

    app.route("/api", methods=["POST"])(pyequion_api_local_flask)

    if __name__ == "__main__":
        app.run(debug=True)
