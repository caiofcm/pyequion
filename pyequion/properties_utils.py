import numpy

global np
np = numpy
from scipy import optimize

pCO2_ref = 0.0003908408957924021


def dieletricconstant_water(TK):
    # for TK: 273-372
    return 0.24921e3 - 0.79069 * TK + 0.72997e-3 * TK ** 2


# @numba.njit
def density_water(TK):
    # for TK: 273-372
    return (
        0.183652
        + 0.00724987 * TK
        - 0.203449e-4 * TK ** 2
        + 1.73702e-8 * TK ** 3
    )


def power_10(el):
    return {key: 10 ** (el[key]) for key in el}


# --------------------------------------------
# 	 NUMERICAL HELPERS
# --------------------------------------------
def solve_with_exception(fun, x0, args=None, jac=None):
    if args is not None:
        # kwarg_root = dict(args=(args,), method='hybr',
        #                   jac=jac, options={'maxfev': int(1e7)})
        # kwarg_root = dict(args=(args,), method='lm',
        #   jac=jac)
        kwarg_root = dict(
            args=args,
            method="lm",
            jac=jac,
            #   options={'factor':0.3}
            options={"factor": 100},
        )
    else:
        # kwarg_root = dict(method='hybr', jac=jac, options={'maxfev': int(1e5)})
        kwarg_root = dict(method="lm", jac=jac)
    sol = optimize.root(fun, x0, tol=1e-15, **kwarg_root)
    if not sol.success:
        print(sol.message)
        raise ValueError("Error during reaction speciation calculation")
    return sol
