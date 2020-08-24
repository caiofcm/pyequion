import numba
import numpy
global np
np = numpy
from scipy import optimize
import numba

pCO2_ref = 0.0003908408957924021

# def use_sympy_as_np()
# @numba.njit
def dieletricconstant_water(TK):
    # for TK: 273-372
    return (0.24921e3 - 0.79069*TK + 0.72997e-3*TK**2)

# @numba.njit
def density_water(TK):
    # for TK: 273-372
    return (0.183652 + 0.00724987*TK - 0.203449e-4*TK**2 + 1.73702e-8*TK**3)

def power_10(el):
    return {key: 10**(el[key]) for key in el}

## IONS DTYPES Definitions
ions_tags_array = ['H+', 'OH-', 'CO2', 'CO3--', 'HCO3-', 'Ca++',
    'CaOH+', 'CaHCO3+', 'CaCO3', 'Na+', 'NaOH', 'NaCO3-', 'NaHCO3',
    'Cl-', 'HCl', 'NaCl', 'H2O', 'Na2CO3']

dtype_ions = [(tag, np.float64) for tag in ions_tags_array]

# ## Species Charge
# charge_species = np.empty(1, dtype=dtype_ions)
# charge_species['Na+'] = 1
# charge_species['H+'] = 1
# charge_species['Ca++'] = 2
# charge_species['Cl-'] = -1
# charge_species['HCO3-'] = -1
# charge_species['NaCO3-'] = -1
# charge_species['OH-'] = -1
# charge_species['CO3--'] = -2
# charge_species['OH-'] = -1
# charge_species['CaOH+'] = 1
# charge_species['CaHCO3+'] = 1
# charge_species['CO2'] = 0
# charge_species['CaCO3'] = 0
# charge_species['NaOH'] = 0
# charge_species['NaHCO3'] = 0
# charge_species['Na2CO3'] = 0
## Species Charge
# charge_species = {} #np.empty(1, dtype=dtype_ions)
# charge_species['Nap'] = 1
# charge_species['Hp'] = 1
# charge_species['Capp'] = 2
# charge_species['Clm'] = -1
# charge_species['HCO3m'] = -1
# charge_species['NaCO3m'] = -1
# charge_species['OHm'] = -1
# charge_species['CO3mm'] = -2
# charge_species['OHm'] = -1
# charge_species['CaOHp'] = 1
# charge_species['CaHCO3p'] = 1
# charge_species['CO2'] = 0
# charge_species['CaCO3'] = 0
# # charge_species['CaCl2'] = 0
# charge_species['NaOH'] = 0
# charge_species['NaHCO3'] = 0
# charge_species['Na2CO3'] = 0

# charge_species['Bapp'] = 2
# charge_species['BaOHp'] = 1
# charge_species['BaHCO3p'] = 1
# charge_species['BaCO3'] = 0

# charge_species['H2O'] = 0
# charge_species['CO2g'] = 0 #Check
# charge_species['CaCO3s'] = 0 #Check
# charge_species['BaCO3s'] = 0 #Check

# Database
tags_db = [
    'Hp',
    'OHm',
    'CO2',
    'CO3mm',
    'HCO3m',

    'Nap',
    'NaOH',
    'NaCO3m',
    'NaHCO3',
    'Na2CO3',

    'Capp',
    'CaOHp',
    'CaCO3',
    'CaHCO3p',
    'Clm',

    'NaCl',
    'HCl',

    'Bapp',
    'BaOHp',
    'BaCO3',
    'BaHCO3p',

    'SO4mm',
    'HSO4m',
    'BaSO4',

    'MnSO4',
    'MnHCO3p',
    'Mnpp',
    'MnCO3',
    'MnOHp',

    'CaSO4',
    'CaHSO4p',

    'H2O',
    'CO2g',
    'CaCO3s__Calcite',
    'CaCO3s__Aragonite',
    'BaCO3s__Witherite',
    'BaSO4s__Barite',
    'NaCls__Halite',
    'CaSO4s__Anhydrite'
] #, 'CaCO3s'

Indexes_db = {tag: i for i, tag in enumerate(tags_db)}


#--------------------------------------------
# 	 NUMERICAL HELPERS
#--------------------------------------------
def solve_with_exception(fun, x0, args=None, jac=None):
    if args is not None:
        # kwarg_root = dict(args=(args,), method='hybr',
        #                   jac=jac, options={'maxfev': int(1e7)})
        # kwarg_root = dict(args=(args,), method='lm',
                        #   jac=jac)
        kwarg_root = dict(args=args, method='lm', #'lm',
                          jac=jac,
                        #   options={'factor':0.3}
                          options={'factor':100}
                          )
    else:
        # kwarg_root = dict(method='hybr', jac=jac, options={'maxfev': int(1e5)})
        kwarg_root = dict(method='lm', jac=jac)
    sol = optimize.root(fun, x0, tol=1e-15, **kwarg_root)
    if sol.success != True:
        print(sol.message)
        raise ValueError('Error during reaction speciation calculation')
    return sol


#--------------------------------------------
#	DATABASE SPECIES PROPERTIES
#--------------------------------------------
DB_SIZE = len(tags_db)


