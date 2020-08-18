import numpy as np
from .properties_utils import dtype_ions, DB_SIZE
import numba

# Reference: http://www.aqion.de/site/194
# \Lambda_0 [S cm2/mol]
#TODO: add more ions values
# conductivity_molar_zero = np.empty(1, dtype=dtype_ions)
# conductivity_molar_zero['Na+'] = 50.0
# conductivity_molar_zero['H+'] = 349.0
# conductivity_molar_zero['Ca++'] = 119.1
# conductivity_molar_zero['Cl-'] = 76.2
# conductivity_molar_zero['HCO3-'] = 44.3
# conductivity_molar_zero['NaCO3-'] = 22.0
# conductivity_molar_zero['OH-'] = 197.9
# conductivity_molar_zero['CO3--'] = 143.5
# conductivity_molar_zero['OH-'] = 197.9
# conductivity_molar_zero['CaOH+'] = 0.0
# conductivity_molar_zero['CaHCO3+'] = 19.0
# conductivity_molar_zero['CO2'] = 0.0
# conductivity_molar_zero['CaCO3'] = 0.0
# conductivity_molar_zero['NaOH'] = 0.0
# conductivity_molar_zero['NaHCO3'] = 0.0
# conductivity_molar_zero['Na2CO3'] = 0.0
conductivity_molar_zero = np.empty(DB_SIZE)
conductivity_molar_zero[0] = 349.0  # H+
conductivity_molar_zero[1] = 197.9  # 'OH-'
conductivity_molar_zero[2] = 0.0  # 'CO2'
conductivity_molar_zero[3] = 143.5  # 'CO3--'
conductivity_molar_zero[4] = 44.3  # 'HCO3-'
conductivity_molar_zero[5] = 50.0  # 'Na+'
conductivity_molar_zero[6] = 0.0  # 'NaOH'
conductivity_molar_zero[7] = 22.0  # 'NaCO3-'
conductivity_molar_zero[8] = 0.0  # 'NaHCO3'
conductivity_molar_zero[9] = 0.0  # 'Na2CO3'
conductivity_molar_zero[10] = 119.1  # 'Ca++'
conductivity_molar_zero[11] = 0.0  # 'CaOH+' #Missing
conductivity_molar_zero[12] = 0.0  # 'CaCO3'
conductivity_molar_zero[13] = 19.0  # 'CaHCO3+'
conductivity_molar_zero[14] = 76.2  # 'Cl-'
conductivity_molar_zero[15] = 0.0  # 'NaCl'
conductivity_molar_zero[16] = 0.0  # 'HCl'
conductivity_molar_zero[17] = 127.4  # 'Bapp'
conductivity_molar_zero[18] = 0.0  # 'BaOHp' Missing
conductivity_molar_zero[19] = 0.0  # 'BaCO3'
conductivity_molar_zero[20] = 19.0  # 'BaHCO3+' -> Missing: using CaHCO3+
conductivity_molar_zero[21] = 0.0  # 'H2O'
conductivity_molar_zero[22] = 0.0  # 'CO2(g)' - dummy
conductivity_molar_zero[23] = 0.0  # 'CaCO3s' - dummy
conductivity_molar_zero[24] = 0.0  #  BaCO3s' - dummy


# for i, tag in enumerate(ions_tags_array):
#     conductivity_molar_zero[tag] = conductivity_molar_zero_array[i]


# @numba.njit()
# def solution_conductivity_ideal(conc_vals):
#     """
#     Ideal conductivity calculation

#     Ref: http://www.aqion.de/site/77#Nernst-Einstein

#     Parameters
#     ----------
#     conc_vals : [dic, np.records with ions tag]
#         Ions Concentrations in mol/L

#     Returns
#     -------
#     [float]
#         Conductivity in S/cm
#     """
#     aux = 0.0
#     # for tag in conc_vals.dtype.names:
#     for tag in conc_vals:
#         aux += conc_vals[tag] * conductivity_molar_zero[tag]
#     aux *= 1e-3
#     return aux

# def solution_conductivity_(I, gamma, conc_vals):
#     """
#     Solution non-ideal conductivity calculation

#     Ref: http://www.aqion.de/site/77#fn:Appelo

#     Parameters
#     ----------
#     I : [float]
#         Ionic strength in mol/L
#     gamma : [type]
#         Activity coefficients
#     conc_vals : [dic, np.records with ions tag]
#         Ions Concentrations in mol/L

#     Returns
#     -------
#     [float]
#         Conductivity in S/cm
#     """
#     ret = 0.0
#     for tag in conc_vals:
#         if charge_species[tag] == 0:
#             continue
#         if I < 0.36*charge_species[tag]:
#             alpha = 0.6*np.sqrt(charge_species[tag])
#         else:
#             alpha = np.sqrt(I) * charge_species[tag]
#         aux = conductivity_molar_zero[tag] * gamma[tag]**alpha * conc_vals[tag]
#         ret += aux
#     ret *= 1e-3
#     return ret

# @numba.njit
def solution_conductivity(I, gamma, conc_vals, charges, cond_molar_zero):
    """
    Solution non-ideal conductivity calculation

    Ref: http://www.aqion.de/site/77#fn:Appelo

    Parameters
    ----------
    I : [float]
        Ionic strength in mol/L
    gamma : [np.ndarray]
        Activity coefficients
    conc_vals : [np.ndarray(float)]
        Ions Concentrations in mol/L
    charges : [np.ndarray(int)]
        Species Charge
    cond_molar_zero : [np.ndarray(float)]
        Species ideal molar conductivity
    Returns
    -------
    [float]
        Conductivity in S/cm
    """
    ret = 0.0
    # for tag in conc_vals:
    #     if charge_species[tag] == 0:
    #         continue
    #     if I < 0.36 * charge_species[tag]:
    #         alpha = 0.6 * np.sqrt(charge_species[tag])
    #     else:
    #         alpha = np.sqrt(I) * charge_species[tag]
    #     aux = conductivity_molar_zero[tag] * gamma[tag]**alpha * conc_vals[tag]
    #     ret += aux
    # ret *= 1e-3

    for i in np.arange(len(conc_vals)):
        if charges[i] == 0:
            continue
        if I < 0.36 * charges[i]:
            alpha = 0.6 / np.sqrt(charges[i])
        else:
            alpha = np.sqrt(I) / charges[i]
        aux = cond_molar_zero[i] * gamma[i]**alpha * conc_vals[i]
        ret += aux
    ret *= 1e-3
    return ret

