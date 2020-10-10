import numpy as np

# Reference: http://www.aqion.de/site/194


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
    for i in np.arange(len(conc_vals)):
        if charges[i] == 0:
            continue
        if I < 0.36 * charges[i]:
            alpha = 0.6 / np.sqrt(np.abs(charges[i]))  # CHECKME
        else:
            alpha = np.sqrt(I) / charges[i]
        aux = cond_molar_zero[i] * gamma[i] ** alpha * conc_vals[i]
        ret += aux
    ret *= 1e-3
    return ret
