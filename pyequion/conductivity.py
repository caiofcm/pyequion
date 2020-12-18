import numpy
import sympy

global np
np = numpy

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
    for i in range(len(conc_vals)):
        if charges[i] == 0:
            continue
        if numpy.isscalar(I) or hasattr(
            I, "NodeAsPlainText"
        ):  # Logic for Numpy or adouble from daetools
            if I < 0.36 * abs(charges[i]):
                alpha = 0.6 / np.sqrt(abs(charges[i]))  # CHECKME
            else:
                alpha = np.sqrt(I) / abs(charges[i])
        else:  # Symbolic
            print(I.__dir__())
            alpha = sympy.Piecewise(
                (0.6 / np.sqrt(abs(charges[i])), I < 0.36 * abs(charges[i])),
                (np.sqrt(I) / abs(charges[i]), True),
            )
        # alpha = np.sqrt(I) / abs(charges[i])  # checking symbolic generation of solution
        aux = cond_molar_zero[i] * gamma[i] ** alpha * conc_vals[i]
        ret += aux
    ret *= 1e-3
    return ret
