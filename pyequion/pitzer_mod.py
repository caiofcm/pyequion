"""
Implementing Pitzer Model from Reference FLOWBAT thesis
"""
import numpy as np
from .activity_coefficients import debye_huckel_constant

b = 1.2

def calc_loggamma(m, z, TK, I, beta0, beta1, beta2, C0):
    # C_ij = Cphi_ij/(2*np.sqrt())
    alpha1 = 2.0 #back here, univalent etc
    alpha2 = 0.0
    sqI = np.sqrt(I)
    Am, _ = debye_huckel_constant(TK) #checkme
    f_gamma = -2*Am*(I/(1+1.2*sqI)) + 2/1.2 * np.log(1+1.2*sqI)

    B = beta0 + \
        2*beta1/(alpha1**2*I)*(1.0 - (1.0+alpha1*sqI)*np.exp(-alpha1*sqI)) +\
        2*beta2/(alpha2**2*I)*(1.0 - (1.0+alpha2*sqI)*np.exp(-alpha2*sqI))

    Bline = 2*beta1/(alpha1**2*I**2)*((-1) + (1+alpha1*sqI+0.5*alpha1**2*I)*np.exp(-alpha1*sqI)) + \
            2*beta2/(alpha2**2*I**2)*((+1) - (1+alpha2*sqI+0.5*alpha2**2*I)*np.exp(-alpha2*sqI))

    Zij = z.reshape((-1,1)) @ z
    Cij = C0/(2*np.sqrt(Zij))

    # Ion Activity Coeff.
    sum_mz = np.sum(m*np.abs(z))
    m_times_m = m.reshape((-1,1)) @ m
    mmBlinha = m_times_m * Bline
    sum_mmB = np.sum(mmBlinha)
    mmC = m_times_m * Cij
    sum_mmC = np.sum(mmC)
    i = 0
    p0 = z[i]**2/2*f_gamma
    p1 = 2*np.sum(m*B[i,:])
    p2 = np.sum(sum_mz * m * Cij[i,:])
    p3 = z[i]**2/2.0 * sum_mmB
    p4 = np.abs(z[i])/2.0 * sum_mmC
    ln_g_ion = p0 + p1 + p2 + p3 + p4

    # Solvents Activity Coeff.
    Ms = 18 #FIXME
    s = 0
    p0 = Am * 4*I/1.2 * np.log(1+b*sqI)
    p1 = -I*f_gamma
    p2_0 = B + Bline.T * I
    p2_1 = m_times_m * p2_0
    p2 = -np.sum(p2_1)
    p3_0 = sum_mz * m_times_m * Cij
    p3 = -np.sum(p3_0)
    ln_g_solv = p0 + p1 + p2 + p3

    return ln_g_ion, ln_g_solv

def run1():



    return

if __name__ == "__main__":
    run1()
