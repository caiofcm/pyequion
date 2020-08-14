import numpy as np
from scipy import optimize


def fugacidade(T, P, Tc=304.2, Pc=73.82, w=0.239):

    Tr = T / Tc
    R = 83.144621
    sigma = 1 + np.sqrt(2)
    epsilon = 1 - np.sqrt(2)
    kapa = 0.37464 + 1.54226 * w - (0.26992 * (w ** 2))
    f = (1 + kapa * (1 - Tr ** 0.5)) ** 2
    b = 0.07780 * R * Tc / Pc
    a = 0.45724 * (R ** 2) * (Tc ** 2) * f / Pc

    A = a*P/((R**2)*(T**2))
    B = b*P/(R*T)

    # Coeficientes da cúbica
    c3 = 1
    c2 = -(1-B)
    c1 = A-(3*B**2)-2*B
    c0 = -(A*B-B**2-B**3)

    # Achando as raízes da cúbica
    coeficientes = [c3, c2, c1, c0]
    raiz = np.roots(coeficientes)
    Zz = np.zeros([3])

    # Escolhendo a raíz do gás
    for i in range(0, len(raiz)):
        if raiz[i].imag == 0.0:
            Zz[i] = raiz[i].real

    Z = np.max(Zz)

    # beta = b*P/(R*T)
    # q = a/(b*R*T)
    #
    # def fun(Z, sigma, epsilon, beta, q):
    #     resp = 1 + beta - q*beta*((Z-beta)/((Z+epsilon*beta)*(Z+sigma*beta))) - Z
    #     return resp
    #
    # Zo = 1
    # Z = optimize.fsolve(fun, Zo, args=(sigma, epsilon, beta, q))
    #
    # I = (1/(sigma-epsilon))*np.log((Z+sigma*beta)/(Z+epsilon*beta))
    #
    # lnfi = Z - 1 - np.log(Z - beta) - q*I

    lnfi = Z - 1 - np.log(Z-B) - (A/(2*np.sqrt(2)*B))*np.log((Z+sigma*B)/(Z+epsilon*B))

    return lnfi

