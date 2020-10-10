import numpy as np
from numpy import pi

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# @numba.njit
def ionicstr(m, z):
    # Função que calcula a força iônica I e o parâmetro Z
    ionic = 0.5 * np.sum(m * (z ** 2))
    zg = np.sum(np.abs(z) * m)  # CHECK
    return ionic, zg


# @numba.njit
def afi(T):
    # Função que calcula o parâmetro de Debye-Huckel Afi
    # na = 6.023 * (10 ** 23)
    # ep0 = 8.85419 * (10 ** (-12))
    # # e = 1.60218 * (10 ** (-19)) #SI
    # # kb = 1.38064852 * (10 ** (-23)) #SI
    # e = 4.8029e-10 #as older
    # kb = 1.38045e-16 #as older
    # epr = 78.41
    # ds = 997.00

    Na = 6.0232e23
    ee = 4.8029e-10
    k = 1.38045e-16

    # CFCM - PitzerNaCl
    ds = -0.0004 * T + 1.1188
    eer = 305.7 * np.exp(
        -np.exp(-12.741 + 0.01875 * T) - T / 219.0
    )  # Zemaitis pg 644
    Aphi = (
        1.0
        / 3.0
        * (2.0 * np.pi * Na * ds / 1000) ** 0.5
        * (ee / (eer * k * T) ** 0.5) ** 3.0
    )

    # As In WaterActivity
    # a = (1.0/3.0)*np.sqrt(2.0*pi*na*ds)*(((e*e)/(4.0*pi*ep0*epr*kb*t))**(3.0/2.0))

    # As Previously
    # dieletric = 0.24921e3 - 0.79069*t + 0.72997e-3*t**2
    # ds = -58.05 - 0.01098*t + 3053/t - 1.75e5/t**2 + 21.8*np.log10(t)
    # a = (1.0/3.0)*np.sqrt(2.0*pi*na*ds)*(((e*e)/(4.0*pi*ep0*epr*kb*t))**(3.0/2.0))
    # a = (1.0/3)*((e/(np.sqrt(dieletric*kb*t)))**3)*np.sqrt(2*pi*ds*na/1000)

    # Aphi = 0.392 from G-A
    return Aphi


# @numba.njit
def funb(ionic, beta, zc, za):
    # Função que calcula o parâmetro B e a sua derivada a partir dos parâmetros ajustados beta
    isp = np.sqrt(ionic)

    if zc == 2 and za == -2:
        alfa = np.array([1.4, 12.0])
        x1 = alfa[0] * isp
        x2 = alfa[1] * isp
        par1 = beta[1] * (2 * (1 - (1 + x1) * np.exp(-x1))) / (x1 ** 2)
        par2 = beta[2] * (2 * (1 - (1 + x2) * np.exp(-x2))) / (x2 ** 2)
        par3 = (
            beta[1]
            * (-2 * (1 - (1 + x1 + 0.5 * x1 * x1) * np.exp(-x1)))
            / (x1 ** 2)
        )
        par4 = (
            beta[2]
            * (-2 * (1 - (1 + x2 + 0.5 * x2 * x2) * np.exp(-x2)))
            / (x2 ** 2)
        )
    else:
        alfa = np.array([2.0, 0.0])
        x1 = alfa[0] * isp
        par1 = beta[1] * (2 * (1 - (1 + x1) * np.exp(-x1))) / (x1 ** 2)
        par2 = 0.0
        par3 = (
            beta[1]
            * (-2 * (1 - (1 + x1 + 0.5 * x1 * x1) * np.exp(-x1)))
            / (x1 ** 2)
        )
        par4 = 0.0
    bb = beta[0] + par1 + par2
    dbb = (1 / ionic) * (par3 + par4)
    return bb, dbb


# @numba.njit
def func(cfi, zc, za):
    # Função que calcula o parâmetro C a partir do Cfi ajustável
    c = cfi / (2 * (np.sqrt(abs(zc * za))))
    return c


# @numba.njit
def funphi(zc, zcc, ionic, a, theta):

    # Equações tiradas de George Anderson

    w = np.array(
        [
            1.925154014814667,
            -0.060076477753119,
            -0.029779077456514,
            -0.007299499690937,
            0.000388260636404,
            0.000636874599598,
            0.000036583601823,
            -0.000045036975204,
            -0.000004537895710,
            0.000002937706971,
            0.000000396566462,
            -0.000000202099617,
            -0.000000025267769,
            0.000000013522610,
            0.000000001229405,
            -0.000000000821969,
            -0.000000000050847,
            0.000000000046333,
            0.000000000001943,
            -0.000000000002563,
            -0.000000000010991,
        ]
    )

    b = np.array(
        [
            0.628023320520852,
            -0.028796057604906,
            0.006519840398744,
            -0.000242107641309,
            -0.000004583768938,
            0.000000216991779,
            -0.000000006944757,
            0.462762985338493,
            0.150044637187895,
            -0.036552745910311,
            -0.001668087945272,
            0.001130378079086,
            -0.000887171310131,
            0.000087294451594,
            0.000034682122751,
            -0.000003548684306,
            -0.000000250453880,
            0.000000080779570,
            0.000000004558555,
            -0.000000002849257,
            0.000000000237816,
        ]
    )

    ak = np.zeros((21, 2), dtype=np.float64)
    for i in range(0, len(w)):
        ak[i, 0] = w[i]
        ak[i, 1] = b[i]

    xca = 6.0 * zc * zcc * a * np.sqrt(ionic)
    xcc = 6.0 * zc * zc * a * np.sqrt(ionic)
    xaa = 6.0 * zcc * zcc * a * np.sqrt(ionic)

    bk = np.zeros(23)
    dk = np.zeros(23)

    for k in range(1, 4):

        if k == 1:
            x = xca
        elif k == 2:
            x = xcc
        else:
            x = xaa

        if x <= 1:
            z = 4.0 * (x ** 0.2) - 2.0
            dzdx = 0.8 * x ** (-0.8)

            m = 20
            while m >= 0:
                bk[m] = z * bk[m + 1] - bk[m + 2] + ak[m, 0]
                dk[m] = bk[m + 1] + z * dk[m + 1] - dk[m + 2]
                m = m - 1
        else:
            z = (40.0 / 9.0) * x ** (-0.1) - (22.0 / 9.0)
            dzdx = -(40.0 / 90.0) * x ** (-1.1)

            m = 20
            while m >= 0:
                bk[m] = z * bk[m + 1] - bk[m + 2] + ak[m, 1]
                dk[m] = bk[m + 1] + z * dk[m + 1] - dk[m + 2]
                m = m - 1
        if k == 1:
            jca = 0.25 * x - 1.0 + 0.5 * (bk[0] - bk[2])
            jlca = 0.25 + 0.5 * dzdx * (dk[0] - dk[2])
        elif k == 2:
            jcc = 0.25 * x - 1.0 + 0.5 * (bk[0] - bk[2])
            jlcc = 0.25 + 0.5 * dzdx * (dk[0] - dk[2])
        else:
            jaa = 0.25 * x - 1.0 + 0.5 * (bk[0] - bk[2])
            jlaa = 0.25 + 0.5 * dzdx * (dk[0] - dk[2])

    etheta = (2 / (4 * ionic)) * (jca - 0.5 * jcc - 0.5 * jaa)
    ethetal = -(etheta / ionic) + (zc * zcc / (8 * (ionic ** 2))) * (
        xca * jlca - 0.5 * xcc * jlcc - 0.5 * xaa * jlaa
    )

    phi = theta + etheta
    phil = ethetal

    return phi, phil


# @numba.njit
def funf(mc, ma, ionic, a, dbb, dphic, dphia):

    # Os parâmetros dbb, dphic e dphia desta função é da interação entre todos os cátions e ânions,
    # que serão calculados no programa principal

    n = len(mc)
    m = len(ma)
    b = 1.2
    par1 = 0
    par2 = 0
    par3 = 0

    # Cálculo do somatório entre cátions e ânions
    for i in range(0, n):
        for j in range(0, m):
            par1 += mc[i] * ma[j] * dbb[i, j]

    # Cálculo do somatório entre cátions
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            par2 += mc[i] * mc[j] * dphic[i, j]

    # Cálculo do somatório entre ânions
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            par3 += ma[i] * ma[j] * dphia[i, j]

    # Cálculo do primeiro fator da equação
    isp = np.sqrt(ionic)
    lnpar = np.log(1 + b * isp)
    par4 = -a * ((isp / (1 + b * isp)) + (2 / b) * lnpar)
    f = par4 + par1 + par2 + par3
    return f


# @numba.njit
def loggammam(
    mc,
    ma,
    mn,
    zc,
    za,
    beta,
    cfi,
    thetac,
    thetaa,
    psiMca,
    psiMaa,
    lambdanM,
    t,
    ic,
    skip_ternary=False,
):

    # (mc, ma, mn, zc, za, beta, cfi, thetac, thetaa, psi1, psi2, psi3, psi4, lamb1, lamb2, t, ic, ia)

    # ic e ia são os índices do elemento do array do sal desejado, e da linha  ou coluna da matriz do sal desejado

    # za e zc são arrays

    # O parâmetro bb e o parâmetro c são as interações entre todos os cátions e do ânions
    # psi1 - psiMca; psi2 - psiMaa'; psi3 - psiXac; psi4 - psiXcc'

    # lamb1 - lambdanM; lamb2 - lambdanX

    # A partir desta parte para teste
    z = np.concatenate((zc, za))
    m = np.concatenate((mc, ma))

    ionic, zg = ionicstr(m, z)
    a = afi(t)

    c = np.zeros((len(mc), len(ma)), dtype=np.float64)
    bb = np.zeros((len(mc), len(ma)), dtype=np.float64)
    dbb = np.zeros((len(mc), len(ma)), dtype=np.float64)
    k = 0
    for i in range(0, len(mc)):
        for j in range(0, len(ma)):
            c[i, j] = func(cfi[i, j], zc[i], za[j])
            bb[i, j], dbb[i, j] = funb(ionic, beta[k, :], zc[i], za[j])
            k = k + 1

    phic = np.zeros((len(mc), len(mc)), dtype=np.float64)
    dphic = np.zeros((len(mc), len(mc)), dtype=np.float64)

    phia = np.zeros((len(ma), len(ma)), dtype=np.float64)
    dphia = np.zeros((len(ma), len(ma)), dtype=np.float64)

    for i in range(0, len(mc)):
        for j in range(0, len(mc)):
            if i == j:
                phic[i, j] = 0.0
                dphic[i, j] = 0.0
            else:
                phic[i, j], dphic[i, j] = funphi(
                    zc[i], zc[j], ionic, a, thetac[i, j]
                )

    for i in range(0, len(ma)):
        for j in range(0, len(ma)):
            if i == j:
                phia[i, j] = 0.0
                dphia[i, j] = 0.0
            else:
                phia[i, j], dphia[i, j] = funphi(
                    za[i], za[j], ionic, a, thetaa[i, j]
                )

    # for i in range(0, len(mc)-1):
    #     for j in range(0, len(mc)):
    #         if i == j:
    #             phic[i, j] = 0
    #             dphic[i, j] = 0
    #             phia[i, j] = 0
    #             dphia[i, j] = 0
    #         else:
    #             phic[i, j], dphic[i, j] = funphi(zc[i], zc[j], ionic, a, thetac[i, j])
    #             phia[i, j], dphia[i, j] = funphi(za[i], za[j], ionic, a, thetaa[i, j])
    f = funf(mc, ma, ionic, a, dbb, dphic, dphia)

    # Calculando o bb e o c da interação entre o cátion desejado e os outros ânios e vice-versa
    bma = np.zeros(len(ma))  # CM-MODIFIED
    cma = np.zeros(len(ma))

    for i in range(0, len(ma)):
        bma[i] = bb[ic, i]
        cma[i] = c[ic, i]

    # termo1 = 0
    # for i in range(0, len(ma)):
    #     for j in range(0, len(mc)):
    #         termo1 += ma[i]*(2*bma[j] + zg*cma[j])

    termo1 = 0.0
    for i in range(0, len(ma)):
        termo1 += ma[i] * (2.0 * bma[i] + zg * cma[i])

    termo2 = 0.0
    if not skip_ternary:
        for i in range(1, len(mc)):
            aux = 0.0
            for j in range(0, len(ma)):
                aux = aux + ma[j] * psiMca[j]
            termo2 = termo2 + mc[i] * (2 * phic[ic, i] + aux)

    termo3 = 0.0
    if not skip_ternary:
        for i in range(0, len(ma) - 1):
            for j in range(i + 1, len(ma)):
                termo3 = termo3 + ma[i] * ma[j] * psiMaa[j - 1]

    termo4 = 0.0
    for i in range(0, len(mc)):
        for j in range(0, len(ma)):
            termo4 = termo4 + mc[i] * ma[j] * c[i, j]
    termo4 = termo4 * zc[ic]

    termo5 = 0.0
    for i in range(0, len(mn)):
        termo5 = termo5 + mn[i] * lambdanM[i, ic]
    termo5 = termo5 * 2.0

    lngammam = (zc[ic] ** 2) * f + termo1 + termo2 + termo3 + termo4 + termo5

    return lngammam


# @numba.njit
def loggammax(
    mc,
    ma,
    mn,
    zc,
    za,
    beta,
    cfi,
    thetac,
    thetaa,
    psiXac,
    psiXcc,
    lambdanX,
    t,
    ia,
    skip_ternary=False,
):

    # (mc, ma, mn, zc, za, beta, cfi, thetac, thetaa, psi1, psi2, psi3, psi4, lamb1, lamb2, t, ic, ia)

    # ic e ia são os índices do elemento do array do sal desejado, e da linha  ou coluna da matriz do sal desejado

    # za e zc são arrays

    # O parâmetro bb e o parâmetro c são as interações entre todos os cátions e do ânions
    # psi1 - psiMca; psi2 - psiMaa'; psi3 - psiXac; psi4 - psiXcc'

    # lamb1 - lambdanM; lamb2 - lambdanX

    # A partir desta parte para teste
    z = np.concatenate((zc, za))
    m = np.concatenate((mc, ma))

    ionic, zg = ionicstr(m, z)
    a = afi(t)

    c = np.zeros((len(mc), len(ma)), dtype=np.float64)
    bb = np.zeros((len(mc), len(ma)), dtype=np.float64)
    dbb = np.zeros((len(mc), len(ma)), dtype=np.float64)
    k = 0
    for i in range(0, len(mc)):
        for j in range(0, len(ma)):
            c[i, j] = func(cfi[i, j], zc[i], za[j])
            bb[i, j], dbb[i, j] = funb(ionic, beta[k, :], zc[i], za[j])
            k = k + 1

    phic = np.zeros((len(mc), len(mc)), dtype=np.float64)
    dphic = np.zeros((len(mc), len(mc)), dtype=np.float64)

    phia = np.zeros((len(ma), len(ma)), dtype=np.float64)
    dphia = np.zeros((len(ma), len(ma)), dtype=np.float64)

    for i in range(0, len(mc)):
        for j in range(0, len(mc)):
            if i == j:
                phic[i, j] = 0.0
                dphic[i, j] = 0.0
            else:
                phic[i, j], dphic[i, j] = funphi(
                    zc[i], zc[j], ionic, a, thetac[i, j]
                )

    for i in range(0, len(ma)):
        for j in range(0, len(ma)):
            if i == j:
                phia[i, j] = 0.0
                dphia[i, j] = 0.0
            else:
                phia[i, j], dphia[i, j] = funphi(
                    za[i], za[j], ionic, a, thetaa[i, j]
                )

    # for i in range(0, len(mc) - 1):
    #     for j in range(0, len(mc)):
    #         if i == j:
    #             phic[i, j] = 0
    #             dphic[i, j] = 0
    #             phia[i, j] = 0
    #             dphia[i, j] = 0
    #         else:
    #             phic[i, j], dphic[i, j] = funphi(zc[i], zc[j], ionic, a, thetac[i, j])
    # phia[i, j], dphia[i, j] = funphi(za[i], za[j], ionic, a, thetaa[i, j])
    f = funf(mc, ma, ionic, a, dbb, dphic, dphia)

    # Calculando o bb e o c da interação entre o cátion desejado e os outros ânios e vice-versa
    bcx = np.zeros(len(mc))
    ccx = np.zeros(len(mc))

    for i in range(0, len(mc)):
        bcx[i] = bb[i, ia]
        ccx[i] = c[i, ia]

    # for i in range(0, len(mc)):
    #     bcx[i] = bb[i, ia]
    #     ccx[i] = c[i, ia]

    termo6 = 0.0
    for i in range(0, len(mc)):
        termo6 += mc[i] * (
            2 * bcx[i] + zg * ccx[i]
        )  # ma[i]*(2.0*bma[i] + zg*cma[i])
    # termo6 = 0
    # for i in range(0, len(mc)):
    #     for j in range(0, len(ma)):
    #         termo6 = termo6 + mc[i]*(2*bcx[j] + zg*ccx[j])

    termo7 = 0.0
    if not skip_ternary:
        for i in range(0, len(ma)):
            aux = 0
            for j in range(0, len(mc)):
                aux = aux + mc[j] * psiXac[j]
            termo7 = termo7 + ma[i] * (2 * phia[ia, i] + aux)

    termo8 = 0.0
    if not skip_ternary:
        for i in range(0, len(mc) - 1):
            for j in range(i + 1, len(mc)):
                termo8 = termo8 + mc[i] * mc[j] * psiXcc[j - 1]

    termo9 = 0.0
    for i in range(0, len(mc)):
        for j in range(0, len(ma)):
            termo9 = termo9 + mc[i] * ma[j] * c[i, j]
    termo9 = termo9 * abs(za[ia])

    termo10 = 0.0
    for i in range(0, len(mn)):
        termo10 = termo10 + mn[i] * lambdanX[i, ia]
    termo10 = termo10 * 2

    lngammax = (za[ia] ** 2) * f + termo6 + termo7 + termo8 + termo9 + termo10

    return lngammax


# @numba.njit
def logneutro(mc, ma, lambdanc, lambdana, epnca, i_n, skip_ternary=True):

    termo1 = 0.0
    for i in range(0, len(mc)):
        termo1 = termo1 + mc[i] * lambdanc[i_n, i]
    termo1 = 2 * termo1

    termo2 = 0.0
    for i in range(0, len(ma)):
        termo2 = termo2 + ma[i] * lambdana[i_n, i]
    termo2 = 2 * termo2

    termo3 = 0.0
    if not skip_ternary:
        for i in range(0, len(mc)):
            for j in range(0, len(ma)):
                termo3 = termo3 + mc[i] * ma[j] * epnca[i, j]

    loggamman = termo1 + termo2 + termo3

    return loggamman


# @numba.njit
def logmedio(lngammam, lngammax, zc, za):

    nua = zc
    nuc = np.abs(za)
    nu = nuc + nua
    lnmx = (1 / nu) * (nuc * lngammam + nua * lngammax)

    return lnmx


# @numba.njit
def activitywater(
    mc, ma, mn, zc, za, beta, cfi, thetac, thetaa, lambdanc, lambdana, t
):

    z = np.concatenate((zc, za))
    m = np.concatenate((mc, ma))

    ionic, zg = ionicstr(m, z)
    a = afi(t)
    b = 1.2

    c = np.zeros((len(mc), len(ma)))
    bfi = np.zeros((len(mc), len(ma)))
    k = 0
    for i in range(0, len(mc)):
        for j in range(0, len(ma)):
            c[i, j] = func(cfi[i, j], zc[i], za[j])
            bfi[i, j], _ = funb(ionic, beta[k, :], zc[i], za[j])
            k = k + 1

    phiphic = np.zeros((len(mc), len(mc)))
    phiphia = np.zeros((len(ma), len(ma)))

    for i in range(0, len(mc)):
        for j in range(0, len(mc)):
            if i == j:
                phiphic[i, j] = 0.0
            else:
                phiphic[i, j], _ = funphi(zc[i], zc[j], ionic, a, thetac[i, j])

    for i in range(0, len(ma)):
        for j in range(0, len(ma)):
            if i == j:
                phiphia[i, j] = 0.0
            else:
                phiphia[i, j], _ = funphi(za[i], za[j], ionic, a, thetaa[i, j])

    termo1 = -a * (ionic ** 1.5) / (1 + b * (ionic ** 0.5))

    termo2 = 0.0
    for i in range(0, len(mc)):
        for j in range(0, len(ma)):
            termo2 += mc[i] * ma[j] * (bfi[i, j] + zg * c[i, j])

    termo3 = 0.0
    for i in range(0, len(mn)):
        for j in range(0, len(ma)):
            termo3 += mn[i] * ma[j] * lambdana[i, j]

    termo4 = 0.0
    for i in range(0, len(mn)):
        for j in range(0, len(mc)):
            termo4 += mn[i] * mc[j] * lambdanc[i, j]
    # Sem considerar as interações triplas
    termo5 = 0.0
    for i in range(0, len(ma) - 1):
        for j in range(i + 1, len(ma)):
            termo5 += ma[i] * ma[j] * phiphia[i, j]
    termo6 = 0.0
    for i in range(0, len(mc) - 1):
        for j in range(i + 1, len(mc)):
            termo6 += mc[i] * mc[j] * phiphic[i, j]

    # mcomp é o vetor das concentrações de todos os componentes

    soma = np.sum(m)
    osmotic = (2 / soma) * (
        termo1 + termo2 + termo3 + termo4 + termo5 + termo6
    ) + 1

    lnactivity = -(18.0154 / 1000.0) * osmotic * soma

    return lnactivity


# -------------------------------------------------
# -------------------------------------------------
#    Compiling IF Numba is present
# -------------------------------------------------
# -------------------------------------------------
if HAS_NUMBA:
    ionicstr = numba.njit()(ionicstr)
    afi = numba.njit()(afi)
    funb = numba.njit()(funb)
    func = numba.njit()(func)
    funphi = numba.njit()(funphi)
    funf = numba.njit()(funf)
    loggammam = numba.njit()(loggammam)
    loggammax = numba.njit()(loggammax)
    logneutro = numba.njit()(logneutro)
    logmedio = numba.njit()(logmedio)
    activitywater = numba.njit()(activitywater)
