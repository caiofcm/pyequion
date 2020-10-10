import numpy as np
from numpy import pi


def ionicstr(m, z):
    # Função que calcula a força iônica I e o parâmetro Z
    ionic = 0.5 * np.sum(m * (z ** 2))
    zg = np.sum(np.abs(z) * m)
    return ionic, zg


def afi(t):
    # Função que calcula o parâmetro de Debye-Huckel Afi
    e = 1.60218 * (10 ** (-19))
    na = 6.023 * (10 ** 23)
    # ep0 = 8.85419 * (10 ** (-12))
    kb = 1.38064852 * (10 ** (-23))
    # epr = 78.41
    ds = 997.00
    # dieletric = 305.7*np.exp(-np.exp(-12.741+0.01875*t)-(t/219))
    dieletric = 0.24921e3 - 0.79069 * t + 0.72997e-3 * t ** 2
    # ds = -58.05 - 0.01098*t + 3053/t - 1.75e5/t**2 + 21.8*np.log10(t)
    # a = (1.0/3.0)*np.sqrt(2.0*pi*na*ds)*(((e*e)/(4.0*pi*ep0*epr*kb*t))**(3.0/2.0))
    a = (
        (1.0 / 3)
        * ((e / (np.sqrt(dieletric * kb * t))) ** 3)
        * np.sqrt(2 * pi * ds * na / 1000)
    )
    return a


def funb(ionic, beta, zc, za):
    # Função que calcula o parâmetro B e a sua derivada a partir dos parâmetros ajustados beta
    isp = np.sqrt(ionic)

    if zc == 2 and za == -2:
        alfa = np.array([1.4, 12.0])
    else:
        alfa = np.array([2.0, 1e-9])
    x1 = alfa[0] * isp
    x2 = alfa[1] * isp
    par1 = beta[1] * np.exp(-x1)
    par2 = beta[2] * np.exp(-x2)
    bfi = beta[0] + par1 + par2
    return bfi


def func(cfi, zc, za):
    # Função que calcula o parâmetro C a partir do Cfi ajustável
    c = cfi / (2 * (np.sqrt(np.abs(zc * za))))
    return c


# def funj(x, xlinha):
#
#     # Equações tiradas de Kaasa cap 2
#     a = 4.581
#     b = 0.7237
#     c = 0.0120
#     d = 0.528
#
#     j = x / (4 + (a / (x ** b)) * np.exp(c * (x ** d)))
#     # par1 = 4 + a*np.exp(c*(x**d))/(x**b)
#     # par2 = (x*a*np.exp(c*(x**d)))*((c*d*(x**(b+d-1))-(b*(x**(b-1))))/(x**(2*b)))
#     # par3 = par1**2
#
#     par1 = (x**b)*np.exp(c*(x**d))
#     par2 = a*b + a*c*d*(x**d) + a + 4*(x**b)*np.exp(c*(x**d))
#     par3 = (a+4*(x**b)*np.exp(c*(x**d)))**2
#     jlinha = xlinha*(par1-par2)/par3
#
#     return j, jlinha


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

    ak = np.zeros((21, 2), dtype=np.float)
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

    phiphi = phi + ionic * phil
    return phiphi


#
# def funphi(zc, zcc, ionic, a, theta):
#
#     # Equações tiradas de Kaasa cap 2
#     # zcc é o número atômico do ânion se for interação entre cátions e vice-versa
#
#     xca = 6*zc*zcc*a*np.sqrt(ionic)
#     xlca = 3*zc*zcc*a*(ionic**(-0.5))
#
#     xcc = 6*zc*zc*a*np.sqrt(ionic)
#     xlcc = 3*zc*zc*a*(ionic**(-0.5))
#
#     xaa = 6*zcc*zcc*a*np.sqrt(ionic)
#     xlaa = 3*zcc*zcc*a*(ionic**(-0.5))
#
#     jca, jlca = funj(xca, xlca)
#     jcc, jlcc = funj(xcc, xlcc)
#     jaa, jlaa = funj(xaa, xlaa)
#
#     #
#     # if zc + za == 0:
#     #     etheta = 0
#     #     ethetal = 0
#     # else:
#     #etheta = (zc*zcc/(4*ionic))*(jca-0.5*jcc-0.5*jaa)
#     etheta = (2 / (4 * ionic)) * (jca - 0.5 * jcc - 0.5 * jaa)
#     ethetal = (zc*zcc/(4*ionic))*(jlca-0.5*jlcc-0.5*jlaa) - (zc*zcc/(4*(ionic**2)))*(jca-0.5*jcc-0.5*jaa)
#     # ethetal = -(etheta/ionic) + (zc*zcc/(8*(ionic**2)))*(xca*jlca - 0.5*xcc*jlcc - 0.5*xaa*jlaa)
#     phi = theta + etheta
#     phil = ethetal
#     phiphi = phi + ionic*phil
#     return phiphi


def activitywater(
    mc, ma, mn, zc, za, beta, cfi, thetac, thetaa, lambdanc, lambdana, t
):

    z = np.concatenate((zc, za), axis=None)
    m = np.concatenate((mc, ma), axis=None)

    ionic, zg = ionicstr(m, z)
    a = afi(t)
    b = 1.2

    c = np.zeros((len(mc), len(ma)), dtype=np.float)
    bfi = np.zeros((len(mc), len(ma)), dtype=np.float)
    k = 0
    for i in range(0, len(mc)):
        for j in range(0, len(ma)):
            c[i, j] = func(cfi[i, j], zc[i], za[j])
            bfi[i, j] = funb(ionic, beta[k, :], zc[i], za[j])
            k = k + 1

    phiphic = np.zeros((len(mc), len(mc)), dtype=np.float)
    phiphia = np.zeros((len(ma), len(ma)), dtype=np.float)

    for i in range(0, len(mc) - 1):
        for j in range(0, len(mc)):
            if i == j:
                phiphic[i, j] = 0
                phiphia[i, j] = 0
            else:
                phiphic[i, j] = funphi(zc[i], zc[j], ionic, a, thetac[i, j])
                phiphia[i, j] = funphi(za[i], za[j], ionic, a, thetaa[i, j])

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
