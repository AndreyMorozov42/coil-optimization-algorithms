import numpy as np

def self_inductance_turn(r, s):
    # mu0 = 4 * np.pi * 10 ** (-7)
    mu0 = 1.26e-6
    a = np.log(8 * r / s)
    b = s ** 2 / (8 * r ** 2)
    return mu0 * r * (a - 7 / 4 + b * (a + 1 / 3))

def mutual_inductance(coil_1, coil_2, d, ro):
    N = K = 30
    fi = 0
    # mu0 = 4 * np.pi * 10 ** (-7)
    mu0 = 1.26e-6
    n = np.arange(N)
    k = n.reshape((K, 1))
    df1 = 2 * np.pi / N
    df2 = 2 * np.pi / K
    mi_ro = np.array([])
    for r in ro:
        mutual_inductance = np.ones((len(coil_1), len(coil_2)))
        for ri in range(len(coil_1)):
            for rj in range(len(coil_2)):
                M = 0

                xk_xn = r + coil_1[ri] * np.cos(df2 * k) * np.cos(fi) - coil_2[rj] * np.cos(df1 * n)
                yk_yn = coil_1[ri] * np.sin(df2 * k) * np.cos(fi) - coil_2[rj] * np.sin(df1 * n)
                zk_zn = d + coil_1[ri] * np.cos(df2 * k) * np.sin(fi)

                r12 = (xk_xn ** 2 + yk_yn ** 2 + zk_zn ** 2) ** 0.5

                M += (np.cos(df2 * k - df1 * n) * df1 * df2) / r12
                M *= mu0 * coil_1[ri] * coil_2[rj] / (4 * np.pi)
                mutual_inductance[ri][rj] = np.sum(M)
        mi_ro = np.append(mi_ro, np.sum(mutual_inductance))
    return mi_ro

def quality_factor(r, l, c):
    q = 1 / r * np.sqrt(l / c)
    return q

def self_inductance_coil(coil, r_turn):
    l = np.sum(self_inductance_turn(r=coil, s=r_turn))

    N = K = 30
    d, ro, fi = 0, 0, 0
    # mu0 = 4 * np.pi * 10 ** (-7)
    mu0 = 1.26e-6
    n = np.arange(N)
    k = n.reshape((K, 1))
    df1 = 2 * np.pi / N
    df2 = 2 * np.pi / K
    mutual_inductance = 0
    for ri in range(len(coil)):
        for rj in range(len(coil)):
            if ri != rj:
                M = 0

                xk_xn = ro + coil[ri] * np.cos(df2 * k) * np.cos(fi) - coil[rj] * np.cos(df1 * n)
                yk_yn = coil[ri] * np.sin(df2 * k) * np.cos(fi) - coil[rj] * np.sin(df1 * n)
                zk_zn = d + coil[ri] * np.cos(df2 * k) * np.sin(fi)

                r12 = (xk_xn ** 2 + yk_yn ** 2 + zk_zn ** 2) ** 0.5

                M += (np.cos(df2 * k - df1 * n) * df1 * df2) / r12
                M *= mu0 * coil[ri] * coil[rj] / (4 * np.pi)
                mutual_inductance += np.sum(M)
    l += mutual_inductance
    return l


def coupling_coefficient(coil_1, coil_2, r_turn, d, ro):
    l_1 = self_inductance_coil(coil_1, r_turn)
    l_2 = self_inductance_coil(coil_2, r_turn)
    m = mutual_inductance(coil_1, coil_2, d, ro)
    return m / np.sqrt(l_1 * l_2)