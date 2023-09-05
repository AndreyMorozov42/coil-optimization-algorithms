# функция расчёта взаимной индуктивности
import numpy as np


def mutual_inductance(coil_1, coil_2, d):
    N = K = 90
    fi = po = 0
    mu0 = 4 * np.pi * 10 ** (-7)
    mutual_inductance = np.ones((len(coil_1), len(coil_2)))
    n = np.arange(N)
    k = n.reshape((N, 1))
    df1 = 2 * np.pi / N
    df2 = 2 * np.pi / K
    for ri in range(len(coil_1)):
        for rj in range(len(coil_2)):
            M = 0
            xk_xn = po + coil_1[ri] * np.cos(df2 * k) * np.cos(fi) - coil_2[rj] * np.cos(df1 * n)
            yk_yn = coil_1[ri] * np.sin(df2 * k) * np.cos(fi) - coil_2[rj] * np.sin(df1 * n)
            zk_zn = d + coil_1[ri] * np.cos(df2 * k) * np.sin(fi)
            r12 = (xk_xn ** 2 + yk_yn ** 2 + zk_zn ** 2) ** 0.5
            M += (np.cos(df2 * k - df1 * n) * df1 * df2) / r12
            M *= mu0 * coil_1[ri] * coil_2[rj] / (4 * np.pi)
            mutual_inductance[ri][rj] = np.sum(M)
    return np.sum(mutual_inductance)


# функция расчёта собственной индуктивности
def self_inductance(coil, thin):
    N = K = 90
    df1 = df2 = 2 * np.pi / N
    d, po, fi = 0, 0, 0
    mu0 = 4 * np.pi * 10 ** (-7)
    L = np.sum(mu0 * coil * (np.log(8 * coil / thin) - 7 / 4 + (thin ** 2) / (8 * coil ** 2) * (np.log(8 * coil / thin) + 1 / 3)))
    mi = np.ones((len(coil), len(coil)))
    n = np.arange(N)
    k = n.reshape((N, 1))
    for ri in range(len(coil)):
        for rj in range(len(coil)):
            M = 0
            if ri != rj:
                M = 0
                xk_xn = po + coil[ri] * np.cos(df2 * k) * np.cos(fi) - coil[rj] * np.cos(df1 * n)
                yk_yn = coil[ri] * np.sin(df2 * k) * np.cos(fi) - coil[rj] * np.sin(df1 * n)
                zk_zn = d + coil[ri] * np.cos(df2 * k) * np.sin(fi)
                r12 = (xk_xn ** 2 + yk_yn ** 2 + zk_zn ** 2) ** 0.5
                M += (np.cos(df2 * k - df1 * n) * df1 * df2) / r12
                M *= mu0 * coil[ri] * coil[rj] / (4 * np.pi)
            mi[ri][rj] = np.sum(M)
    M = np.sum(mi)
    L += M
    return L


# функция расчёта коэффициента связи
def coupling_coefficient(coil_1, thin1, coil_2, thin2, dist):
    mi = mutual_inductance(coil_1, coil_2, dist)
    l1 = self_inductance(coil_1, thin1)
    l2 = self_inductance(coil_2, thin2)
    k = mi / (l1 * l2) ** 0.5
    return k