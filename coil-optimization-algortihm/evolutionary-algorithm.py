import numpy as np
import matplotlib.pyplot as plt

from main_function import *


def debug(ro, m_max, m_min, m=None, title=None):
    plt.xlabel("ro, м")
    plt.ylabel("M, Гн")
    plt.plot(ro, m_max * np.ones(ro.shape), "k--", )
    plt.plot(ro, m_min * np.ones(ro.shape), "k--", )
    if m is not None:
        plt.plot(ro, m, label="Оптимизированный случай")
    plt.grid()
    if title is not None:
        plt.title(title)
    plt.legend(loc="best")
    plt.show()


def main():
    # Step 1. Assignment of the initial values
    power = 10
    n = 10
    f = 2e6
    w = 2 * np.pi * f
    c_t = c_r = 1e-9

    # parameter of coil
    r_out_r = 0.03
    r_turn = 0.0004

    # distance
    fi = 0
    d = 0.015
    ro = np.linspace(0, 30 * 1e-3, num=30)

    p_max = power * (1 + n / 100)
    p_min = power * (1 - n / 100)

    print(f"Range of power:\n {p_min} ... {p_max} Вт\n")

    # Step 2. Calculation of L_r and L_t
    l_t = l_r = 1 / (c_t * w ** 2)

    print(f"Value of self-inductances:\n Lt={l_t * 10**6} мкГн\n Lr={l_r * 10**6} мкГн\n")

    # Step 3. Calculation of N and K
    r_out_t = r_in_t = r_in_r = r_out_r
    l_turn = self_inductance_turn(r=r_out_t, s=r_turn)
    n_t = int(np.ceil(np.sqrt(l_t / l_turn)))
    k_r = int(np.ceil(np.sqrt(l_r / l_turn)))


    print(f"Count of turn:\n Nt={n_t}\n Kr={k_r}\n")

    # Step 4. Calculation of Vs
    r_t = 3
    r_r = 0
    r_l = 20

    # calculation quality factor
    q_t = np.round(quality_factor(r_t, l_t, c_t), 4)
    q_r = np.round(quality_factor(r_r + r_l, l_r, c_r), 4)
    k_crit = np.round(1 / np.sqrt(q_t * q_r), 4)

    z_t = np.abs(1j * w * l_t + 1 / (1j * w * c_t) + r_t)
    z_r = np.abs(1j * w * l_r + 1 / (1j * w * c_r) + r_l + r_r)

    a = z_r * z_t / (w * k_crit * np.sqrt(l_t * l_r))
    b = w * k_crit * np.sqrt(l_t * l_r)
    vs = (a + b) * np.sqrt(p_max / r_l)

    print(f"Value of Vs:\n {vs} В\n")

    # Step 5. Calculation of m_min and m_max
    a = np.sqrt(r_l * vs ** 2 - 4 * p_min * z_t * z_r)
    m_max = (vs * np.sqrt(r_l) + a) / (2 * w * np.sqrt(p_min))
    m_min = (vs * np.sqrt(r_l) - a) / (2 * w * np.sqrt(p_min))

    print(f"m_max = {m_max * 10 ** 6} мкГн")
    print(f"m_min = {m_min * 10 ** 6} мкГн\n")

    debug(
        ro=ro, m_max=m_max,
        m_min=m_min,
        title="Шаг 1. Получение диапазона\n для взаимной индуктивности (M)"
    )

    # Step 6. Calculation of r in_t and r in_t
    a = 1e-3
    b = r_out_r - (k_r - 1) * (r_turn * 2 + a)
    eps = 0.5e-3

    while(b - a) >= eps:
        x1 = (a + b - eps) / 2
        m_x1 = np.max(
            mutual_inductance(
                coil_1=np.linspace(x1, r_out_t, n_t),
                coil_2=np.linspace(x1, r_out_r, k_r),
                ro=ro, d=d
            )
        )

        x2 = (a + b + eps) / 2
        m_x2 = np.max(
            mutual_inductance(
                coil_1=np.linspace(x2, r_out_t, n_t),
                coil_2=np.linspace(x2, r_out_r, k_r),
                ro=ro, d=d
            )
        )

        if np.abs(m_max - m_x1) <= np.abs(m_max - m_x2):
            b = x1
        else:
            a = x2

    r_in_r = r_in_t = (a + b) / 2

    m = mutual_inductance(
        coil_1=np.linspace(r_in_t, r_out_t, n_t),
        coil_2=np.linspace(r_in_r, r_out_r, k_r),
        d=d, ro=ro
    )

    debug(
        ro=ro, m_max=m_max, m_min=m_min,
        m=m, title="Шаг 2. Расчёт взаимной индуктивности\n при подобранных внутренних радиусах"
    )

    print(f"Calculated values of internal radii:\n r_in_t={r_in_t} м\n r_in_r={r_in_r} м\n")

    r_max_t = 0
    r_i0 = r_out_r
    m_i0 = mutual_inductance(
        coil_1=np.linspace(r_in_r, r_i0, n_t),
        coil_2=np.linspace(r_in_t, r_out_r, k_r),
        d=d, ro=ro
    )

    r_i1 = r_out_r

    while np.min(m_i0) < m_min:
        r_i1 += r_out_t
        m_i1 = mutual_inductance(
            coil_1=np.linspace(r_in_r, r_i1, n_t),
            coil_2=np.linspace(r_in_t, r_out_r, k_r),
            d=d, ro=ro
        )

        if np.min(m_i1) > np.min(m_i0):
            m_i0 = m_i1
            r_i0 = r_i1
        else:
            if np.min(m_i0) < m_min:
                n_t += 1
                r_i0 = r_out_r
                m_i0 = mutual_inductance(
                    coil_1=np.linspace(r_in_r, r_i0, n_t),
                    coil_2=np.linspace(r_in_t, r_out_r, k_r),
                    d=d, ro=ro
                )
                r_i1 = r_out_r
                print(f"Changed count of turn n_t={n_t}")
    r_max_t = r_i0


    print(f"Calculated values of r_out_t max = {r_max_t} м\n")
    print(f"Count of turn n_t={n_t}")

    m_rmax = mutual_inductance(
        coil_1=np.linspace(r_in_r, r_max_t, n_t),
        coil_2=np.linspace(r_in_t, r_out_r, k_r),
        d=d, ro=ro
    )

    debug(
        ro=ro, m_max=m_max,
        m_min=m_min,
        m=m_rmax, title="Шаг 3. Определение RmaxT"
    )

    r_out_t = np.random.uniform(low=r_in_t, high=r_max_t)
    print(r_out_t)




if __name__ == "__main__":
    main()
