import numpy as np
import matplotlib.pyplot as plt

from main_function import *


def debug(ro, d, m_max, m_min, m):
    plt.xlabel("ro, м")
    plt.ylabel("M, Гн")
    plt.plot(ro, m_max * np.ones(ro.shape), "k--", )
    plt.plot(ro, m_min * np.ones(ro.shape), "k--", )
    plt.plot(ro, m, label="Оптимизированный случай")
    plt.grid()
    plt.legend(loc="best")
    plt.show()


def main():
    # Step 1. Set initial value
    power = 10
    n = 10
    f = 1e6
    w = 2 * np.pi * f
    c_t = c_r = 1e-9

    # parameter of coil
    r_out_r = 0.03
    r_turn = 0.0004

    # distance
    fi = 0
    d = 0.01
    ro = np.linspace(0, 30 * 1e-3, num=30)

    p_max = power * (1 + n / 100)
    p_min = power * (1 - n / 100)

    # Step 2. Calculation self inductance of coil
    l_t = l_r = 1 / (c_t * w ** 2)

    # Step 3. Calculation of N and K
    r_out_t = r_in_t = r_in_r = r_out_r
    l_turn = self_inductance_turn(r=r_out_t, s=r_turn)
    n_t = int(np.ceil(np.sqrt(l_t / l_turn)))
    k_r = int(np.ceil(np.sqrt(l_r / l_turn)))

    # Step 4. Calculation of Vs
    r_t = 3
    r_r = 0
    r_l = 20

    # calculation quality factor
    q_t = quality_factor(r_t, l_t, c_t)
    q_r = quality_factor(r_r + r_l, l_r, c_r)
    k_crit = 1 / np.sqrt(q_t * q_r)

    z_t = np.abs(1j * w * l_t + 1 / (1j * w * c_t) + r_t)
    z_r = np.abs(1j * w * l_r + 1 / (1j * w * c_r) + r_l + r_r)
    a = z_r * z_t / (w * k_crit * np.sqrt(l_t * l_r))
    b = w * k_crit * np.sqrt(l_t * l_r)
    vs = (a + b) * np.sqrt(p_max / r_l)

    # Step 5. Calculation of m_min and m_max
    a = np.sqrt(r_l * vs ** 2 - 4 * p_min * z_t * z_r)
    m_max = (vs * np.sqrt(r_l) + a) / (2 * w * np.sqrt(p_min))
    m_min = (vs * np.sqrt(r_l) - a) / (2 * w * np.sqrt(p_min))

    # Step 6. Calculation of r_in_t and r_int_r
    # ToDo: check this step
    a = r_turn
    b = r_out_r - (k_r - 1) * a
    eps = 1e-5

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
        ro=ro, m_max=m_max,
        m_min=m_min, d=d,
        m=m
    )

    # step 7. Calculation m_max and m_min
    # while np.min(m) < m_min and np.max(m) > m_max:

    # step 8. Calculation R_out_t max
    r_max_t = 0

    print(f"Begining count of turn n_t {n_t}")

    while np.min(m) < m_min:
        r_max_t += r_out_t
        m = mutual_inductance(
                coil_1=np.linspace(r_in_r, r_max_t, n_t),
                coil_2=np.linspace(r_in_t, r_out_r, k_r),
                d=d, ro=ro
            )

        if np.min(m) < m_min:
            n_t += 1
            r_max_t = r_out_t

    print(f"Finnish count of turn n_t {n_t}\n")

    debug(
        ro=ro, m_max=m_max,
        m_min=m_min, d=d,
        m=m
    )

    # Step 9. Calculation of R_out_T
    a = r_out_r
    b = r_max_t

    while np.abs(a - b) >= eps:
        x1 = (a + b - eps) / 2
        m_x1 = np.max(mutual_inductance(
            coil_1=np.linspace(r_in_t, x1, n_t),
            coil_2=np.linspace(r_in_r, r_out_r, k_r),
            d=d, ro=ro
        ))

        x2 = (a + b + eps) / 2
        m_x2 = np.max(mutual_inductance(
            coil_1=np.linspace(r_in_t, x2, n_t),
            coil_2=np.linspace(r_in_r, r_out_r, k_r),
            d=d, ro=ro
        ))

        if np.abs(m_min - m_x1) < np.abs(m_min - m_x2):
            b = x1
        else:
            a = x2

        print(a, b)

    r_out_t = (a + b) / 2
    m = mutual_inductance(
        coil_1=np.linspace(r_in_t, r_out_t, n_t),
        coil_2=np.linspace(r_in_r, r_out_r, k_r),
        d=d, ro=ro
    )

    debug(
        ro=ro, m_max=m_max,
        m_min=m_min, d=d,
        m=m
    )

    if n_t > (r_out_t - r_turn) / r_turn:
        print("Process terminated. Geometric optimization coil is not possible.")

    # ToDo: make other ranges
    # Step 10. Recalculation r_in_t, r_int_r
    a_t = r_turn
    b_t = r_out_t - 2 * r_turn

    a_r = r_turn
    b_r = r_out_r - 2 * r_turn

    while np.abs(a_t - b_t) >= eps and np.abs(a_r - b_r) >= eps:
        x1_t = (a_t + b_t - eps) / 2
        m_x1t = mutual_inductance(
            coil_1=np.linspace(r_in_t, r_out_t, n_t),
            coil_2=np.linspace(r_in_r, r_out_r, k_r),
            d=d, ro=ro
        )

        x2_t = (a_t + b_t + eps) / 2
        m_x2t = mutual_inductance(
            coil_1=np.linspace(r_in_t, r_out_t, n_t),
            coil_2=np.linspace(r_in_r, r_out_r, k_r),
            d=d, ro=ro
        )


if __name__ == "__main__":
    main()
