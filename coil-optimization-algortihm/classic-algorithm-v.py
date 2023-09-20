import numpy as np
import matplotlib.pyplot as plt

from main_function import *


def debug(ro, m_max, m_min, m=None, title=None, x_label="ro, м", y_label="M, Гн"):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
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
    print("Step 1. Assignment of the initial values.\n")
    power = 0.01
    n = 15
    f = 0.2e6
    w = 2 * np.pi * f
    c_t = c_r = 15e-9
    # parameter of coil
    r_out_r = 0.035
    r_turn = 0.0004
    # distance
    fi = 0
    d = 0.01
    ro = np.linspace(0, 35 * 1e-3, num=35)
    p_max = power * (1 + n / 100)
    p_min = power * (1 - n / 100)
    print(f"Range of power:\n {p_min} ... {p_max} Вт\n")

    print("Step 2. Calculation of L_r and L_t")
    l_t = l_r = 1 / (c_t * w ** 2)
    print(f"Value of self-inductances:\n Lt={l_t * 10**6} мкГн\n Lr={l_r * 10**6} мкГн\n")

    print("Step 3. Calculation of N and K")
    r_out_t = r_in_t = r_in_r = r_out_r
    l_turn = self_inductance_turn(r=r_out_t, s=r_turn)
    n_t = int(np.ceil(np.sqrt(l_t / l_turn)))
    k_r = int(np.ceil(np.sqrt(l_r / l_turn)))
    print(f"Count of turn:\n Nt={n_t}\n Kr={k_r}\n")

    print("Step 4. Calculation of Vs")
    r_t = 3
    r_r = 0
    r_l = 20

    # calculation quality factor
    q_t = quality_factor(r_t, l_t, c_t)
    q_r = quality_factor(r_r + r_l, l_r, c_r)
    k_crit = 1 / np.sqrt(q_t * q_r)

    z_t = 1j * w * l_t + 1 / (1j * w * c_t) + r_t
    z_r = 1j * w * l_r + 1 / (1j * w * c_r) + r_l + r_r

    a = z_r * z_t / (w * k_crit * np.sqrt(l_t * l_r))
    b = w * k_crit * np.sqrt(l_t * l_r)
    vs = np.abs((a + b) * np.sqrt(p_max / r_l))
    print(f"Value of Vs: {vs} В\n")


    print("Step 5. Calculation of m_min and m_max.\n")
    a = np.sqrt(r_l * vs ** 2 - 4 * p_min * z_t * z_r)
    m_max = np.abs((vs * np.sqrt(r_l) + a)) / (2 * w * np.sqrt(p_min))
    m_min = np.abs((vs * np.sqrt(r_l) - a)) / (2 * w * np.sqrt(p_min))
    print(f"m_max = {m_max * 10 ** 6} мкГн")
    print(f"m_min = {m_min * 10 ** 6} мкГн\n")


    print("Step 6. Calculation of r in_t and r in_t.\n")
    a = 1e-3
    b = r_out_r - (k_r - 1) * a
    pogr = 5e-4
    eps = (a + b) / 2
    x1 = a
    m_x1 = mutual_inductance(
        coil_1=np.linspace(x1, r_out_t, n_t),
        coil_2=np.linspace(x1, r_out_r, k_r),
        ro=ro, d=d
    )
    x2 = b
    m_x2 = mutual_inductance(
        coil_1=np.linspace(x2, r_out_t, n_t),
        coil_2=np.linspace(x2, r_out_r, k_r),
        ro=ro, d=d
    )
    x0 = (x1 + x2) / 2
    m_x0 = mutual_inductance(
        coil_1=np.linspace(x0, r_out_t, n_t),
        coil_2=np.linspace(x0, r_out_r, k_r),
        ro=ro, d=d
    )
    koff2 = 0
    iter1 = 0
    while eps >= pogr or np.max(m_x0) > m_max:
        iter1 += 1
        if np.max(m_x0) > m_max:
            x2 = x0

            x0 = (x1 + x2) / 2
            m_x0 = mutual_inductance(
                coil_1=np.linspace(x0, r_out_t, n_t),
                coil_2=np.linspace(x0, r_out_r, k_r),
                ro=ro, d=d
            )
        else:
            x1 = x0

            x0 = (x1 + x2) / 2
            m_x0 = mutual_inductance(
                coil_1=np.linspace(x0, r_out_t, n_t),
                coil_2=np.linspace(x0, r_out_r, k_r),
                ro=ro, d=d
            )

        eps = (x2 - x1) / 2
        if iter1 > 15:
            koff2 = 1
            break

    r_in_r = r_in_t = x0

    m = mutual_inductance(
        coil_1=np.linspace(r_in_t, r_out_t, n_t),
        coil_2=np.linspace(r_in_r, r_out_r, k_r),
        ro=ro, d=d
    )
    koff3 = 0
    while np.min(m) < m_min or np.max(m) > m_max and koff3 == 0:

        print("Step 8. Calculation of R outT max.\n")

        m = mutual_inductance(
            coil_1=np.linspace(r_in_t, r_out_t, n_t),
            coil_2=np.linspace(r_in_r, r_out_r, k_r),
            ro=ro, d=d
        )
        m_prev = m.copy()
        r_out_t = 0

        koff = 0
        iter2 = 0
        while np.min(m) < m_min:
            iter2 += 1
            r_out_t += r_out_r

            m = mutual_inductance(
                coil_1=np.linspace(r_in_t, r_out_t, n_t),
                coil_2=np.linspace(r_in_r, r_out_r, k_r),
                ro=ro, d=d
            )

            if np.min(m_prev) > np.min(m) and iter2 > 10:
                koff = 1
                break
            m_prev = m

        while koff == 1:
            n_t += 1

            m = mutual_inductance(
                coil_1=np.linspace(r_in_t, r_out_t, n_t),
                coil_2=np.linspace(r_in_r, r_out_r, k_r),
                ro=ro, d=d
            )

            m_prev = m.copy()
            r_out_t = 0

            koff = 0
            iter2 = 0
            while np.min(m) < m_min:
                iter2 += 1
                r_out_t += r_out_r

                m = mutual_inductance(
                    coil_1=np.linspace(r_in_t, r_out_t, n_t),
                    coil_2=np.linspace(r_in_r, r_out_r, k_r),
                    ro=ro, d=d
                )

                if np.min(m_prev) > np.min(m) and iter2 > 10:
                    koff = 1
                    break
                m_prev = m

        print("Step 9. Calculation of RoutT.\n")
        a = r_out_r
        b = r_out_t

        eps = (a + b) / 2

        x1 = a
        m_x1 = mutual_inductance(
            coil_1=np.linspace(r_in_t, x1, n_t),
            coil_2=np.linspace(r_in_r, r_out_r, k_r),
            ro=ro, d=d
        )

        x2 = b
        m_x2 = mutual_inductance(
            coil_1=np.linspace(r_in_t, x2, n_t),
            coil_2=np.linspace(r_in_r, r_out_r, k_r),
            ro=ro, d=d
        )

        x0 = (x1 + x2) / 2
        m_x0 = mutual_inductance(
            coil_1=np.linspace(r_in_t, x0, n_t),
            coil_2=np.linspace(r_in_r, r_out_r, k_r),
            ro=ro, d=d
        )

        while eps >= pogr or np.min(m_x0) < m_min:
            if np.min(m_x0) > m_min:
                x2 = x0

                x0 = (x1 + x2) / 2
                m_x0 = mutual_inductance(
                    coil_1=np.linspace(r_in_t, x0, n_t),
                    coil_2=np.linspace(r_in_r, r_out_r, k_r),
                    ro=ro, d=d
                )
            else:
                x1 = x0

                x0 = (x1 + x2) / 2
                m_x0 = mutual_inductance(
                    coil_1=np.linspace(r_in_t, x0, n_t),
                    coil_2=np.linspace(r_in_r, r_out_r, k_r),
                    ro=ro, d=d
                )

            eps = (x2 - x1) / 2

        r_out_t = x0

        print("Step 10. Recalculation R_in.\n")
        a = 1e-3
        b = r_out_r - (k_r - 1) * a
        eps = (a + b) / 2

        x1 = a
        m_x1 = mutual_inductance(
            coil_1=np.linspace(x1, r_out_t, n_t),
            coil_2=np.linspace(x1, r_out_r, k_r),
            ro=ro, d=d
        )

        x2 = b
        m_x2 = mutual_inductance(
            coil_1=np.linspace(x2, r_out_t, n_t),
            coil_2=np.linspace(x2, r_out_r, k_r),
            ro=ro, d=d
        )

        x0 = (x1 + x2) / 2
        m_x0 = mutual_inductance(
            coil_1=np.linspace(x0, r_out_t, n_t),
            coil_2=np.linspace(x0, r_out_r, k_r),
            ro=ro, d=d
        )

        while eps >= pogr or np.max(m_x0) > m_max:
            if np.max(m_x0) > m_max:
                x2 = x0

                x0 = (x1 + x2) / 2
                m_x0 = mutual_inductance(
                    coil_1=np.linspace(x0, r_out_t, n_t),
                    coil_2=np.linspace(x0, r_out_r, k_r),
                    ro=ro, d=d
                )
            else:
                x1 = x0

                x0 = (x1 + x2) / 2
                m_x0 = mutual_inductance(
                    coil_1=np.linspace(x0, r_out_t, n_t),
                    coil_2=np.linspace(x0, r_out_r, k_r),
                    ro=ro, d=d
                )

            eps = (x2 - x1) / 2

            if iter1 > 15:
                koff2 = 1
                break

        r_in_r = r_in_t = x0

        while koff2 == 1:
            if k_r >= 2:
                k_r -= 1

                a = 1e-3
                b = r_out_r - (k_r - 1) * a

                eps = (a + b) / 2

                x1 = a
                m_x1 = mutual_inductance(
                    coil_1=np.linspace(x1, r_out_t, n_t),
                    coil_2=np.linspace(x1, r_out_r, k_r),
                    ro=ro, d=d
                )

                x2 = b
                m_x2 = mutual_inductance(
                    coil_1=np.linspace(x2, r_out_t, n_t),
                    coil_2=np.linspace(x2, r_out_r, k_r),
                    ro=ro, d=d
                )

                x0 = (x1 + x2) / 2
                m_x0 = mutual_inductance(
                    coil_1=np.linspace(x0, r_out_t, n_t),
                    coil_2=np.linspace(x0, r_out_r, k_r),
                    ro=ro, d=d
                )
                iter1 = 0
                while eps >= pogr or np.max(m_x0) > m_max:
                    iter1 += 1
                    if np.max(m_x0) > m_max:
                        x2 = x0

                        x0 = (x1 + x2) / 2
                        m_x0 = mutual_inductance(
                            coil_1=np.linspace(x0, r_out_t, n_t),
                            coil_2=np.linspace(x0, r_out_r, k_r),
                            ro=ro, d=d
                        )
                    else:
                        x1 = x0

                        x0 = (x1 + x2) / 2
                        m_x0 = mutual_inductance(
                            coil_1=np.linspace(x0, r_out_t, n_t),
                            coil_2=np.linspace(x0, r_out_r, k_r),
                            ro=ro, d=d
                        )

                    eps = (x2 - x1) / 2

                    if iter1 > 15:
                        koff2 = 1
                        break

                r_in_r = r_in_t = x0

            else:
                print("Optimization is not Possible.")
                koff2 = 0
                koff3 = 1

        m = mutual_inductance(
            coil_1=np.linspace(r_in_t, r_out_t, n_t),
            coil_2=np.linspace(r_in_r, r_out_r, k_r),
            ro=ro, d=d
        )

    l_t = self_inductance_coil(np.linspace(r_in_t, r_out_t, n_t), r_turn)
    c_t = 1 / (w ** 2 * l_t)
    print(f"Transmitting part:\n r in_t={r_in_t * 1e3} мм"
          f"                  \n r out_t={r_out_t * 1e3} мм"
          f"                  \n Nt={n_t}",
          f"                  \n Lt={l_t * 1e6} мкГн",
          f"                  \n Ct={c_t * 1e9} нФ\n"
          )

    l_r = self_inductance_coil(np.linspace(r_in_r, r_out_r, k_r), r_turn)
    c_r = 1 / (w ** 2 * l_r)
    print(f"Receiving part:\n r in_r={r_in_r * 1e3} мм"
          f"               \n r out_r={r_out_r * 1e3} мм"
          f"               \n Kr={k_r}"
          f"               \n Lr={l_r * 1e6} мкГн",
          f"               \n Cr={c_r * 1e9} нФ\n")

    # k = coupling_coefficient(
    #     coil_1=np.linspace(r_in_r, r_out_r, k_r),
    #     coil_2=np.linspace(r_in_t, r_out_t, n_t),
    #     r_turn=r_turn, d=d, ro=ro)
    # print(f"Coupling coefficient: {k}")


if __name__ == "__main__":
    main()

