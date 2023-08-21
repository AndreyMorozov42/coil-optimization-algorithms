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
    # Step 1. Assignment of the initial values
    power = 0.3
    n = 10
    f = 6.78e6
    w = 2 * np.pi * f
    c_t = c_r = 0.1e-9

    # parameter of coil
    r_out_r = 0.025
    r_turn = 0.0004

    # distance
    fi = 0
    d = 0.01
    ro = np.linspace(0, 35 * 1e-3, num=30)

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

    # n_t = int(np.ceil(l_t / l_turn))
    # k_r = int(np.ceil(l_r / l_turn))

    print(f"Count of turn:\n Nt={n_t}\n Kr={k_r}\n")

    # Step 4. Calculation of Vs
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

    print(f"Value of Vs:\n {vs} В\n")

    # Step 5. Calculation of m_min and m_max
    a = np.sqrt(r_l * vs ** 2 - 4 * p_min * z_t * z_r)
    m_max = np.abs((vs * np.sqrt(r_l) + a)) / (2 * w * np.sqrt(p_min))
    m_min = np.abs((vs * np.sqrt(r_l) - a)) / (2 * w * np.sqrt(p_min))

    print(f"m_max = {m_max * 10 ** 6} мкГн")
    print(f"m_min = {m_min * 10 ** 6} мкГн\n")

    debug(
        ro=ro, m_max=m_max,
        m_min=m_min,
        title="Шаг 5. Получение диапазона\n для взаимной индуктивности (M)"
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
        m=m, title="Шаг 6. Расчёт взаимной индуктивности\n при подобранных внутренних радиусах"
    )

    print(f"Calculated values of internal radii:\n r_in_t={r_in_t} м\n r_in_r={r_in_r} м\n")

    print(f"m_calc_max={np.max(m)}")
    print(f"m_calc_min={np.min(m)}")

    flag = True
    # while np.min(m) < m_min or np.max(m) > m_max or np.max(m) / m_max <= 0.85 or m_min / np.min(m) <= 0.85:
    while np.min(m) < m_min or np.max(m) > m_max:

        # if np.max(m) / m_max <= 0.85:
        #     n_t += 1
        #
        # if m_min / np.min(m) <= 0.85:
        #     k_r -= 1

        # Step 8. Calculation r out_t max
        r_max_t = 0

        r_i0 = r_out_r
        m_i0 = mutual_inductance(
            coil_1=np.linspace(r_in_r, r_i0, n_t),
            coil_2=np.linspace(r_in_t, r_out_r, k_r),
            d=d, ro=ro
        )

        r_i1 = r_out_r

        while np.min(m_i0) < m_min:
            r_i1 += r_out_r
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
        print("Finish step 8.")

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
            m=m_rmax, title="Шаг 8. Определение RmaxT"
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

        r_out_t = (a + b) / 2
        m = mutual_inductance(
            coil_1=np.linspace(r_in_t, r_out_t, n_t),
            coil_2=np.linspace(r_in_r, r_out_r, k_r),
            d=d, ro=ro
        )

        print("Finish step 9.")
        print(f"Calculated values of r_out_t={r_out_t} м\n")
        debug(
            ro=ro, m_max=m_max,
            m_min=m_min,
            m=m, title="Шаг 9. Расчёт RoutT"
        )

        # check if procedure optimization is possible
        if n_t > (r_out_t - 1e-3) / 1e-3:
            print("Process terminated. Geometric optimization coil is impossible. Break 1")
            flag = False
            break

        while True:
            # Step 10. Recalculation of r_in_t and r_int_r
            a = 1e-3
            b = r_out_r - (k_r - 1) * (r_turn * 2 + a)

            while np.abs(a - b) >= eps:
                x1 = (a + b - eps) / 2
                x2 = (a + b + eps) / 2

                m_x1 = np.max(mutual_inductance(
                    coil_1=np.linspace(x1, r_out_t, n_t),
                    coil_2=np.linspace(x1, r_out_r, k_r),
                    d=d, ro=ro
                ))

                m_x2 = np.max(mutual_inductance(
                    coil_1=np.linspace(x2, r_out_t, n_t),
                    coil_2=np.linspace(x2, r_out_r, k_r),
                    d=d, ro=ro
                ))

                if np.abs(m_max - m_x1) < np.abs(m_max - m_x2):
                    b = x1
                else:
                    a = x2

            r_in_t = r_in_r = (a + b) / 2

            m = mutual_inductance(
                coil_1=np.linspace(r_in_t, r_out_t, n_t),
                coil_2=np.linspace(r_in_r, r_out_r, k_r),
                d=d, ro=ro
            )

            print(f"Finnish step 10.")
            print(f"Calculated values of r_in_t={r_in_t} м\n")
            print(f"Calculated values of r_in_r={r_in_r} м\n")

            debug(
                ro=ro, m_max=m_max,
                m_min=m_min,
                m=m, title="Шаг 10. Расчёт внутренних радиусов катушек r inT, r inR"
            )

            if np.max(m) > m_max:
                if k_r > 1:
                    k_r -= 1
                    print(f"Changed count of turn k_r={k_r}")
                else:
                    flag = False
                    break
            else:
                break

            if flag is False:
                break


    m = mutual_inductance(
        coil_1=np.linspace(r_in_t, r_out_t, n_t),
        coil_2=np.linspace(r_in_r, r_out_r, k_r),
        d=d, ro=ro
    )

    debug(
        ro=ro, m_max=m_max,
        m_min=m_min, m=m,
        title="После процедуры оптимизации"
    )

    print("Step 11. Пересчёт L_t, L_r и С_t, C_r")

    # Step 11. Пересчёт L_t, L_r и С_t, C_r
    l_t = self_inductance_coil(coil=np.linspace(r_in_t, r_out_t, n_t),
                               r_turn=r_turn)
    c_t = 1 / (l_t * w ** 2)
    q_t = 1 / (r_t) * np.sqrt(l_t / c_t)

    l_r = self_inductance_coil(coil=np.linspace(r_in_r, r_out_r, k_r),
                               r_turn=r_turn)
    c_r = 1 / (l_r * w ** 2)
    q_r = 1 / (r_l) * np.sqrt(l_r / c_r)


    if flag:
        print(f"Transmitting part:\n r in_t={r_in_t * 1e3} мм\n r out_t={r_out_t * 1e3} мм\n Nt={n_t}")
        print(f"l_t={l_t * 1e6} мкГн")
        print(f"c_t={c_t * 1e9} нФ")
        # print(f"q_t={q_t}\n")

        print(f"Receiving part:\n r in_r={r_in_r * 1e3} мм\n r out_r={r_out_r * 1e3} мм\n Kr={k_r}")
        print(f"l_r={l_r * 1e6} мкГн")
        print(f"c_r={c_r * 1e9} нФ")
        # print(f"q_r={q_r}\n")
    else:
        print("Optimization is impossible!")

    z_t = 1j * w * l_t + 1 / (1j * w * c_t) + r_t
    z_r = 1j * w * l_r + 1 / (1j * w * c_r) + r_l + r_r
    p_l = (w ** 2) * (m ** 2) * (vs ** 2) * r_l / (np.abs(z_t * z_r) + (w ** 2) * (m ** 2)) ** 2

    debug(ro=m, m_max=p_max, m_min=p_min,
          m=p_l, title="Выходная мощность",
          x_label="M, Гн", y_label="P, Вт")

    print(f"Перепад выходной мощности:\n p={(np.max(p_l) - np.min(p_l)) / np.max(p_l)}")

if __name__ == "__main__":
    main()

