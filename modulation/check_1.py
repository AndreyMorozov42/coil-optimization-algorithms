import numpy as np
import matplotlib.pyplot as plt

from main_function import *

def debug(ro, m_max, m_min, m=None, title=None, x_label="ro, м", y_label="P, Вт"):
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
    # Step 4. Calculation of Vs
    r_t = 3
    r_r = 0
    r_l = 20

    p_l = 0.01
    p_max = p_l + 0.15 * p_l
    p_min = p_l - 0.15 * p_l

    f = 0.2e6
    w = 2 * np.pi * f

    c_t, c_r = 35.2e-9, 37.2e-9
    n, k = 15, 15

    # l_t = 1 / (c_t * w ** 2)
    # l_r = 1 / (c_r * w ** 2)

    r_turn = 0.0004

    r_outT = 0.0545
    r_inT = 0.0207

    r_outR = 0.035
    r_inR = 0.0207

    # Step 11. Пересчёт L_t, L_r и С_t, C_r
    l_t = self_inductance_coil(coil=np.linspace(r_inT, r_outT, n),
                               r_turn=r_turn)

    l_r = self_inductance_coil(coil=np.linspace(r_inR, r_outR, k),
                               r_turn=r_turn)

    d = 0.01
    ro = np.linspace(0, 35 * 1e-3, num=30)

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

    print(f"l_t={l_t} Гн")
    c_t = 1 / (l_t * w ** 2)
    print(f"c_t={c_t} Ф")
    q_t = 1 / (r_t) * np.sqrt(l_t / c_t)
    print(f"q_t={q_t}\n")

    print(f"l_r={l_r} Гн")
    c_r = 1 / (l_r * w ** 2)
    print(f"c_r={c_r} Ф")
    q_r = 1 / (r_l) * np.sqrt(l_r / c_r)
    print(f"q_r={q_r}\n")

    m = mutual_inductance(
        coil_1=np.linspace(r_inT, r_outT, n),
        coil_2=np.linspace(r_inR, r_outR, k),
        d=d, ro=ro
    )

    z_t = 1j * w * l_t + 1 / (1j * w * c_t) + r_t
    z_r = 1j * w * l_r + 1 / (1j * w * c_r) + r_l + r_r
    p_l = (w ** 2) * (m ** 2) * (vs ** 2) * r_l / (np.abs(z_t * z_r) + (w ** 2) * (m ** 2)) ** 2

    debug(
        ro=ro, m_max=p_max,
        m_min=p_min, m=p_l,
        title="После процедуры оптимизации"
    )


if __name__ == "__main__":
    main()