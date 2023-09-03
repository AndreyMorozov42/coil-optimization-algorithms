import numpy as np
import matplotlib.pyplot as plt

from tools.mutual_inductance import mutual_inductance
from tools.coupling_coefficient import coupling_coefficient
from tools.mutation import mutation_lb


def show_plot(x, y, x_label="x", y_label="y", title=None):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    plt.grid()
    plt.show()


def hill_climbing(start, finish, coil_2, r_turn, ro, d):
    all_mutation = []
    good_mutation = []
    bad_mutation = []
    i = 0

    thr = 1e-3

    coil_1 = np.linspace(0.028, 0.07, 4)
    coil_1[2] = mutation_lb(start, finish)
    fit_k = coupling_coefficient(coil_1=coil_1, coil_2=coil_2, r_turn=r_turn, ro=ro, d=d)

    coil_1q = coil_1.copy()
    coil_1q[2] = mutation_lb(start, finish, x=coil_1[2])
    fit_kq = coupling_coefficient(coil_1=coil_1q, coil_2=coil_2, r_turn=r_turn, ro=ro, d=d)

    all_mutation.append((coil_1q[2].copy(), fit_kq.copy()))

    print(f"Initial Coupling coefficient {fit_k} for coil:\n"
          f"coil1 = {coil_1}\n"
          f"coil2 = {coil_2}")

    while np.abs(fit_kq - fit_k) >= thr:
        i += 1
        print(f"Algorithm iteration: {i}")
        if fit_kq > fit_k:
            print(f"Found a new maximum value of the coupling coefficient: {fit_kq}")
            coil_1 = coil_1q.copy()
            fit_k = fit_kq.copy()
            good_mutation.append((coil_1q[2].copy(), fit_kq.copy()))
        else:
            bad_mutation.append((coil_1q[2].copy(), fit_kq.copy()))

        coil_1q[2] = mutation_lb(start, finish, x=coil_1[2])
        fit_kq = coupling_coefficient(coil_1=coil_1q, coil_2=coil_2, r_turn=r_turn, ro=ro, d=d)

        all_mutation.append((coil_1q[2].copy(), fit_kq.copy()))

        if i > 1000:
            return [], [], []

    if fit_kq > fit_k:
        print(f"Found a new maximum value of the coupling coefficient: {fit_kq}")
        good_mutation.append((coil_1q[1].copy(), fit_kq.copy()))
    else:
        bad_mutation.append((coil_1q[1].copy(), fit_kq.copy()))

    return all_mutation, good_mutation, bad_mutation


def show_climbing(x, y, x_label="x", y_label="y", title=None, good_points=None, bad_points=None):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    if good_points is not None:
        for p in good_points:
            plt.scatter(p[0], p[1], c="green")
    if bad_points is not None:
        for p in bad_points:
            plt.scatter(p[0], p[1], c="red")
    plt.grid()
    plt.show()

def main():
    # transmitting coil
    coil_t = np.linspace(0.028, 0.07, 4)

    r_turn = 0.0004

    # receiving coil
    coils_r = np.linspace(0.028, 0.07, 4) + np.zeros((50, 4))
    coils_r.T[2] = np.linspace(coils_r[0][1] + 2 * r_turn, coils_r[0][3] - 2 * r_turn, 50)

    # distance
    d = 0.005
    ro = [0]

    # calculation mutual inductance and couple
    m = np.zeros(coils_r.shape[0])
    k = np.zeros(coils_r.shape[0])
    for ind_c in range(coils_r.shape[0]):
        coil_r = coils_r[ind_c]
        m[ind_c] = mutual_inductance(coil_1=coil_t, coil_2=coil_r, d=d, ro=ro)
        k[ind_c] = coupling_coefficient(coil_1=coil_t, coil_2=coil_r, r_turn=r_turn, d=d)

    # show distribution of mutual inductance and couple coefficient
    show_plot(x=coils_r.T[2], y=m * 1e6, x_label="r, м", y_label="M, мкГн", title="Mutual Inductance")
    show_plot(x=coils_r.T[2], y=k, x_label="r, м", y_label="k", title="Couple Coefficient")

    # show the maximum value of mutual inductance
    # and the corresponding radius value
    m_max = np.max(m)
    r2t_m_max = coils_r[np.argmax(m)][2]
    print(f"M_max = {m_max * 1e6} мкГн, for R 3T = {r2t_m_max} м")

    # show the maximum value of couple coefficient
    # and the corresponding radius value
    k_max = np.max(k)
    r2t_k_max = coils_r[np.argmax(k)][2]
    print(f"k_max = {k_max}, for R 3T = {r2t_k_max} м")

    # hill climbing algorithm testing on one iteration
    allm, good, bad = hill_climbing(
        start=coils_r[0][2] + 2 * r_turn, finish=coils_r[-1][2] - 2 * r_turn,
        coil_2=coil_t, r_turn=r_turn,
        ro=ro, d=d
    )

    show_climbing(x=coils_r.T[2], y=k, x_label="r, м", y_label="k", title="Hill Climbing with all mutation",
                  good_points=good, bad_points=bad)
    show_climbing(x=coils_r.T[2], y=k, x_label="r, м", y_label="k", title="Hill Climbing with good mutation",
                  good_points=good)
    show_climbing(x=coils_r.T[2], y=k, x_label="r, м", y_label="k", title="Hill Climbing with bad mutation",
                  bad_points=bad)

    print(f"Total mutations: {len(allm)}")
    print(f"Good mutations: {len(good)}")
    print(f"Bad mutations: {len(bad)}")

    if len(good) != 0:
        print(f"The resulting value of the coupling coefficient: {good[-1][1][0]}\n"
              f"for coil_1 = {good[-1]} and coil_2 = {coil_t}")
    else:
        print(f"The resulting value of the coupling coefficient: {allm[-1][1][0]}\n"
              f"for coil_1 = {allm[-1]} and coil_2 = {coil_t}")


if __name__ == "__main__":
    main()
