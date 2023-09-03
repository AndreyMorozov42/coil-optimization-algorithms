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


def show_climbing(x, y, x_label="x", y_label="y", title=None, good_points=None, bad_points=None):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    if good_points is not None:
        for p in good_points:
            plt.scatter(p[0][0], p[1][0], c="green")
    if bad_points is not None:
        for p in bad_points:
            plt.scatter(p[0][0], p[1][0], c="red")
    plt.grid()
    plt.show()


def hill_climbing(start, finish, coil_2, r_turn, ro, d):
    all_mutation = []
    good_mutation = []
    bad_mutation = []
    i = 0

    thr = 1e-3

    coil_1 = np.array([mutation_lb(start, finish)])
    fit_k = coupling_coefficient(coil_1=coil_1, coil_2=coil_2, r_turn=r_turn, ro=ro, d=d)

    coil_1q = np.array([mutation_lb(start, finish, x=coil_1[0])])
    fit_kq = coupling_coefficient(coil_1=coil_1q, coil_2=coil_2, r_turn=r_turn, ro=ro, d=d)

    all_mutation.append((coil_1q.copy(), fit_kq.copy()))

    # print(f"Initial Coupling coefficient {fit_k} for coil:\n"
    #       f"coil1 = {coil_1}\n"
    #       f"coil2 = {coil_2}")

    while np.abs(fit_kq - fit_k) >= thr:
        i += 1
        # print(f"Algorithm iteration: {i}")
        if fit_kq > fit_k:
            # print(f"Found a new maximum value of the coupling coefficient: {fit_kq}")
            coil_1 = coil_1q.copy()
            fit_k = fit_kq.copy()
            good_mutation.append((coil_1q.copy(), fit_kq.copy()))
        else:
            bad_mutation.append((coil_1q.copy(), fit_kq.copy()))

        coil_1q = np.array([mutation_lb(start, finish, x=coil_1[0])])
        fit_kq = coupling_coefficient(coil_1=coil_1q, coil_2=coil_2, r_turn=r_turn, ro=ro, d=d)

        all_mutation.append((coil_1q.copy(), fit_kq.copy()))

        if i > 1000:
            return [], [], []

    if fit_kq > fit_k:
        # print(f"Found a new maximum value of the coupling coefficient: {fit_kq}")
        good_mutation.append((coil_1q.copy(), fit_kq.copy()))
    else:
        bad_mutation.append((coil_1q.copy(), fit_kq.copy()))

    return all_mutation, good_mutation, bad_mutation


def launch(iterations, start, finish, coil_2, r_turn, ro, d, k_max):
    avr_good = np.array([])
    avr_bad = np.array([])
    avr_all = np.array([])

    # counter when the algorithm has not found the maximum
    failure = 0

    for _ in range(iterations):
        allm, good, bad = hill_climbing(start=start, finish=finish,
                                        coil_2=coil_2, r_turn=r_turn, ro=ro, d=d)

        # check received value and maximum
        if len(good) and np.abs(good[-1][1] - k_max) > 1e-1:
            failure += 1
        else:
            avr_all = np.append(avr_all, len(allm))
            avr_good = np.append(avr_good, len(good))
            avr_bad = np.append(avr_bad, len(bad))

    # ToDo: calculate variance and standard deviation

    avr_all = np.mean(avr_all)
    avr_good = np.mean(avr_good)
    avr_bad = np.mean(avr_bad)

    all_iterations = iterations - failure

    return avr_good, avr_bad, avr_all, all_iterations



def main():
    # two coils
    coil_t = np.array([0.028])
    coils_r = np.linspace(0.02, 0.1, num=50)
    r_turn = 0.0004

    # distance
    d = 0.005
    ro = [0]

    # calculation mutual inductance and couple
    m = np.zeros(coils_r.shape[0])
    k = np.zeros(coils_r.shape[0])
    for ind_r in range(coils_r.shape[0]):
        coil_r = np.array([coils_r[ind_r]])
        m[ind_r] = mutual_inductance(coil_1=coil_t, coil_2=coil_r, d=d, ro=ro)
        k[ind_r] = coupling_coefficient(coil_1=coil_t, coil_2=coil_r, r_turn=r_turn, d=d)

    # show distribution of mutual inductance and couple coefficient
    # show_plot(x=coils_r, y=m * 1e6, x_label="r, м", y_label="M, мкГн", title="Mutual Inductance")
    # show_plot(x=coils_r, y=k, x_label="r, м", y_label="k", title="Couple Coefficient")

    # show the maximum value of mutual inductance
    # and the corresponding radius value
    m_max = np.max(m)
    r_m_max = coils_r[np.argmax(m)]
    print(f"M_max = {m_max * 1e6} мкГн, for r = {r_m_max} м")

    # show the maximum value of couple coefficient
    # and the corresponding radius value
    k_max = np.max(k)
    r_k_max = coils_r[np.argmax(k)]
    print(f"k_max = {k_max}, for r = {r_k_max} м")


    ##### !!! hill climbing algorithm testing on one iteration !!!
    # allm, good, bad = hill_climbing(start=coils_r[0], finish=coils_r[-1],
    #                                 coil_2=coil_t, r_turn=r_turn, ro=ro, d=d)
    #
    # show_climbing(x=coils_r, y=k, x_label="r, м", y_label="k", title="Hill Climbing with all mutation",
    #               good_points=good, bad_points=bad)
    # show_climbing(x=coils_r, y=k, x_label="r, м", y_label="k", title="Hill Climbing with good mutation",
    #               good_points=good)
    # show_climbing(x=coils_r, y=k, x_label="r, м", y_label="k", title="Hill Climbing with bad mutation",
    #               bad_points=bad)
    #
    # print(f"Total mutations: {len(allm)}")
    # print(f"Good mutations: {len(good)}")
    # print(f"Bad mutations: {len(bad)}")
    #
    # if len(good) != 0:
    #     print(f"The resulting value of the coupling coefficient: {good[-1][1][0]}\n"
    #           f"for coil_1 = {good[-1][0][0]} and coil_2 = {coil_t[0]}")
    # else:
    #     print(f"The resulting value of the coupling coefficient: {allm[-1][1][0]}\n"
    #           f"for coil_1 = {allm[-1][0][0]} and coil_2 = {coil_t[0]}")


    ##### !!! hill climbing algorithm testing on multiple iteration !!!
    iterations = 1000
    average_good, average_bad, average_all, counter = launch(iterations=iterations,
                                                             start=coils_r[0], finish=coils_r[-1],
                                                             coil_2=coil_t, r_turn=r_turn,
                                                             ro=ro, d=d, k_max=k_max)
    print(f"Average good mutation: {average_good}")
    print(f"Average bad mutation: {average_bad}")
    print(f"Average all mutation: {average_all}")
    print(f"Total iterations of running algorithms: {counter}")




if __name__ == "__main__":
    main()
