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

    # arrays for storing mutation values
    all_mutation = []
    good_mutation = []
    bad_mutation = []

    i = 0                   # iteration counter

    # objective function increment threshold
    thr = 1e-3

    # generate a coil and mutate it
    coil_1 = np.array([mutation_lb(start, finish)])
    fit_k = coupling_coefficient(coil_1=coil_1, coil_2=coil_2, r_turn=r_turn, ro=ro, d=d)

    print(f"Initial Coupling coefficient {fit_k} for coils:\n"
          f"coil1 = {coil_1}\n"
          f"coil2 = {coil_2}\n")

    # save the mutation
    all_mutation.append((coil_1.copy(), fit_k.copy()))
    i += 1

    coil_1q = np.array([mutation_lb(start, finish, x=coil_1[0])])
    fit_kq = coupling_coefficient(coil_1=coil_1q, coil_2=coil_2, r_turn=r_turn, ro=ro, d=d)

    # save the mutation
    all_mutation.append((coil_1q.copy(), fit_kq.copy()))
    i += 1

    while np.abs(fit_kq - fit_k) >= thr and i != 1000:
        i += 1
        # print(f"Algorithm iteration: {i}")

        if fit_kq > fit_k:
            print(f"{i}: Found a new maximum value of the coupling coefficient: {fit_kq}")
            coil_1 = coil_1q.copy()
            fit_k = fit_kq.copy()
            # save the good mutation
            good_mutation.append((coil_1q.copy(), fit_kq.copy()))
        else:
            # save the bad mutation
            bad_mutation.append((coil_1q.copy(), fit_kq.copy()))

        # mutate the coil
        coil_1q = np.array([mutation_lb(start, finish, x=coil_1[0])])
        fit_kq = coupling_coefficient(coil_1=coil_1q, coil_2=coil_2, r_turn=r_turn, ro=ro, d=d)

        # save the mutation
        all_mutation.append((coil_1q.copy(), fit_kq.copy()))

    print(f"Stop at {i} iterations\n")

    if fit_kq > fit_k:
        print(f"Found a new maximum value of the coupling coefficient: {fit_kq}")
        good_mutation.append((coil_1q.copy(), fit_kq.copy()))
    else:
        bad_mutation.append((coil_1q.copy(), fit_kq.copy()))

    return all_mutation, good_mutation, bad_mutation


def launch(iterations, start, finish, coil_2, r_turn, ro, d, k_max):
    # array of mutation counters
    arr_good = np.array([])
    arr_bad = np.array([])
    arr_all = np.array([])

    # counter when the algorithm has not found the maximum
    failure = 0

    # algorithm convergence criterion
    thr = 1e-1

    for _ in range(iterations):
        allm, good, bad = hill_climbing(start=start, finish=finish,
                                        coil_2=coil_2, r_turn=r_turn, ro=ro, d=d)

        if len(good) != 0:
            # checking that the algorithm has converged
            if np.abs(good[-1][1] - k_max) < thr:
                arr_all = np.append(arr_all, len(allm))
                arr_good = np.append(arr_good, len(good))
                arr_bad = np.append(arr_bad, len(bad))
            else:
                failure += 1
        else:
            # if the first generated coil gave the maximum coupling coefficient
            if np.abs(allm[0][1] - k_max) < thr:
                arr_all = np.append(arr_all, 1)
                arr_good = np.append(arr_good, 1)
                arr_bad = np.append(arr_bad, 0)
            else:
                failure += 1

    mean_agb = (np.average(arr_all), np.average(arr_good), np.average(arr_bad))
    median_agb = (np.median(arr_all), np.median(arr_good), np.mean(arr_bad))
    deviation_agb = (np.std(arr_all), np.std(arr_good), np.std(arr_bad))

    # calculate count of successful run
    all_iterations = iterations - failure

    return mean_agb, median_agb, deviation_agb, all_iterations



def main():
    coil_r = np.array([0.028])                              # receiving coil
    coils_t = np.round(np.linspace(0.02, 0.1, num=50), 3)   # transmitting coils

    r_turn = 0.0004     # radius of coil turns

    # distance
    d = 0.01
    ro = [0]

    # calculation mutual inductance and couple
    m = np.zeros(coils_t.shape[0])
    k = np.zeros(coils_t.shape[0])
    for ind_r in range(coils_t.shape[0]):
        coil_t = np.array([coils_t[ind_r]])
        m[ind_r] = mutual_inductance(coil_1=coil_t, coil_2=coil_r, d=d, ro=ro)
        k[ind_r] = coupling_coefficient(coil_1=coil_t, coil_2=coil_r, r_turn=r_turn, d=d)

    FLAG_SHOW_PLOT = True
    FLAG_RUN_MULTIITER = True

    # show the maximum value of mutual inductance
    # and the corresponding radius value
    m_max = np.max(m)
    r_m_max = coils_t[np.argmax(m)]
    print(f"M_max = {m_max * 1e6} мкГн, for Rt = {r_m_max} м")

    # show the maximum value of couple coefficient
    # and the corresponding radius value
    k_max = np.max(k)
    r_k_max = coils_t[np.argmax(k)]
    print(f"k_max = {k_max}, for r = {r_k_max} м\n")

    if FLAG_SHOW_PLOT:
        '''
        ------------------------------------------------------------
        Show distribution plots of
        coupling coefficient and mutual inductance.
        ------------------------------------------------------------
        '''
        # show distribution of mutual inductance and couple coefficient
        show_plot(x=coils_t, y=m * 1e6, x_label="R2, м", y_label="M, мкГн", title="Взаимная индуктивность двух витков")
        show_plot(x=coils_t, y=k, x_label="R2, м", y_label="k", title="Коэффициент связи двух витков")


    if not FLAG_RUN_MULTIITER:
        '''
        ------------------------------------------------------------
        Testing the algorithm for Hill Climbing in one run.
        ------------------------------------------------------------
        '''
        allm, good, bad = hill_climbing(start=coils_t[0], finish=coils_t[-1],
                                        coil_2=coil_r, r_turn=r_turn, ro=ro, d=d)

        show_climbing(x=coils_t, y=k, x_label="R2, м", y_label="k", title="Поиск максимума коэффициента связи алгоритмом\n "
                                                                          "\"Поиск восхождением к вершине\"",
                      good_points=good, bad_points=bad)
        show_climbing(x=coils_t, y=k, x_label="R2, м", y_label="k", title="Поиск максимума коэффициента связи алгоритмом\n "
                                                                          "\"Поиск восхождением к вершине\"",
                      good_points=good)
        show_climbing(x=coils_t, y=k, x_label="R2, м", y_label="k", title="Поиск максимума коэффициента связи алгоритмом\n "
                                                                          "\"Поиск восхождением к вершине\"",
                      bad_points=bad)

        if len(good) != 0:
            print(f"Total mutations: {len(allm)}")
            print(f"Good mutations: {len(good)}")
            print(f"Bad mutations: {len(bad)}\n")
            print(f"The resulting value of the coupling coefficient: {good[-1][1]}\n"
                  f"for coil_t = {good[-1][0]} м and coil_r = {coil_r[0]} м")
        elif len(good) == 0:
            print(f"Total mutations: {len(allm)}")
            print(f"Good mutations: {len(good)}")
            print(f"Bad mutations: {len(bad) + 1}")
            print(f"The resulting value of the coupling coefficient: {allm[0][1]}\n"
                  f"for coil_t = {allm[-1][0]} м and coil_r = {coil_r} м")
    elif FLAG_RUN_MULTIITER:
        '''
        ------------------------------------------------------------
        Testing the algorithm for climbing
        to the top of a hill on several runs.
        ------------------------------------------------------------
        '''
        iterations = 1000
        mean_agb, median_agb, deviation_agb, counter = launch(iterations=iterations,
                                                              start=coils_t[0], finish=coils_t[-1],
                                                              coil_2=coil_r, r_turn=r_turn,
                                                              ro=ro, d=d, k_max=k_max)

        print(f"Average good mutation: {mean_agb[1]}")
        print(f"Average bad mutation: {mean_agb[2]}")
        print(f"Average all mutation: {mean_agb[0]}\n")

        print(f"Median good mutation: {median_agb[1]}")
        print(f"Median bad mutation: {median_agb[2]}")
        print(f"Median all mutation: {median_agb[0]}\n")

        print(f"Deviation good mutation: {deviation_agb[1]}")
        print(f"Deviation bad mutation: {deviation_agb[2]}")
        print(f"Deviation all mutation: {deviation_agb[0]}\n")

        print(f"Total iterations of running algorithms: {counter}")


if __name__ == "__main__":
    main()
