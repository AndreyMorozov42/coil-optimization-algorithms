import time
import numpy as np

from hill_climbing_optimization.functions import coupling_coefficient
from tools.mutation import mutation_lb, mutation_random


def steepest_ascent_hill_climbing(coil_1, r1_turn, coil_2, r2_turn, d):
    # mutation counter
    all_mutation = 0
    good_mutation = 0
    bad_mutation = 0

    # objective function increment threshold
    thr = 1e-3

    fit_k = coupling_coefficient(coil_1=coil_1, r1_turn=r1_turn,
                                 coil_2=coil_2, r2_turn=r2_turn,
                                 dist=d)

    print(f"Initial Coupling coefficient {fit_k} for coils:\n"
          f"coil1 = {coil_1}\n"
          f"coil2 = {coil_2}\n")

    # initialization of initial values
    fit_kq = 0
    coil_1q = coil_1.copy()
    coil_2q = coil_2.copy()

    array = []

    i = 0  # iteration counter
    limit = 1000

    while np.abs((fit_kq - fit_k) / fit_k) > thr and i != limit:
        i += 1

        if array != [] and array[0][2] > fit_k:
            print(f"{i}: Found a new maximum value of the coupling coefficient: {array[0][2]}")
            coil_1 = array[0][0].copy()
            coil_2 = array[0][1].copy()
            fit_k = array[0][2].copy()
            good_mutation += 1
        elif array:
            bad_mutation += 1

        array = []

        # mutate the radii of the internal turns of two coils
        for ind in range(len(coil_1) + len(coil_2)):
            if ind // len(coil_1) == 0 and ind != 0 and (ind + 1) != len(coil_1):
                coil_1q = coil_1.copy()

                # old mutation
                coil_1q[ind] = mutation_lb(start=coil_1q[ind - 1] + 2 * r1_turn,
                                           finish=coil_1q[ind + 1] - 2 * r1_turn,
                                           x=coil_1q[ind].copy())

                # new mutation
                coil_1q[ind] = mutation_random(
                    start=coil_1q[ind - 1] + 2 * r1_turn,
                    finish=coil_1q[ind + 1] - 2 * r1_turn,
                )

                fit_kq = coupling_coefficient(coil_1=coil_1q, r1_turn=r1_turn,
                                              coil_2=coil_2, r2_turn=r2_turn,
                                              dist=d)

                array.append((coil_1q.copy(), coil_2.copy(), fit_kq))

            elif ind // len(coil_1) == 1 and ind != len(coil_1) and ind != (len(coil_2) + len(coil_1) - 1):
                coil_2q = coil_2.copy()

                # old mutation
                # coil_2q[ind - len(coil_1)] = mutation_lb(start=coil_2q[ind - len(coil_1) - 1] + 2 * r2_turn,
                #                                          finish=coil_2q[ind - len(coil_1) + 1] - 2 * r2_turn,
                #                                          x=coil_2q[ind - len(coil_1)].copy())

                # new mutation
                coil_2q[ind - len(coil_1)] = mutation_random(
                    start=coil_2q[ind - len(coil_1) - 1] + 2 * r2_turn,
                    finish=coil_2q[ind - len(coil_1) + 1] - 2 * r2_turn,
                )

                fit_kq = coupling_coefficient(coil_1=coil_1, r1_turn=r1_turn,
                                              coil_2=coil_2q, r2_turn=r2_turn,
                                              dist=d)

                array.append((coil_1.copy(), coil_2q.copy(), fit_kq))
        all_mutation += 1

        # sorting by maximum coupling coefficient value
        # the maximum value is index 0
        array.sort(key=lambda x: x[2], reverse=True)

    if array[0][2] > fit_k:
        print(f"{i}: Found a new maximum value of the coupling coefficient: {array[0][2]}")
        coil_1 = array[0][0].copy()
        coil_2 = array[0][1].copy()
        fit_k = array[0][2].copy()
        good_mutation += 1
    elif array:
        bad_mutation += 1

    print(f"Stop at {i} iterations\n")

    return coil_1.copy(), coil_2.copy(), fit_k, all_mutation, bad_mutation, good_mutation


def launch(iterations, coil_t, rt_turn, coil_r, rr_turn, d):
    # array of mutation counters
    arr_good = np.array([])
    arr_bad = np.array([])
    arr_all = np.array([])
    arr_time = np.array([])
    fit = []

    for _ in range(iterations):
        # search time calculation
        delta_t = time.time()
        coil_t1, coil_r1, k, allm, badm, goodm = steepest_ascent_hill_climbing(
            coil_1=coil_t, r1_turn=rt_turn,
            coil_2=coil_r, r2_turn=rr_turn,
            d=d
        )
        delta_t = time.time() - delta_t

        arr_all = np.append(arr_all, allm)
        arr_good = np.append(arr_good, goodm)
        arr_bad = np.append(arr_bad, badm)
        arr_time = np.append(arr_time, delta_t)

        fit.append((coil_t1, coil_r1, k))

    # calculation of the ratio of the number of iterations
    # that gave a larger coupling coefficient to the total number of iterations
    k_max = max(fit, key=lambda x: x[2])[2]
    n_k_max = 0
    thr_equal = 0.1
    for el in fit:
        if np.abs((k_max - el[2]) / k_max) / k_max < thr_equal:
            n_k_max += 1
    ratio = n_k_max / len(fit)

    # calculate characteristics of series
    mean_agb = (np.average(arr_all), np.average(arr_good), np.average(arr_bad))
    median_agb = (np.median(arr_all), np.median(arr_good), np.mean(arr_bad))
    deviation_agb = (np.std(arr_all), np.std(arr_good), np.std(arr_bad))

    # min and max values of the coupling coefficient
    # and their corresponding coils
    fit_values = (min(fit, key=lambda x: x[2]), max(fit, key=lambda x: x[2]))

    return fit_values, mean_agb, median_agb, deviation_agb, arr_time, ratio


def main():
    coil_t = np.linspace(0.02, 0.05, 4)  # transmitting coil
    coil_r = np.linspace(0.03, 0.09, 4)  # receiving coil
    r_turn = 0.0004  # radius of coil turns

    # distance
    d = 0.01

    FLAG_RUN_MULTIITER = True
    if not FLAG_RUN_MULTIITER:
        '''
        --------------------------------------------------------------------
        Testing the algorithm for Steepest Ascent Hill Climbing in one run.
        --------------------------------------------------------------------
        '''
        coil_t, coil_r, k, allm, badm, goodm = steepest_ascent_hill_climbing(coil_1=coil_t, r1_turn=r_turn,
                                                                             coil_2=coil_r, r2_turn=r_turn,
                                                                             d=d)

        print(f"The resulting value of the coupling coefficient: k={k}\n"
              f"for coil_t={coil_t} м and coil_r={coil_r} м\n")
        print(f"All mutation: {allm}")
        print(f"Good mutation: {goodm}")
        print(f"Bad mutation: {badm}\n")
    elif FLAG_RUN_MULTIITER:
        '''
        ---------------------------------------------------------------------
        Testing the algorithm Steepest Ascent Hill Climbing on several runs.
        ---------------------------------------------------------------------
        '''
        iterations = 1000
        fit_values, mean_agb, median_agb, deviation_agb, times, ratio = launch(
            iterations=iterations,
            coil_t=coil_t, rt_turn=r_turn,
            coil_r=coil_r, rr_turn=r_turn,
            d=d)

        print(f"Average good mutation: {mean_agb[1]}")
        print(f"Average bad mutation: {mean_agb[2]}")
        print(f"Average all mutation: {mean_agb[0]}\n")

        print(f"Median good mutation: {median_agb[1]}")
        print(f"Median bad mutation: {median_agb[2]}")
        print(f"Median all mutation: {median_agb[0]}\n")

        print(f"Deviation good mutation: {deviation_agb[1]}")
        print(f"Deviation bad mutation: {deviation_agb[2]}")
        print(f"Deviation all mutation: {deviation_agb[0]}\n")

        print(f"N(kmax)/N = {ratio}")
        print(f"Average time to find a value: {np.average(times)} sec.")

        print(f"Max of couple coefficient: {fit_values[1][2]}\n"
              f"    for coil_t={fit_values[1][0]} м;\n"
              f"    for coil_r={fit_values[1][1]} м;\n")
        print(f"Min of couple coefficient: {fit_values[0][2]}\n"
              f"    for coil_t={fit_values[0][0]} м;\n"
              f"    for coil_r={fit_values[0][1]} м;\n")


if __name__ == "__main__":
    main()
