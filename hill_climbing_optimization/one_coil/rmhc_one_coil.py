import numpy as np

from hill_climbing_optimization.functions import coupling_coefficient
from tools.mutation import mutation_lb


def random_mutation_hill_climbing(coil_t, coil_r, r_turn, d):
    # mutation counter
    all_mutation = 0
    good_mutation = 0
    bad_mutation = 0

    # objective function increment threshold
    thr = 1e-3

    fit_k = coupling_coefficient(coil_1=coil_t, r1_turn=r_turn,
                                 coil_2=coil_r, r2_turn=r_turn,
                                 dist=d)

    print(f"Initial Coupling coefficient {fit_k} for coils:\n"
          f"coil1 = {coil_t}\n"
          f"coil2 = {coil_r}\n")

    # initialization of initial values
    fit_kq = 0
    coil_tq = coil_t.copy()

    i = 0                   # iteration counter
    limit = 1000

    while np.abs(fit_kq - fit_k) > thr and i != limit:

        i += 1

        if fit_kq > fit_k:
            print(f"{i}: Found a new maximum value of the coupling coefficient: {fit_kq}")
            coil_t = coil_tq.copy()
            fit_k = fit_kq.copy()
            good_mutation += 1
        elif fit_kq != 0:
            bad_mutation += 1

        # random index of the internal turn of the transmitting coil
        ind_q = np.random.randint(low=1, high=len(coil_t) - 1)

        coil_tq = coil_t.copy()
        # mutate internal turn of the transmitting coil
        coil_tq[ind_q] = mutation_lb(start=coil_t[ind_q - 1] + 2 * r_turn,
                                     finish=coil_t[ind_q + 1] - 2 * r_turn,
                                     x=coil_tq[ind_q].copy())

        fit_kq = coupling_coefficient(coil_1=coil_tq, r1_turn=r_turn,
                                      coil_2=coil_r, r2_turn=r_turn,
                                      dist=d)
        all_mutation += 1

    if fit_kq > fit_k:
        print(f"{i}: Found a new maximum value of the coupling coefficient: {fit_kq}")
        coil_t = coil_tq.copy()
        fit_k = fit_kq.copy()
        good_mutation += 1
    else:
        bad_mutation += 1

    print(f"Stop at {i} iterations\n")

    return coil_t.copy(), fit_k.copy(), all_mutation, bad_mutation, good_mutation


def launch(iterations, coil_t, coil_r, r_turn, d):
    # array of mutation counters
    arr_good = np.array([])
    arr_bad = np.array([])
    arr_all = np.array([])
    fit = []


    for _ in range(iterations):

        coil, k, allm, badm, goodm = random_mutation_hill_climbing(
            coil_t=coil_t,
            coil_r=coil_r,
            r_turn=r_turn,
            d=d
        )

        arr_all = np.append(arr_all, allm)
        arr_good = np.append(arr_good, goodm)
        arr_bad = np.append(arr_bad, badm)

        fit.append((coil, k))

    # calculate characteristics of series
    mean_agb = (np.average(arr_all), np.average(arr_good), np.average(arr_bad))
    median_agb = (np.median(arr_all), np.median(arr_good), np.mean(arr_bad))
    deviation_agb = (np.std(arr_all), np.std(arr_good), np.std(arr_bad))

    # min and max values of the coupling coefficient
    # and their corresponding coils
    fit_values = (min(fit, key=lambda x: x[1]), max(fit, key=lambda x: x[1]))

    return fit_values, mean_agb, median_agb, deviation_agb


def main():

    coil_t = np.linspace(start=0.02, stop=0.05, num=4)  # transmitting coil
    coil_r = np.linspace(start=0.03, stop=0.09, num=4)  # receiving coil
    r_turn = 0.0004                                     # radius of coil turns

    # distance
    d = 0.01

    FLAG_RUN_MULTIITER = True
    if not FLAG_RUN_MULTIITER:
        '''
        ------------------------------------------------------------
        Testing the algorithm for Hill Climbing in one run.
        ------------------------------------------------------------
        '''
        coil_t, k, allm, badm, goodm = random_mutation_hill_climbing(
            coil_t=coil_t,
            coil_r=coil_r,
            r_turn=r_turn,
            d=d
        )
        print(f"The resulting value of the coupling coefficient: k={k}\n"
              f"for coil_t={coil_t} м and coil_r={coil_r} м\n")
        print(f"All mutation: {allm}")
        print(f"Good mutation: {goodm}")
        print(f"Bad mutation: {badm}\n")
    elif FLAG_RUN_MULTIITER:
        '''
        ------------------------------------------------------------
        Testing the algorithm for climbing
        to the top of a hill on several runs.
        ------------------------------------------------------------
        '''
        iterations = 1000
        fit_values, mean_agb, median_agb, deviation_agb = launch(
            iterations=iterations,
            coil_t=coil_t,
            coil_r=coil_r,
            r_turn=r_turn,
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

        print(f"Max of couple coefficient: {fit_values[1][1]}\n"
              f"    for coil_t={fit_values[1][0]}\n")
        print(f"Min of couple coefficient: {fit_values[0][1]}\n"
              f"    for coil_t={fit_values[0][0]}")


if __name__ == "__main__":
    main()

