import numpy as np

from functions import mutual_inductance, self_inductance, coupling_coefficient
from tools.mutation import mutation_lb


def steepest_ascent_hill_climbing(coil_t, coil_r, r_turn, d):

    all_mutation = 0
    good_mutation = 0
    bad_mutation = 0

    thr = 0.5e-3

    fit_k = coupling_coefficient(coil_1=coil_t, r1_turn=r_turn,
                                 coil_2=coil_r, r2_turn=r_turn,
                                 dist=d)

    # random generate two internal turns in coil
    fit_kq = 0
    coil_tq = coil_t.copy()
    array = []
    i = 0

    while np.abs(fit_kq - fit_k) > thr:

        i += 1

        if array != [] and fit_k > array[0][1]:
            print(f"{i} find good mutation")
            coil_t = array[0][0].copy()
            fit_k = fit_kq.copy()
            good_mutation += 1
        elif array != []:
            bad_mutation += 1

        array = []

        for ind in range(1, len(coil_tq) - 1):
            coil_tq = coil_t.copy()
            coil_tq[ind] = mutation_lb(start=coil_t[ind - 1] + 2 * r_turn,
                                       finish=coil_t[ind + 1] - 2 * r_turn,
                                       x=coil_tq[ind].copy(),
                                       dr_max=0.05)
            fit_kq = coupling_coefficient(coil_1=coil_tq, r1_turn=r_turn,
                                          coil_2=coil_r, r2_turn=r_turn,
                                          dist=d)
            array.append((coil_tq.copy(), fit_kq.copy()))
        all_mutation += 1

        if i > 500:
            break

        array.sort(key=lambda x: x[1], reverse=True)

    print(f"All iterations: {i}")

    if fit_k > array[0][1]:
        print("find good mutation")
        coil_t = array[0][0].copy()
        fit_k = fit_kq.copy()
        good_mutation += 1
    else:
        bad_mutation += 1

    return coil_t.copy(), fit_k.copy(), all_mutation, bad_mutation, good_mutation


def launch(iterations, coil_t, coil_r, r_turn, d):
    bad_mutation = []
    good_mutation = []
    all_mutation = []
    fit_k = []

    allm = 0
    badm = 0
    goodm = 0
    k = 0

    for _ in range(iterations):

        coil_t, k, allm, badm, goodm = steepest_ascent_hill_climbing(
            coil_t=coil_t,
            coil_r=coil_r,
            r_turn=r_turn,
            d=d
        )

        bad_mutation.append(badm)
        good_mutation.append(goodm)
        all_mutation.append(allm)
        fit_k.append(k)

    bad_mutation = np.mean(np.array(bad_mutation))
    good_mutation = np.mean(np.array(good_mutation))
    all_mutation = np.mean(np.array(all_mutation))

    return max(fit_k), all_mutation, bad_mutation, good_mutation

def main():
    coil_t = np.linspace(start=0.02, stop=0.05, num=4)
    coil_r = np.linspace(start=0.03, stop=0.09, num=4)
    r_turn = 0.0004
    d = 0.01

    # coil_t, k, allm, badm, goodm = steepest_ascent_hill_climbing(
    #     coil_t=coil_t,
    #     coil_r=coil_r,
    #     r_turn=r_turn,
    #     d=d
    # )
    #
    # print(f"coil_t={coil_t} k={k}")
    # print(f"All mutation {allm}")
    # print(f"Good mutation {goodm}")
    # print(f"Bad mutation {badm}")

    k, a, g, b = launch(iterations=1000, coil_t=coil_t, coil_r=coil_r, r_turn=r_turn, d=d)

    print(f"All mutation {a}")
    print(f"Good mutation {g}")
    print(f"Bad mutation {b}")
    print(f"Max k = {k}")


if __name__ == "__main__":
    main()

