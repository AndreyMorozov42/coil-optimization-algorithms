import numpy as np

from functions import mutual_inductance, self_inductance, coupling_coefficient
from tools.mutation import mutation_lb

def adaptive_hill_climbing(coil_t, coil_r, r_turn, d):
    all_mutation = 0
    good_mutation = 0
    bad_mutation = 0

    thr = 0.5e-3

    fit_k = coupling_coefficient(coil_1=coil_t, r1_turn=r_turn,
                                 coil_2=coil_r, r2_turn=r_turn,
                                 dist=d)

    # random generate two internal turns in coil
    coil_tq = coil_t.copy()
    fit_kq = 0

    i = 0
    rm = np.random.random()

    print(rm)

    while np.abs(fit_kq - fit_k) > thr:

        i += 1

        if fit_kq > fit_k:
            coil_t = coil_tq.copy()
            fit_k = fit_kq.copy()
            good_mutation += 1
            print(f"{i} find good mutation")
        elif fit_kq != 0:
            bad_mutation += 1
            # print(f"{i} find bad mutation")

        for ind in range(1, len(coil_tq) - 1):
            r = np.random.random()
            if r < rm:
                coil_tq[ind] = mutation_lb(start=coil_t[ind - 1] + 2 * r_turn,
                                       finish=coil_t[ind + 1] - 2 * r_turn,
                                       x=coil_tq[ind].copy(),
                                       dr_max=0.05)

        fit_kq = coupling_coefficient(coil_1=coil_tq, r1_turn=r_turn,
                                      coil_2=coil_r, r2_turn=r_turn,
                                      dist=d)

        all_mutation += 1

        if i > 500:
            break

    if fit_kq > fit_k:
        coil_t = coil_tq.copy()
        fit_k = fit_kq.copy()
        good_mutation += 1
        print(f"{i} find good mutation")
    else:
        bad_mutation += 1

    print(f"All iterations: {i}")
    return coil_t.copy(), fit_k.copy(), all_mutation, bad_mutation, good_mutation

def main():
    coil_t = np.linspace(start=0.01, stop=0.05, num=4)
    coil_r = np.linspace(start=0.01, stop=0.05, num=4)
    r_turn = 0.0004
    d = 0.01

    coil_t, k, allm, badm, goodm = adaptive_hill_climbing(
        coil_t=coil_t,
        coil_r=coil_r,
        r_turn=r_turn,
        d=d
    )

    print(f"coil_t={coil_t} k={k}")
    print(f"All mutation {allm}")
    print(f"Good mutation {goodm}")
    print(f"Bad mutation {badm}")


if __name__ == "__main__":
    main()

