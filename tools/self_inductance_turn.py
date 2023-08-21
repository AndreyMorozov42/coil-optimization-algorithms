import numpy as np


def self_inductance_turn(r, r_turn):
    # vacuum permeability
    mu0 = 4 * np.pi * 10 ** (-7)
    a = np.log(8 * r / r_turn)
    b = r_turn ** 2 / (8 * r ** 2)
    return mu0 * r * (a - 7 / 4 + b * (a + 1 / 3))