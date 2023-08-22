import numpy as np

from tools.mutual_inductance import mutual_inductance
from tools.self_inductance import self_inductance


def coupling_coefficient(coil_1, coil_2, r_turn, d, ro=[0], fi=0):
    m = mutual_inductance(coil_1, coil_2, d, ro=ro, fi=fi)
    l1 = self_inductance(coil_1, r_turn)
    l2 = self_inductance(coil_2, r_turn)
    k = m / (l1 * l2) ** 0.5
    return k