import numpy as np
import matplotlib.pyplot as plt

from tools.mutual_inductance import mutual_inductance
from tools.coupling_coefficient import coupling_coefficient


def show_plot(x, y, x_label="x", y_label="y", title=None):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    plt.grid()
    plt.show()


def main():
    # transmitting coil
    coil_t = np.linspace(0.028, 0.028 / 0.4, 4)

    r_turn = 0.0004

    # receiving coil
    coils_r = np.linspace(0.028, 0.028 / 0.4, 4) + np.zeros((50, 4))
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


if __name__ == "__main__":
    main()
