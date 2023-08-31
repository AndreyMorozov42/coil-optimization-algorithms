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
    coil_t = np.array([0.028, 0.028 / 0.4])

    # receiving coils
    coil_r = np.array([0.4 * np.linspace(0.01, 0.1, 50)]).T
    coil_rl = np.array([np.linspace(0.01, 0.1, 50)]).T
    coils_r = np.hstack([coil_r, coil_rl])

    # coil winding thickness
    r_turn = 0.0004
    # axial distance
    d = 0.005
    # lateral distance
    ro = [0]

    # calculation mutual inductance and couple
    m = np.zeros(coils_r.shape[0])
    k = np.zeros(coils_r.shape[0])
    for ind_c in range(coils_r.shape[0]):
        coil_r = coils_r[ind_c]
        m[ind_c] = mutual_inductance(coil_1=coil_t, coil_2=coil_r, d=d, ro=ro)
        k[ind_c] = coupling_coefficient(coil_1=coil_t, coil_2=coil_r, r_turn=r_turn, d=d)

    # show distribution of mutual inductance and couple coefficient
    show_plot(x=coils_r.T[0], y=m * 1e6, x_label="r, м", y_label="M, мкГн", title="Mutual Inductance")
    show_plot(x=coils_r.T[0], y=k, x_label="r, м", y_label="k", title="Couple Coefficient")


if __name__ == "__main__":
    main()

