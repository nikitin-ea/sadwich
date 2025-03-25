# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:27:05 2024

@author: devoi
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

from elastic import get_stiff_matrix

plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['font.size'] = 12
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.grid"] = True


class State():
    W = 0
    GAMMA = 1
    THETA = 2
    V = 3
    N = 4
    H = 5
    M = 6
    Q = 7


def load_param(file_path):
    with open(file_path, mode="r+", encoding="utf-8") as file:
        param = json.load(file)
    return param


def loading(z, Y, param):
    zi = np.asarray(param["distributed_load"]["zi"])
    qi = np.asarray(param["distributed_load"]["qi"])
    g = np.zeros_like(Y)
    g[3] = -np.interp(z, zi, qi)
    return g


def sandwich_bend_rhs(z, Y, param):
    dYdt = np.zeros_like(Y)
    for i, (zi, Yi) in enumerate(zip(z, Y.T)):
        G, C_sh = get_stiff_matrix(zi, param)
        delta = G[1, 1] * G[2, 2] - G[1, 2]**2
        A25 =  G[1, 1] / delta
        A26 = -G[1, 2] / delta
        A36 =  G[2, 2] / delta

        A = np.array([[0, -1,    0, 0,   0,   0],
                      [0,  0,    0, 0, A25, A26],
                      [0,  0,    0, 0, A26, A36],
                      [0,  0,    0, 0,   0,   0],
                      [0,  0,    0, 1,   0,   0],
                      [0,  0, C_sh, 0,   0,   0]])
        g = loading(zi, Yi, param)

        dYdt[:, i] = A @ Yi + g
    return dYdt


def get_idx_by_bc(end_type, param):
    match param[end_type]["bc"]:
        case "fix":
            return [0, 1, 5]
        case "slider":
            return [1, 3, 5]
        case "pin":
            return [0, 4, 5]
        case "free":
            return [3, 4, 5]
        case "fix+":
            return [0, 1, 2]
        case "slider+":
            return [1, 2, 3]
        case "pin+":
            return [0, 2, 4]
        case "free+":
            return [2, 3, 4]


def bc_bvp(Y_a, Y_b, param):
    idx_left = get_idx_by_bc("left_end", param)
    idx_right = get_idx_by_bc("right_end", param)
    Y_a0 = np.array([param["left_end"]["v"],
                     param["left_end"]["theta"],
                     param["left_end"]["gamma"],
                     param["left_end"]["Q"],
                     param["left_end"]["M"],
                     param["left_end"]["H"]])
    Y_b0 = np.array([param["right_end"]["v"],
                     param["right_end"]["theta"],
                     param["right_end"]["gamma"],
                     param["right_end"]["Q"],
                     param["right_end"]["M"],
                     param["right_end"]["H"]])
    bc_a = Y_a[idx_left] - Y_a0[idx_left]
    bc_b = Y_b[idx_right] - Y_b0[idx_right]
    return np.hstack((bc_a, bc_b))


def print_solution(z_eval, Y, step=0.1):
    length = z_eval[-1]
    n_rows = int(1 / step) + 1
    z_print = np.linspace(0, length, n_rows)

    print("┌" + "─" * (13 * 7 - 1) + "┐")
    print(f"│ {'z, мм':^10s} │ {'v, мм':^10s} │ {'ϑ, рад':^10s} "
          f"│ {'γ, рад':^10s} │ {'Q, Н':^10s} "
          f"│ {'M, Н·мм':^10s} │ {'H, Н·мм':^10s} │")
    print("╞" + "═" * (13 * 7 - 1) + "╡")
    for zi in z_print:
        v = np.interp(zi, z_eval, Y[0])
        t = np.interp(zi, z_eval, Y[1])
        g = np.interp(zi, z_eval, Y[2])
        Q = np.interp(zi, z_eval, Y[3])
        M = np.interp(zi, z_eval, Y[4])
        H = np.interp(zi, z_eval, Y[5])
        print(f"│ {zi: 9.3E} │ {v: 9.3E} │ {t: 9.3E} │ {g: 9.3E} │ {Q: 9.3E} "
              f"│ {M: 9.3E} │ {H: 9.3E} │")
        print("├" + "─" * (13 * 7 - 1) + "┤")
    print("└" + "─" * (13 * 7 - 1) + "┘")


if __name__ == "__main__":
    fname = "taper_composite.json"
    param = load_param(fname)

    NPTS = 226

    z_span = (0, param["general"]["L"])
    z_eval = np.linspace(*z_span, NPTS)

    sol = solve_bvp(lambda z, Y: sandwich_bend_rhs(z, Y, param),
                    lambda Y_a, Y_b: bc_bvp(Y_a, Y_b, param), x=z_eval,
                    y=np.zeros((6, z_eval.size)), verbose=2)
    Y = sol.sol(z_eval)

    fig, ax = plt.subplots(ncols=2, nrows=3, dpi=600, figsize=(6, 6))
    ax[0, 0].plot(z_eval, Y[0], "k")
    ax[0, 0].set_xlabel("$z$, мм")
    ax[0, 0].set_ylabel("$v$, мм")
    ax[0, 0].margins(x=0)

    ax[1, 0].plot(z_eval, Y[1], "k")
    ax[1, 0].set_xlabel("$z$, мм")
    ax[1, 0].set_ylabel(r"$\vartheta$")
    ax[1, 0].margins(x=0)

    ax[0, 1].plot(z_eval, Y[2], "k")
    ax[0, 1].set_xlabel("$z$, мм")
    ax[0, 1].set_ylabel(r"$\gamma$")
    ax[0, 1].margins(x=0)

    ax[1, 1].plot(z_eval, Y[3], "k")
    ax[1, 1].set_xlabel("$z$, мм")
    ax[1, 1].set_ylabel(r"$Q$, Н")
    ax[1, 1].margins(x=0)

    ax[2, 0].plot(z_eval, Y[4], "k")
    ax[2, 0].set_xlabel("$z$, мм")
    ax[2, 0].set_ylabel(r"$M$, Н$\cdot$мм")
    ax[2, 0].margins(x=0)

    ax[2, 1].plot(z_eval, Y[5], "k")
    ax[2, 1].set_xlabel("$z$, мм")
    ax[2, 1].set_ylabel(r"$H$, Н$\cdot$мм")
    ax[2, 1].margins(x=0)
    fig.tight_layout()

    print_solution(z_eval, Y)
