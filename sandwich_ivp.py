# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:27:05 2024

@author: devoi
"""
import json
import numpy as np
from scipy.integrate import solve_ivp

from elastic import get_stiff_matrix
from plotting import plot_solution, print_solution


def load_param(file_path):
    with open(file_path, mode="r+", encoding="utf-8") as file:
        param = json.load(file)
    return param


def loading(z, Y, param):
    zi = np.array(param["distributed_load"]["zi"])
    qi = np.array(param["distributed_load"]["qi"])
    g = np.zeros_like(Y)
    g[3, 3] = -np.interp(z, zi, qi)
    return g


def sandwich_bend_rhs(z, Y, q, param):
    Y = Y.reshape((6, 4))
    G, C_sh = get_stiff_matrix(param)
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
    g = loading(z, Y, param)

    dYdt = A @ Y + g
    return dYdt.reshape(-1)


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


def get_init_cond(param):
    zero_idx = get_idx_by_bc("left_end", param)
    nonzero_idx = [x for x in range(6) if x not in zero_idx]
    Y0 = np.zeros((6, 4))
    for i, j in enumerate(nonzero_idx):
        Y0[j, i] = 1

    Y_a0 = np.array([param["left_end"]["v"],
                     param["left_end"]["theta"],
                     param["left_end"]["gamma"],
                     param["left_end"]["Q"],
                     param["left_end"]["M"],
                     param["left_end"]["H"]])
    Y0[:,-1] = -Y_a0
    return Y0.reshape(-1)


def get_init_param_matrix(Y, param):
    zero_idx = get_idx_by_bc("right_end", param)
    A = np.eye((4))
    g = np.ones((4, ))
    for j, idx in enumerate(zero_idx):
        A[j, :] = np.array([Y[idx, 0, -1], Y[idx, 1, -1], Y[idx, 2, -1], 0])
        g[j] = Y[idx, 3, -1]

    Y_b0 = np.array([param["right_end"]["v"],
                     param["right_end"]["theta"],
                     param["right_end"]["gamma"],
                     param["right_end"]["Q"],
                     param["right_end"]["M"],
                     param["right_end"]["H"]])
    g[:-1] += Y_b0[zero_idx]
    g[-1] *= -1
    return A, g


def get_solution(param, npts):
    Y0 = get_init_cond(param)

    z_span = (0, param["general"]["L"])
    z_eval = np.linspace(*z_span, npts)
    sol = solve_ivp(lambda z, Y: sandwich_bend_rhs(z, Y, 1, param),
                    z_span, Y0, rtol=1e-10, atol=1e-6,
                    t_eval=z_eval, method="RK23")

    Y = sol.y.reshape((6, 4, npts))

    A, g = get_init_param_matrix(Y, param)
    C = np.linalg.solve(A, g)

    return z_eval, C @ Y


if __name__ == "__main__":
    fname = "al-polyurethane.json"
    param = load_param(fname)

    z_eval, Y = get_solution(param, 501)

    plot_solution(z_eval, Y)
    print_solution(z_eval, Y)
