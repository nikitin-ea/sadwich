# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:27:05 2024

@author: devoi
"""
import json
import numpy as np
from scipy.integrate import solve_bvp

from elastic import get_nondim_complex
from plotting import plot_solution, print_solution


def load_param(file_path):
    with open(file_path, mode="r+", encoding="utf-8") as file:
        param = json.load(file)
    return param


def loading(z, Y, param):
    zi = np.asarray(param["distributed_load"]["zi"]) / param["general"]["L"]
    qi = np.asarray(param["distributed_load"]["qi"])
    f = np.zeros_like(Y)
    f[3, :] = -np.interp(z, zi, qi)
    return f


def sandwich_bend_rhs(z, Y, ndim_param):
    g, l, a = ndim_param

    A = np.array([[0, -1, 0, 0,            0,      0],
                  [0,  0, 0, 0, 1 + l**2 * g, -l * g],
                  [0,  0, 0, 0,       -l * g,      g],
                  [0,  0, 0, 0,            0,      0],
                  [0,  0, 0, 1,            0,      0],
                  [0,  0, a, 0,            0,      0]])
    f = loading(z, Y, param)

    dYdt = A @ Y + f
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


def get_solution(ndim_param, param, npts):
    z_eval = np.linspace(0, 1, npts)

    sol = solve_bvp(lambda z, Y: sandwich_bend_rhs(z, Y, ndim_param),
                    lambda Y_a, Y_b: bc_bvp(Y_a, Y_b, param), x=z_eval,
                    y=np.zeros((6, z_eval.size)), verbose=2)
    Y = sol.sol(z_eval)
    return z_eval, Y


def convert_from_ndim(z_eval, Y, q0, L, g33):
    B = np.diag([q0 * L**4 / g33,
                 q0 * L**3 / g33,
                 q0 * L**3 / g33,
                 q0 * L,
                 q0 * L**2,
                 q0 * L**2])
    return z_eval * L, B @ Y


def gamma(ndim_param, z):
    G, l, A = ndim_param
    return l / A * (1 )


if __name__ == "__main__":
    fname = "al-polyurethane.json"
    param = load_param(fname)

    ndim_param = [300.0, 0.8, 0.9]

    z_eval, Yi = get_solution(ndim_param, param, 1001)

    plot_solution(z_eval, Yi)
    print_solution(z_eval, Yi)
