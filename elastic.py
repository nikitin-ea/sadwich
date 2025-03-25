# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:08:14 2024

@author: devoi
"""
import json
import numpy as np


def interpolate(z, param, part, typ):
    zi = np.asarray(param[part]["zi"])
    hi = np.asarray(param[part][typ])
    return np.interp(z, zi, hi)


def tanh_taper(z):
    return 0.5 * ((5 - 0.05) * np.tanh(0.1 *(z - 50.0)) + 5.05)


def get_nondim_param(z, param):
    h1 = interpolate(z, param, "top_face", "hi")
    h2 = interpolate(z, param, "bot_face", "hi")
    h3 = interpolate(z, param, "core", "hi")
    E1 = interpolate(z, param, "top_face", "Ei")
    E2 = interpolate(z, param, "bot_face", "Ei")
    E3 = interpolate(z, param, "core", "Ei")

    h = h1 + h2 + h3
    E = (E1 * h1 + E2 * h2 + E3 * h3) / h

    al1 = E1 * h1 / (E * h)
    al2 = E2 * h2 / (E * h)
    al3 = E3 * h3 / (E * h)
    t1 = h1 / h
    t2 = h2 / h
    t3 = h3 / h

    return [al1, al2, al3], [t1, t2, t3]


def get_membrane_stiff(z, param):
    als, _ = get_nondim_param(z, param)
    h1 = interpolate(z, param, "top_face", "hi")
    h2 = interpolate(z, param, "bot_face", "hi")
    h3 = interpolate(z, param, "core", "hi")
    E1 = interpolate(z, param, "top_face", "Ei")
    E2 = interpolate(z, param, "bot_face", "Ei")
    E3 = interpolate(z, param, "core", "Ei")
    b = param["general"]["b"]

    h = h1 + h2 + h3
    E = (E1 * h1 + E2 * h2 + E3 * h3) / h
    B = E * b * h
    B1 = als[0] * B
    B2 = als[1] * B
    B3 = als[2] * B
    return [B1, B2, B3], B, h


def get_coupling_stiff(z, param):
    als, ts = get_nondim_param(z, param)
    Bs, B, h = get_membrane_stiff(z, param)

    K1 =  1 / 2 * als[0] * (ts[0] + ts[2]) * B * h
    K2 = -1 / 2 * als[1] * (ts[1] + ts[2]) * B * h
    return [K1, K2]


def get_bending_stiff(z, param):
    als, ts = get_nondim_param(z, param)
    Bs, B, h = get_membrane_stiff(z, param)

    D0 = 1 / 12 * B * h**2

    D1 = als[0] * (4 * ts[0]**2 + 6 * ts[0] * ts[2] + 3 * ts[2]**2) * D0
    D2 = als[1] * (4 * ts[1]**2 + 6 * ts[1] * ts[2] + 3 * ts[2]**2) * D0
    D3 = als[2] * ts[2]**2 * D0

    return [D1, D2, D3]


def get_stiff_matrix(z, param, transpose=True):
    Bs, B, h = get_membrane_stiff(z, param)
    Ks = get_coupling_stiff(z, param)
    Ds = get_bending_stiff(z, param)
    c = interpolate(z, param, "core", "hi") / 2
    G = interpolate(z, param, "core", "Gi")
    b = param["general"]["b"]

    print(f"{Bs=}")
    print(f"{Ks=}")
    print(f"{Ds=}")

    Gm = np.array([[B,
                    (Bs[0] - Bs[1]) * c,
                    Ks[0] + Ks[1]],
                   [(Bs[0] - Bs[1]) * c,
                    Ds[2] + c**2 * (Bs[0] + Bs[1]),
                    Ds[2] + c * (Ks[0] - Ks[1])],
                   [Ks[0] + Ks[1],
                    Ds[2] + c * (Ks[0] - Ks[1]),
                    Ds[0] + Ds[1] + Ds[2]]])

    T = np.array([[1, -(Bs[0] - Bs[1]) * c / B, -(Ks[0] + Ks[1]) / B],
                  [0,                        1,                    0],
                  [0,                        0,                    1]])
    C_sh = G * b * 2 * c
    if transpose:
        Gm = T.T @ Gm @ T
    return Gm, C_sh


def get_nondim_complex(z, param):
    G, C_sh = get_stiff_matrix(z, param)
    g = G[2, 2]**2 / (G[1, 1] * G[2, 2] - G[1, 2]**2)
    l = G[1, 2] / G[2, 2]
    a = C_sh * param["general"]["L"]**2 / G[2, 2]
    return g, l, a


if __name__ == "__main__":
    fname = "al-polyurethane.json"
    with open(fname, mode="r+", encoding="utf-8") as file:
        param = json.load(file)

    print(get_stiff_matrix(0.0, param, False))
