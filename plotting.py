# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:35:16 2024

@author: devoi
"""
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['font.size'] = 12
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.grid"] = True


def plot_solution(z_eval, Ys):
    default_cycler = cycler(color=['k', 'r', 'g', 'b'])
    if isinstance(Ys, np.ndarray):
        Ys = (Ys, )

    fig, ax = plt.subplots(ncols=2, nrows=3, dpi=600, figsize=(6, 6))
    ax[0, 0].set_xlabel(r"$\xi$")
    ax[0, 0].set_ylabel(r"$\overline{v}$")
    ax[0, 0].margins(x=0)
    ax[0, 0].set_prop_cycle(default_cycler)

    ax[1, 0].set_xlabel(r"$\xi$")
    ax[1, 0].set_ylabel(r"$\overline{\vartheta}$")
    ax[1, 0].margins(x=0)
    ax[1, 0].set_prop_cycle(default_cycler)

    ax[0, 1].set_xlabel(r"$\xi$")
    ax[0, 1].set_ylabel(r"$\overline{\gamma}$")
    ax[0, 1].margins(x=0)
    ax[0, 1].set_prop_cycle(default_cycler)

    ax[1, 1].set_xlabel(r"$\xi$")
    ax[1, 1].set_ylabel(r"$\overline{Q}$")
    ax[1, 1].margins(x=0)
    ax[1, 1].set_prop_cycle(default_cycler)

    ax[2, 0].set_xlabel(r"$\xi$")
    ax[2, 0].set_ylabel(r"$\overline{M}$")
    ax[2, 0].margins(x=0)
    ax[2, 0].set_prop_cycle(default_cycler)

    ax[2, 1].set_xlabel(r"$\xi$")
    ax[2, 1].set_ylabel(r"$\overline{H}$")
    ax[2, 1].margins(x=0)
    ax[2, 1].set_prop_cycle(default_cycler)

    for Yi in Ys:
        ax[0, 0].plot(z_eval, Yi[0])
        ax[1, 0].plot(z_eval, Yi[1])
        ax[0, 1].plot(z_eval, Yi[2])
        ax[1, 1].plot(z_eval, Yi[3])
        ax[2, 0].plot(z_eval, Yi[4])
        ax[2, 1].plot(z_eval, Yi[5])

    fig.tight_layout()


def print_solution(z_eval, Y, step=0.1):
    length = z_eval[-1]
    n_rows = int(1 / step) + 1
    z_print = np.linspace(0, length, n_rows)

    print("┌" + "─" * (13 * 7 - 1) + "┐")
    print(f"│ {'ξ':^10s} │ {'v':^10s} │ {'ϑ':^10s} "
          f"│ {'γ':^10s} │ {'Q':^10s} "
          f"│ {'M':^10s} │ {'H':^10s} │")
    print("╞" + "═" * (13 * 7 - 1) + "╡")
    for i, zi in enumerate(z_print):
        v = np.interp(zi, z_eval, Y[0])
        t = np.interp(zi, z_eval, Y[1])
        g = np.interp(zi, z_eval, Y[2])
        Q = np.interp(zi, z_eval, Y[3])
        M = np.interp(zi, z_eval, Y[4])
        H = np.interp(zi, z_eval, Y[5])
        print(f"│ {zi: 9.3E} │ {v: 9.3E} │ {t: 9.3E} │ {g: 9.3E} │ {Q: 9.3E} "
              f"│ {M: 9.3E} │ {H: 9.3E} │")
        if i != z_eval.size - 1:
            print("├" + "─" * (13 * 7 - 1) + "┤")
    print("└" + "─" * (13 * 7 - 1) + "┘")
