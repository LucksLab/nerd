# nerd/utils/plotting.py

import matplotlib.pyplot as plt
from rich.table import Table
from rich.console import Console
import numpy as np

console = Console()


# === 1. Plot fit overlay ===
def plot_fit_with_data(x, y, result, title="Fit", xlabel="x", ylabel="y", save_path=None):
    """
    Plot raw data and lmfit model fit.
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Data", color="black")
    ax.plot(x, result.best_fit, label="Fit", color="blue")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


# === 2. Arrhenius plot: ln(k) vs 1/T ===
def plot_arrhenius(x, y, result, save_path=None):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="black", label="Data")
    ax.plot(x, result.best_fit, color="red", label="Linear fit")

    ax.set_xlabel("1 / Temperature (1/K)")
    ax.set_ylabel("ln(k)")
    ax.set_title("Arrhenius Plot")
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


# === 3. Melt curve: normalized kobs vs T ===
def plot_melt_curve(T, k_norm, result=None, save_path=None):
    fig, ax = plt.subplots()
    ax.scatter(T, k_norm, color="black", label="Data")

    if result:
        T_fit = np.linspace(min(T), max(T), 100)
        k_fit = result.eval(T=T_fit)
        ax.plot(T_fit, k_fit, color="blue", label="Fit")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Normalized k_obs")
    ax.set_title("Two-State Melting Fit")
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


# === 4. Rich table display for parameters ===
def display_fit_params(result, title="Fit Results"):
    table = Table(title=title)
    table.add_column("Parameter")
    table.add_column("Value", justify="right")
    table.add_column("± Error", justify="right")

    for name, param in result.params.items():
        table.add_row(name, f"{param.value:.4g}", f"{param.stderr:.2g}" if param.stderr else "—")

    console.print(table)