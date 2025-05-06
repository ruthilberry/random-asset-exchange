"""visualizations.py – Herramientas para visualizar simulaciones del Yard‑Sale Model.

Incluye:
    1. Gini vs. tiempo – muestra cómo evoluciona la desigualdad.
    2. Histograma de riqueza – observa la distribución, colas ricas o condensación.
    3. Curva de Lorenz – visualiza la desigualdad acumulativa.
    4. Comparación de políticas – compara Gini bajo diferentes parámetros fiscales.
    5. Animación de histogramas – ve cómo cambia la riqueza con el tiempo.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_gini(history, label=None):
    """1. Gini vs. tiempo
    Muestra la trayectoria temporal del coeficiente de Gini,
    útil para detectar si hay condensación o estabilidad.
    """
    t, g = history[:, 0], history[:, 1]
    plt.plot(t, g, label=label)
    plt.xlabel("Iteración")
    plt.ylabel("Coeficiente de Gini")
    if label:
        plt.legend()
    plt.title("Evolución del Gini")
    plt.tight_layout()
    plt.show()


def plot_wealth_histogram(wealth, log_y=True, bins=100):
    """2. Histograma de riqueza
    Visualiza la densidad de riqueza P(w). La escala logarítmica revela colas pesadas o acumulación.
    """
    plt.hist(wealth, bins=bins, density=True)
    if log_y:
        plt.yscale("log")
    plt.xlabel("Riqueza")
    plt.ylabel("P(w)")
    plt.title("Distribución de riqueza")
    plt.tight_layout()
    plt.show()


def plot_lorenz_curve(wealth):
    """3. Curva de Lorenz
    Representa la desigualdad acumulada. El área entre la curva y la diagonal es el Gini.
    """
    w = np.sort(wealth)
    cum = np.cumsum(w)
    L = cum / cum[-1]
    L = np.insert(L, 0, 0)
    x = np.linspace(0, 1, len(L))
    plt.plot(x, L, label="Curva de Lorenz")
    plt.plot(x, x, "--", label="Igualdad perfecta")
    plt.xlabel("Fracción acumulada de población")
    plt.ylabel("Fracción acumulada de riqueza")
    plt.title("Curva de Lorenz")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_policies(histories, labels):
    """4. Comparación de políticas
    Traza múltiples curvas Gini para comparar efectos de distintas tasas (ω, τ, χ…).
    """
    for hist, label in zip(histories, labels):
        t, g = hist[:, 0], hist[:, 1]
        plt.plot(t, g, label=label)
    plt.xlabel("Iteración")
    plt.ylabel("Gini")
    plt.title("Comparación de políticas")
    plt.legend()
    plt.tight_layout()
    plt.show()


def animate_histogram(models, steps, interval=200, bins=100, filename="wealth_evolution.gif"):
    """5. Animación de histogramas
    Muestra cómo evoluciona la distribución P(w) a lo largo del tiempo. Guarda un GIF.
    """
    fig, ax = plt.subplots()
    hist_data = []

    for model in models:
        w = model.wealth_distribution()
        hist, edges = np.histogram(w, bins=bins, density=True)
        hist_data.append((hist, edges))

    def update(frame):
        ax.clear()
        hist, edges = hist_data[frame]
        ax.bar(edges[:-1], hist, width=np.diff(edges), align="edge", alpha=0.7)
        ax.set_title(f"Paso {steps[frame]}")
        ax.set_xlabel("Riqueza")
        ax.set_ylabel("P(w)")
        ax.set_yscale("log")

    ani = animation.FuncAnimation(fig, update, frames=len(models), repeat=False)
    ani.save(filename, writer="pillow", fps=2)
    print(f"Animación guardada en {filename}")
