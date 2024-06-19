import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Callable

def plot_paths(f: Callable, optimizations_paths: Dict[str, list], xlim = (-2,2), ylim=(-2,2), levels=20, title="Contour Plot with Paths", output_path="../out/example_plots2.png"):
    """
    Create a contour plot and plot the optimization paths.

    Parameters:
    f : The objective function to plot.
    optimizations_paths : Dictionary containing paths of different methods {method_name: path}.
    xlim : Limits for the x-axis (xmin, xmax).
    ylim : Limits for the y-axis (ymin, ymax).
    levels : Number of contour levels.
    title : Title of the plot.
    output_path : The location of the saved plot.
    """
    
    # Calculate the dynamic limits based on the paths
    all_x_vals = []
    all_y_vals = []
    for path in optimizations_paths.values():
        all_x_vals.extend([p[0] for p in path])
        all_y_vals.extend([p[1] for p in path])

    x_min, x_max = min(all_x_vals), max(all_x_vals)
    y_min, y_max = min(all_y_vals), max(all_y_vals)

    # Add padding
    padding_factor = 0.2  # Increase padding to zoom out
    x_margin = (x_max - x_min) * padding_factor
    y_margin = (y_max - y_min) * padding_factor
    xlim = (x_min - x_margin, x_max + x_margin)
    ylim = (y_min - y_margin, y_max + y_margin)

    x_vals = np.linspace(xlim[0], xlim[1], 100)
    y_vals = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))[0]

    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, Z, levels=levels)
    plt.clabel(contour, inline=True, fontsize=8)

    for optimization_type_name, path in optimizations_paths.items():
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'o-', label=optimization_type_name)
    
    plt.xlim(xlim)  # Set the x limits dynamically
    plt.ylim(ylim)  # Set the y limits dynamically
    plt.title(title)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_function_values(optimizations_objective_values: Dict[str, list], title="Function Values vs Iterations", output_path="../out/example_plots3.png"):
    """
    Plot function values at each iteration to compare methods.

    Parameters:
    optimizations_objective_values : Dictionary containing function values of different methods {method_name: values}.
    title : Title of the plot.
    output_path : The location of the saved plot.

    """
    plt.figure(figsize=(8, 6))
    for optimization_type_name, objective_values in optimizations_objective_values.items():
        plt.plot(objective_values, '-', label=optimization_type_name)

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    
