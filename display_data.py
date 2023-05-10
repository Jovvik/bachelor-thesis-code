import argparse

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D

import dataset
from simulation import Simulations

cm = plt.get_cmap('tab10')


def display_position(data: Simulations):
    def make_circle(r):
        t = np.arange(0, np.pi * 2.0, 0.01)
        t = t.reshape((len(t), 1))
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.hstack((x, y))

    legend = []
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(-2200, 2000)
    ax.set_ylim(-1500, 1500)
    ax.set_aspect('equal')
    ax.set_xlabel('$x_1$, м')
    ax.set_ylabel('$x_2$, м')

    ax.add_patch(plt.Circle(data.start_position, 20, color=cm(1)))
    legend.append(Line2D([0], [0], color=cm(1), marker='o', linestyle='None', label='Начальное положение'))

    ax.plot(data.beacon_positions[:, 0], data.beacon_positions[:, 1], "x", color=cm(0))
    legend.append(Line2D([0], [0], color=cm(0), marker='x', linestyle='None', label='Маяки'))

    states = data.simulations[0].states
    ax.plot([s.real_position[0] for s in states], [s.real_position[1] for s in states], color=cm(1))
    legend.append(Line2D([0], [0], color=cm(1), label='Траектория'))

    ax.add_patch(plt.Circle(data.start_position, 3 * data.sigma_position, color=cm(4), linestyle='--', fill=None))
    legend.append(Line2D([0], [0], color=cm(4), linestyle='--', label=r'Априорная неопределенность положения (3$\sigma$)'))

    for i in range(3):
        distance = np.linalg.norm(data.start_position - data.beacon_positions[i])
        inside_vertices = make_circle(distance - 3 * data.sigma_measurement) + data.beacon_positions[i]
        outside_vertices = make_circle(distance + 3 * data.sigma_measurement) + data.beacon_positions[i]
        codes = np.ones(len(inside_vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
        codes[0] = mpath.Path.MOVETO
        vertices = np.concatenate((outside_vertices[::-1], inside_vertices[::1]))
        all_codes = np.concatenate((codes, codes))
        path = mpath.Path(vertices, all_codes)
        patch = mpatches.PathPatch(path, facecolor='black', edgecolor='none', alpha=0.1)
        ax.add_patch(patch)
        ax.add_patch(plt.Circle(data.beacon_positions[i], distance, color='black', linestyle='--', fill=None, alpha=0.1))
    legend.append(Line2D([0], [0], color='black', linestyle='--', label=r'Неопределенность расстояния ($\pm 3 \sigma$) до маяка', alpha=0.1))
    ax.legend(handles=legend)
    plt.show()


def display_distances(data: Simulations):
    pass


def display_attitude(data: Simulations):
    pass


def display_data(data: Simulations):
    display_position(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Display data')
    parser.add_argument('filename')
    # parser.add_argument('-s', '--samples')
    args = parser.parse_args()
    data = dataset.load_data(args.filename)
    display_data(data)
