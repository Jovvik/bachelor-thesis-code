import argparse
import pickle
import random
import os

import numpy as np
from numpy import random as npr
from rich.progress import track

from simulation import Simulation, State, Simulations

""" Simulation parameters """
BEACONS = np.array([[-1000., 250.], [-1000., 0.], [-1000., -250.]])
STARTING_POSITION = np.array([0., 0.])
STARTING_ATTITUDE = np.deg2rad(5)
VELOCITY_PARALLEL = 3
VELOCITY_PERPENDICULAR = 1
MODELLING_TIME = 200
DISCRETIZATION = 1
SIGMA_POSITION = 250.
SIGMA_ATTITUDE = np.deg2rad(0.5)
SIGMA_VELOCITY = 0.1
SIGMA_MEASUREMENT = 15.
SIMULATION_COUNT = 10000
VELOCITY_PARALLEL_NOISE = 2, 4
VELOCITY_PERPENDICULAR_NOISE = 0.2, 0.5
ATTITUDE_NOISE_DEG = -10, 20
STARTING_POSITION_NOISE = -200, 200


def simulate_trajectory(position: np.ndarray, attitude: float, velocity_parallel: float,
                        velocity_perpendicular: float) -> Simulation:
    initial_measured_position = position + npr.normal(0, SIGMA_POSITION, 2)
    meas_attitude = attitude + npr.normal(0, SIGMA_ATTITUDE)
    states = []
    for _ in range(0, MODELLING_TIME, DISCRETIZATION):
        # apply parallel velocity
        position += np.array([
            velocity_parallel * np.sin(attitude),
            velocity_parallel * np.cos(attitude)
        ])
        # apply perpendicular velocity
        position += np.array([
            velocity_perpendicular * np.cos(-attitude),
            velocity_perpendicular * np.sin(-attitude)
        ])
        distances = np.linalg.norm(BEACONS - position, axis=1)
        t = SIGMA_VELOCITY * DISCRETIZATION
        states.append(State(position.copy(),
                            attitude,
                            meas_attitude,
                            velocity_parallel + npr.normal(0, t),
                            velocity_perpendicular + npr.normal(0, t),
                            distances + npr.normal(0, SIGMA_MEASUREMENT, size=3)))
    return Simulation(states, initial_measured_position, meas_attitude)


def simulate_test(count: int) -> Simulations:
    simulations = []
    for _ in track(range(count), "Simulating test data"):
        simulations.append(
            simulate_trajectory(STARTING_POSITION.copy(), STARTING_ATTITUDE, VELOCITY_PARALLEL, VELOCITY_PERPENDICULAR))
    return Simulations(simulations, BEACONS, STARTING_POSITION, SIGMA_POSITION, SIGMA_ATTITUDE, SIGMA_VELOCITY,
                       SIGMA_MEASUREMENT)


def simulate_train(count: int) -> Simulations:
    simulations = []
    for _ in track(range(count), "Simulating training data"):
        velocity_parallel = random.uniform(*VELOCITY_PARALLEL_NOISE)
        velocity_perpendicular = velocity_parallel * random.uniform(*VELOCITY_PERPENDICULAR_NOISE)
        att = np.deg2rad(random.uniform(*ATTITUDE_NOISE_DEG))
        pos = STARTING_POSITION + np.random.uniform(STARTING_POSITION_NOISE[0], STARTING_POSITION_NOISE[1], 2)
        simulations.append(simulate_trajectory(pos, att, velocity_parallel, velocity_perpendicular))
    return Simulations(simulations, BEACONS, STARTING_POSITION, SIGMA_POSITION, SIGMA_ATTITUDE, SIGMA_VELOCITY,
                       SIGMA_MEASUREMENT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Data generation',
        description='Simulates ship trajectories',
    )
    parser.add_argument('-x', type=float)
    parser.add_argument('-y', type=float)
    parser.add_argument('-c', '--count', type=int, default=SIMULATION_COUNT)
    args = parser.parse_args()
    fname_suffix = ''
    if args.x is not None:
        STARTING_POSITION[0] = args.x
        fname_suffix += f'_x{args.x}'
    if args.y is not None:
        STARTING_POSITION[1] = args.y
        fname_suffix += f'_y{args.y}'

    train_simulations = simulate_train(args.count)
    with open(os.path.join('simulations', f'train{fname_suffix}.pkl'), 'wb') as f:
        pickle.dump(train_simulations, f)

    test_simulations = simulate_test(args.count)
    with open(os.path.join('simulations', f'test{fname_suffix}.pkl'), 'wb') as f:
        pickle.dump(test_simulations, f)
