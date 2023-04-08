from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class State:
    real_position: np.ndarray
    real_attitude: float
    # next 4 are measured
    measured_attitude: float
    velocity_parallel: float
    velocity_perpendicular: float
    distances: np.ndarray


@dataclass
class Simulation:
    states: List[State]
    initial_position: np.ndarray
    initial_attitude: float


@dataclass
class Simulations:
    simulations: List[Simulation]
    beacon_positions: np.ndarray
    start_position: np.ndarray
    sigma_position: float
    sigma_attitude: float
    sigma_velocity: float
    sigma_measurement: float

    def split(self, ratio):
        n = int(len(self.simulations) * ratio)
        return Simulations(self.simulations[:n], self.beacon_positions, self.start_position, self.sigma_position,
                           self.sigma_attitude, self.sigma_velocity, self.sigma_measurement), Simulations(
            self.simulations[n:], self.beacon_positions, self.start_position, self.sigma_position, self.sigma_attitude,
            self.sigma_velocity, self.sigma_measurement)
