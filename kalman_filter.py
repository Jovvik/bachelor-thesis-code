from typing import Callable, Tuple

import numpy as np
import torch

from simulation import Simulations, Simulation

Fun = Callable[..., np.ndarray]


class KalmanFilter:
    def __init__(self, is_extended: bool, initial_state: np.ndarray, P: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 dt: float, beacon_positions: np.ndarray):
        self.is_extended = is_extended
        self.state = initial_state
        self.P = P
        self.Q = Q
        self.R = R
        self.dt = dt
        self.beacon_positions = beacon_positions

    def step(self, distances: np.ndarray, velocity: np.ndarray, attitude: float, state: np.ndarray) -> np.ndarray:
        v1 = velocity[0]
        v2 = velocity[1]
        F = np.array([
            [1, 0, (-v1 * np.cos(attitude) + v2 * np.sin(attitude)) * self.dt],
            [0, 1, (v1 * np.sin(attitude) + v2 * np.cos(attitude)) * self.dt],
            [0, 0, 1]
        ])
        extrapolated_state = F @ self.state
        P_extrapolated = F @ self.P @ F.T + self.Q
        extrapolated_measurements = np.array(
            [np.linalg.norm(state[:2] - self.state[:2] - beacon_position) for beacon_position in self.beacon_positions])
        linpoint = state - self.state if not self.is_extended else state - extrapolated_state
        H = np.array(
            [[-(linpoint[0] - beacon_position[0]) / extrapolated_measurements[i],
              -(linpoint[1] - beacon_position[1]) / extrapolated_measurements[i],
              0]
             for i, beacon_position in enumerate(self.beacon_positions)]
        )
        S = H @ P_extrapolated @ H.T + self.R
        K = P_extrapolated @ H.T @ np.linalg.inv(S)
        self.state = extrapolated_state + K @ (distances - extrapolated_measurements)
        self.P = (np.eye(3) - K @ H) @ P_extrapolated
        return self.state


class ErrorStateKalmanFilter:
    def __init__(self, is_extended: bool, initial_state: np.ndarray, P: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 dt: float, beacon_positions: np.ndarray):
        self.eskf = KalmanFilter(is_extended, np.zeros_like(initial_state), P, Q, R, dt, beacon_positions)
        self.is_extended = is_extended
        self.state = initial_state
        self.dt = dt

    def step(self, distances: np.ndarray, velocity: np.ndarray, attitude: float) -> np.ndarray:
        v1 = velocity[0]
        v2 = velocity[1]
        extrapolated_state = np.array([
            self.state[0] + v1 * np.sin(attitude) * self.dt + v2 * np.cos(attitude) * self.dt,
            self.state[1] + v1 * np.cos(attitude) * self.dt - v2 * np.sin(attitude) * self.dt,
            self.state[2]
        ])
        error_state = self.eskf.step(distances, velocity, attitude,
                                     extrapolated_state if self.is_extended else self.state)
        self.state = extrapolated_state - error_state
        return self.state


class FullKalmanFilter:
    def __init__(self, is_extended: bool, initial_state: np.ndarray, P: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 dt: float, beacon_positions: np.ndarray):
        self.is_extended = is_extended
        self.state = initial_state
        self.P = P
        self.Q = Q
        self.R = R
        self.dt = dt
        self.beacon_positions = beacon_positions

    def step(self, distances: np.ndarray, velocity: np.ndarray, attitude: float) -> np.ndarray:
        v1 = velocity[0]
        v2 = velocity[1]
        F = np.array([
            [1, 0, (-v1 * np.cos(attitude) + v2 * np.sin(attitude)) * self.dt],
            [0, 1, (v1 * np.sin(attitude) + v2 * np.cos(attitude)) * self.dt],
            [0, 0, 1]
        ])
        corrected_attitude = attitude - self.state[2]
        corr_sin = np.sin(corrected_attitude)
        corr_cos = np.cos(corrected_attitude)
        extrapolated_state = np.array([
            self.state[0] + v1 * corr_sin * self.dt + v2 * corr_cos * self.dt,
            self.state[1] + v1 * corr_cos * self.dt - v2 * corr_sin * self.dt,
            self.state[2]
        ])
        P_extrapolated = F @ self.P @ F.T + self.Q
        extrapolated_measurements = np.array(
            [np.linalg.norm(self.state[:2] - beacon_position) for beacon_position in self.beacon_positions])
        linpoint = self.state if not self.is_extended else extrapolated_state
        H = np.array(
            [[(linpoint[0] - beacon_position[0]) / extrapolated_measurements[i],
              (linpoint[1] - beacon_position[1]) / extrapolated_measurements[i],
              0]
             for i, beacon_position in enumerate(self.beacon_positions)]
        )
        S = H @ P_extrapolated @ H.T + self.R
        K = P_extrapolated @ H.T @ np.linalg.inv(S)
        self.state = extrapolated_state + K @ (distances - extrapolated_measurements)
        self.P = (np.eye(3) - K @ H) @ P_extrapolated
        return np.array([self.state[0], self.state[1], attitude - self.state[2]])


def make_ekf(is_extended: bool, simulation: Simulation, simulations: Simulations) -> ErrorStateKalmanFilter:
    return ErrorStateKalmanFilter(
        is_extended,
        np.array([simulation.initial_position[0], simulation.initial_position[1], simulation.initial_attitude]),
        np.diag([simulations.sigma_position ** 2, simulations.sigma_position ** 2, simulations.sigma_attitude ** 2]),
        np.diag([simulations.sigma_velocity ** 2, simulations.sigma_velocity ** 2, 0]),
        np.diag([simulations.sigma_measurement ** 2, simulations.sigma_measurement ** 2,
                 simulations.sigma_measurement ** 2]),
        1,
        simulations.beacon_positions
    )


def make_full_kf(is_extended: bool, simulation: Simulation, simulations: Simulations) -> FullKalmanFilter:
    return FullKalmanFilter(
        is_extended,
        np.array([simulation.initial_position[0], simulation.initial_position[1], simulation.initial_attitude]),
        np.diag([simulations.sigma_position ** 2, simulations.sigma_position ** 2, simulations.sigma_attitude ** 2]),
        np.diag([simulations.sigma_velocity ** 2, simulations.sigma_velocity ** 2, 0]),
        np.diag([simulations.sigma_measurement ** 2, simulations.sigma_measurement ** 2,
                 simulations.sigma_measurement ** 2]),
        1,
        simulations.beacon_positions
    )


def kf_predict(is_extended: bool, simulation: Simulation, simulations: Simulations) -> Tuple[np.ndarray, np.ndarray]:
    kf = make_full_kf(is_extended, simulation, simulations)
    states = np.zeros((len(simulation.states) + 1, 3))  # +1 to store the initial state
    P_history = np.zeros((len(simulation.states) + 1, 3, 3))
    states[0] = np.array([simulation.initial_position[0], simulation.initial_position[1], simulation.initial_attitude])
    P_history[0] = kf.P
    for i, state in enumerate(simulation.states):
        states[i + 1] = kf.step(state.distances, np.array([state.velocity_parallel, state.velocity_perpendicular]),
                                state.measured_attitude)
        P_history[i + 1] = kf.P
    return states, P_history
