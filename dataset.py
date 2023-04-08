import pickle
from typing import Tuple

import numpy as np
import torch
import wandb as wandb
from torch import Tensor
from torch.utils.data import Dataset

from simulation import Simulations


def load_data(filename: str) -> Simulations:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        print(f"Loaded {len(data.simulations)} simulations from {filename} with:")
        print(f"    Beacon positions: {data.beacon_positions}")
        print(f"    Starting position: {data.start_position}")
        print(f"    Sigma of initial position: {data.sigma_position}")
        print(f"    Sigma of attitude: {data.sigma_attitude}")
        print(f"    Sigma of velocity: {data.sigma_velocity}")
        print(f"    Sigma of distance measurements: {data.sigma_measurement}")
    return data


class TrajectoryDataset(Dataset):
    def __init__(self, simulations: Simulations):
        data = simulations.simulations
        try:
            dev = wandb.config.device
        except wandb.Error:
            dev = "cpu"
        dtype = torch.float32
        self.initial_positions = torch.tensor(np.array([s.initial_position for s in data])).to(dev).to(dtype)
        self.initial_attitudes = torch.tensor([s.initial_attitude for s in data]).to(dev).to(dtype)
        self.target = torch.tensor(
            [[[s.real_position[0], s.real_position[1], s.real_attitude] for s in t.states] for t in data]).to(dev).to(
            dtype)
        self.vel = torch.tensor(
            [[[s.velocity_parallel, s.velocity_perpendicular] for s in t.states] for t in data]).to(dev).to(dtype)
        self.att = torch.tensor([[[s.measured_attitude] for s in t.states] for t in data]).to(dev).to(dtype)
        self.dist = torch.tensor(
            [[[s.distances[0], s.distances[1], s.distances[2]] for s in t.states] for t in data]).to(dev).to(dtype)
        self.beacon_positions = simulations.beacon_positions
        self.sigma_position = simulations.sigma_position
        self.start_position = simulations.start_position

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.initial_positions[idx], self.initial_attitudes[idx], self.target[idx], self.vel[idx], self.att[idx], \
            self.dist[idx]

    def __len__(self):
        return self.initial_positions.shape[0]
