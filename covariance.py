import pytorch_lightning as pl
import torch

from nn import Batch
from simulation import Simulations


def compute_covariance(predict, KG, batch, simulations: Simulations) -> torch.Tensor:
    assert batch[2].shape[0] == 1
    _, _, target, vel, att, dist = batch

    P_history = torch.zeros((KG.shape[1] + 1, 3, 3))
    P_history[0] = torch.diag(torch.tensor(
        [simulations.sigma_position ** 2, simulations.sigma_position ** 2, simulations.sigma_attitude ** 2]))
    Q = torch.diag(torch.tensor([simulations.sigma_velocity ** 2, simulations.sigma_velocity ** 2, 0]))
    for t in range(KG.shape[1]):
        v1, v2 = vel[0][t]
        x = predict[0, t, 0]
        y = predict[0, t, 1]
        ang = att[0, t] + predict[0, t, 2]
        # ang = predict[0, t, 2]
        sin = torch.sin(ang)
        cos = torch.cos(ang)
        F = torch.tensor([[1, 0, v2 * sin - v1 * cos],
                          [0, 1, v1 * sin + v2 * cos],
                          [0, 0, 1]])
        u = torch.tensor([v1 * sin + v2 * cos, v1 * cos - v2 * sin, 0])
        Phat = F @ P_history[t] @ F.t() + Q # * 100
        d1 = torch.linalg.vector_norm(torch.tensor([x, y]) - simulations.beacon_positions[0])
        d2 = torch.linalg.vector_norm(torch.tensor([x, y]) - simulations.beacon_positions[1])
        d3 = torch.linalg.vector_norm(torch.tensor([x, y]) - simulations.beacon_positions[2])
        H = torch.tensor(
            [[(x - simulations.beacon_positions[0][0]) / d1, (y - simulations.beacon_positions[0][1]) / d1, 0],
             [(x - simulations.beacon_positions[1][0]) / d2, (y - simulations.beacon_positions[1][1]) / d2, 0],
             [(x - simulations.beacon_positions[2][0]) / d3, (y - simulations.beacon_positions[2][1]) / d3, 0]]
            , dtype=torch.float32
        )
        R = torch.diag(torch.tensor([simulations.sigma_measurement ** 2, simulations.sigma_measurement ** 2,
                                     simulations.sigma_measurement ** 2], dtype=torch.float32))
        S = H @ Phat @ H.t() + R  # * 100
        K = Phat @ H.t() @ torch.linalg.inv(S)
        P = Phat - K @ S @ K.t()
        P_history[t + 1] = P
    return P_history
