from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor

Batch = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


class KGainModel(pl.LightningModule):
    def __init__(self, input_dim, recur_hidden_dim, n_recur_layers, recur_dropout_rate, fc_dim, output_dim,
                 fc_dropout_rate, lr, device_str, train_timesteps, apply_const, att_coeff=1):
        super().__init__()
        self.beacons = None
        self.save_hyperparameters()
        self.state = None
        self.lr = lr
        self.att_coeff = att_coeff
        self.device_str = device_str
        self.train_timesteps = train_timesteps
        self.apply_const = apply_const
        self.lstm = nn.LSTM(input_dim, recur_hidden_dim, n_recur_layers, batch_first=True, dropout=recur_dropout_rate)
        self.lstm_state = None
        self.tail = nn.Sequential(
            # nn.BatchNorm1d(recur_hidden_dim),
            nn.Linear(recur_hidden_dim, fc_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(fc_dim),
            nn.Dropout(fc_dropout_rate),
            nn.Linear(fc_dim, output_dim),  # note: copilot wants to add more linear layers
        )
        self.old_vel = None
        self.old_att = None
        self.old_dist = None

    def set_beacons(self, beacons: np.ndarray):
        self.beacons = torch.from_numpy(beacons).float().to(self.device_str)

    def loss_fn(self, y, stage: str):
        pos_loss = torch.sqrt(F.mse_loss(self.state[:, :2], y[:, :2]))
        att_loss = torch.mean(torch.abs(torch.sin(self.state[:, 2] - y[:, 2])))
        if stage != "predict":
            self.log(f"{stage}_pos_loss", pos_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_att_loss", att_loss, on_step=False, on_epoch=True, prog_bar=True)
            if stage == "val":
                self.log(f"{stage}_loss", pos_loss + att_loss * self.att_coeff, on_step=False, on_epoch=True,
                         prog_bar=True)
        # return torch.sqrt(pos_loss * (0.5 + att_loss))
        # return torch.exp((torch.log(pos_loss) + torch.log(att_loss) * self.att_coeff) / (1 + self.att_coeff))
        return pos_loss + att_loss * self.att_coeff

    def get_kalman_gain(self, vel, att, dist):
        lstm_in = torch.cat([vel, att, dist, self.state], dim=1)
        lstm_out, self.lstm_state = self.lstm(lstm_in.unsqueeze(1), self.lstm_state)
        kalman_gain = self.tail(lstm_out.squeeze(1)).reshape(-1, 3, 3)
        self.KG = kalman_gain / 1000

    def reset_lstm_state(self, batch_size: int):
        self.lstm_state = (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device),
                           torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device))

    def apply_kalman_gain(self, vel, att, dist):
        if self.apply_const:
            sin = torch.sin(self.state[:, 2])
            cos = torch.cos(self.state[:, 2])
            v1 = vel[:, 0]
            v2 = vel[:, 1]
            zeros = torch.zeros_like(v1)
            u = torch.stack([torch.mul(v1, sin) + torch.mul(v2, cos), torch.mul(v1, cos) - torch.mul(v2, sin), zeros],
                            dim=1)
            self.state = self.state + u
        self.get_kalman_gain(vel, att, dist)
        dist_pred = torch.stack([torch.norm(self.state[:, :2] - self.beacons[i], dim=1) for i in range(3)], dim=1)
        dist_diff = dist - dist_pred
        self.state = self.state + torch.bmm(self.KG, dist_diff.unsqueeze(2)).reshape(-1, 3)

    def plot_trajectory(self, ref, state_history, batch_idx):
        if batch_idx == 0:
            fig, ax = plt.subplots()
            ax.plot(ref[0, :, 0].cpu().numpy(), ref[0, :, 1].cpu().numpy(), "k")
            ax.plot(state_history[:, 0].cpu().numpy(), state_history[:, 1].cpu().numpy(), "r")
            self.logger.experiment.log({"val_path": fig})

    def set_initial_state(self, batch: Batch):
        i_pos, i_att, _, vel, att, dist = batch
        self.old_att = att[:, 0]
        self.old_vel = vel[:, 0]
        self.old_dist = dist[:, 0]
        self.state = torch.cat([i_pos, i_att.unsqueeze(1)], dim=1)

    def process_batch(self, batch: Batch, stage: str) -> Tuple[Tensor, Tensor, Tensor]:
        self.set_initial_state(batch)
        loss = 0
        _, _, target, vel, att, dist = batch
        self.reset_lstm_state(target.shape[0])
        timesteps = target.shape[1]
        if stage == "train":
            timesteps = min(self.train_timesteps, timesteps)
        state_history = torch.zeros((target.shape[0], timesteps + 1, 3)).to(self.device)
        state_history[:, 0] = self.state
        KG_history = torch.zeros((target.shape[0], timesteps, 3, 3)).to(self.device)
        for time_step in range(0, timesteps):
            self.apply_kalman_gain(vel[:, time_step, :], att[:, time_step, :], dist[:, time_step, :])
            state_history[:, time_step + 1] = self.state
            KG_history[:, time_step] = self.KG
            loss = loss + self.loss_fn(target[:, time_step, :], stage)
        loss = loss / timesteps
        return loss, state_history, KG_history

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        loss, *_ = self.process_batch(batch, "train")
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:
        loss, *_ = self.process_batch(batch, "val")
        if batch_idx == 0:
            self.logger.experiment.log({"train_timesteps": self.train_timesteps})
        return loss

    def test_step(self, batch, batch_idx):
        loss, *_ = self.process_batch(batch, "test")
        return loss

    def predict(self, batch: Batch) -> Tuple[Tensor, Tensor, Tensor]:
        return self.process_batch(batch, "predict")

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
        return {
            "optimizer": opt,
            "lr_scheduler": sched,
            "monitor": "val_loss"
        }
