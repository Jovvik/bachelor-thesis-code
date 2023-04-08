import math

import pytorch_lightning as pl


class ConstantTimesteps(pl.Callback):
    def __init__(self, timesteps):
        self.timesteps = timesteps

    def on_train_start(self, trainer, pl_module):
        pl_module.train_timesteps = self.timesteps


# linearly interpolates between start & end
class LinearTimesteps(pl.Callback):
    def __init__(self, start, end, end_epoch):
        self.start = start
        self.end = end
        self.end_epoch = end_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        ratio = min(trainer.current_epoch / self.end_epoch, 1.0)
        pl_module.train_timesteps = int(self.start + (self.end - self.start) * ratio)


class CutOffTimesteps(pl.Callback):
    def __init__(self, timesteps, stop_epoch):
        self.timesteps = timesteps
        self.stop_epoch = stop_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch > self.stop_epoch:
            pl_module.train_timesteps = math.inf
        else:
            pl_module.train_timesteps = self.timesteps
