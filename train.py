import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader

import callbacks
from dataset import TrajectoryDataset, load_data
from nn import KGainModel

if __name__ == '__main__':
    wandb_logger = WandbLogger(project="eskf-ship-2", log_model=True)
    wandb.config.device = "cuda"
    wandb.config.gradient_clip_val = 1
    wandb.config.apply_const = False
    wandb.config.train_timesteps = 10
    wandb.config.recur_hidden_dim = 32
    wandb.config.n_recur_layers = 2
    wandb.config.recur_dropout = 0
    wandb.config.fc_dim = 32
    wandb.config.output_dim = 9
    wandb.config.fc_dropout = 0
    wandb.config.lr = 5e-4
    wandb.config.att_coef = 10
    wandb.run.log_code('.', exclude_fn=lambda f: 'venv' in f)

    tensorboard_logger = TensorBoardLogger("lightning_logs")

    torch.manual_seed(0)

    data = load_data('simulations/train.pkl')
    test_data = load_data('simulations/test.pkl')

    train_data, val_data = data.split(0.8)

    train_dataset = TrajectoryDataset(train_data)
    val_dataset = TrajectoryDataset(val_data)
    test_dataset = TrajectoryDataset(test_data)

    train_dl = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = KGainModel(9, wandb.config.recur_hidden_dim, wandb.config.n_recur_layers, wandb.config.recur_dropout,
                       wandb.config.fc_dim, wandb.config.output_dim, wandb.config.fc_dropout, wandb.config.lr,
                       wandb.config.device, wandb.config.train_timesteps, wandb.config.apply_const,
                       wandb.config.att_coef)
    model.set_beacons(data.beacon_positions)
    wandb_logger.watch(model, log="all")

    # trainer uses "gpu" instead of "cuda"
    dev = wandb.config["device"]
    dev = "gpu" if dev == "cuda" else "cpu"
    trainer = pl.Trainer(accelerator=dev, max_epochs=5000, logger=[wandb_logger, tensorboard_logger],
                         callbacks=[callbacks.LinearTimesteps(2, 1000, 100),
                                    ModelCheckpoint(monitor='val_loss', mode='min')],
                         gradient_clip_val=wandb.config["gradient_clip_val"],
                         log_every_n_steps=8)
    trainer.fit(model, train_dl, val_dl)

    # test
    test_result = trainer.test(model, test_dl)
    print(test_result)
