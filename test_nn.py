import argparse

import torch
import torch.nn.functional as F
from rich.progress import track
from torch.utils.data import DataLoader

import covariance
from dataset import load_data, TrajectoryDataset
from nn import KGainModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test neural network'
    )
    parser.add_argument('-m', '--model', type=str, help='Path to the model')
    parser.add_argument('-d', '--data', type=str, help='Path to the test data', default='simulations/test.pkl')
    args = parser.parse_args()

    test_data = load_data(args.data)
    test_dataset = TrajectoryDataset(test_data)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = KGainModel.load_from_checkpoint(args.model, map_location='cpu', device_str='cpu')
    model.eval()
    model.set_beacons(test_data.beacon_positions)

    real_errors = torch.zeros((len(test_data.simulations), len(test_data.simulations[0].states), 3))
    cov_errors = torch.zeros((len(test_data.simulations), len(test_data.simulations[0].states), 3))

    with torch.no_grad():
        for idx, batch in track(enumerate(test_dl), total=len(test_data.simulations),
                                description='Testing neural network'):
            _, _, target, _, _, _ = batch
            target = target.squeeze(0)
            losses, predict, KG = model.predict(batch)
            predicted_states = predict.squeeze(0)
            P_history = covariance.compute_covariance(predict, KG, batch, test_data)

            real_errors[idx] = F.mse_loss(predicted_states, target, reduction='none')
            cov_errors[idx] = P_history[1:].diagonal(dim1=1, dim2=2)

    torch.save(real_errors, 'errors/nn_real_errors.pt')
    torch.save(cov_errors, 'errors/nn_cov_errors.pt')
