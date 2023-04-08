import argparse

import torch
import torch.nn.functional as F
from rich.progress import track

from dataset import load_data
from kalman_filter import kf_predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test neural network'
    )
    parser.add_argument('-d', '--data', type=str, help='Path to the test data', default='simulations/test.pkl')
    args = parser.parse_args()

    test_data = load_data(args.data)

    real_errors = torch.zeros((len(test_data.simulations), len(test_data.simulations[0].states), 3))
    cov_errors = torch.zeros((len(test_data.simulations), len(test_data.simulations[0].states), 3))
    real_ext_errors = torch.zeros((len(test_data.simulations), len(test_data.simulations[0].states), 3))
    cov_ext_errors = torch.zeros((len(test_data.simulations), len(test_data.simulations[0].states), 3))

    with torch.no_grad():
        for idx, simulation in track(enumerate(test_data.simulations), description='Testing Kalman filter'):
            ekf_states, ekf_P_history = kf_predict(False, simulation, test_data)
            ekf_ext_states, ekf_ext_P_history = kf_predict(True, simulation, test_data)

            states = torch.tensor([(state.real_position[0], state.real_position[1], state.real_attitude) for state in simulation.states])
            real_errors[idx] = F.mse_loss(torch.from_numpy(ekf_states[1:]), states, reduction='none')
            cov_errors[idx] = torch.from_numpy(ekf_P_history[1:]).diagonal(dim1=1, dim2=2)
            real_ext_errors[idx] = F.mse_loss(torch.from_numpy(ekf_ext_states[1:]), states, reduction='none')
            cov_ext_errors[idx] = torch.from_numpy(ekf_ext_P_history[1:]).diagonal(dim1=1, dim2=2)

    torch.save(real_errors, 'errors/kf_real_errors.pt')
    torch.save(cov_errors, 'errors/kf_cov_errors.pt')
    torch.save(real_ext_errors, 'errors/kf_ext_real_errors.pt')
    torch.save(cov_ext_errors, 'errors/kf_ext_cov_errors.pt')
