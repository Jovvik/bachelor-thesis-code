import argparse

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import covariance
from dataset import load_data, TrajectoryDataset
from kalman_filter import kf_predict
from nn import KGainModel
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def plot_sigma(ax, sigmas, color, label):
    ax.plot(sigmas, color=color, linestyle='--', label='Расчетная ошибка ' + label)
    ax.plot(-1 * sigmas, color=color, linestyle='--')


if __name__ == '__main__':
    cm = plt.get_cmap('tab10')
    parser = argparse.ArgumentParser(
        description='Display predicted trajectory and errors',
    )
    parser.add_argument('-m', '--model', type=str, help='Path to the model')
    parser.add_argument('-d', '--data', type=str, help='Path to the test data', default='simulations/test.pkl')
    args = parser.parse_args()

    test_data = load_data(args.data)
    test_dataset = TrajectoryDataset(test_data)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = KGainModel.load_from_checkpoint(args.model, device_str='cpu')
    model.eval()
    model.set_beacons(test_data.beacon_positions)

    # test_result = pl.Trainer(accelerator="gpu").test(model, test_dl)
    # print(f"Test result: {test_result}")

    with torch.no_grad():
        for idx, batch in enumerate(test_dl):
            print(idx)
            if idx > 3:
                break

            # NN
            _, _, target, _, _, _ = batch
            target = target.squeeze(0)
            losses, predict, KG = model.predict(batch)
            predicted_states = predict.squeeze(0)
            P_history = covariance.compute_covariance(predict, KG, batch, test_data)

            # EKF
            simulation = test_data.simulations[idx]
            ekf_states, ekf_P_history = kf_predict(False, simulation, test_data)
            ekf_ext_states, ekf_ext_P_history = kf_predict(True, simulation, test_data)

            pos_nn_loss = F.mse_loss(predicted_states[1:, :2], target[:, :2])
            att_nn_loss = F.mse_loss(predicted_states[1:, 2], target[:, 2])

            pos_ekf_loss = F.mse_loss(torch.from_numpy(ekf_states[1:, :2]), target[:, :2])
            att_ekf_loss = F.mse_loss(torch.from_numpy(ekf_states[1:, 2]), target[:, 2])
            pos_ekf_ext_loss = F.mse_loss(torch.from_numpy(ekf_ext_states[1:, :2]), target[:, :2])
            att_ekf_ext_loss = F.mse_loss(torch.from_numpy(ekf_ext_states[1:, 2]), target[:, 2])

            print(f"NN: pos {pos_nn_loss.item():.3f}, att {att_nn_loss.item():.6f}")
            print(f"EKF: pos {pos_ekf_loss.item():.3f}, att {att_ekf_loss.item():.6f}")
            print(f"EKF ext: pos {pos_ekf_ext_loss.item():.3f}, att {att_ekf_ext_loss.item():.6f}")

            fix, axs = plt.subplots()
            axs.set_title('Траектория')
            axs.plot(target[:, 0], target[:, 1], color=cm(0), label='Истинная')
            axs.plot(predicted_states[:, 0], predicted_states[:, 1], color=cm(1), label='Предсказанная нейронной сетью')
            # axs.plot(ekf_states[:, 0], ekf_states[:, 1], color=cm(2), label='Предсказанная ФК')
            axs.plot(ekf_ext_states[:, 0], ekf_ext_states[:, 1], color=cm(2), label='Предсказанная расширенным ФК')
            axs.legend()
            axs.axis('equal')
            axs.set_xlabel('x, м')
            axs.set_ylabel('y, м')
            axs.set_xlim(-400, 400)

            axins = zoomed_inset_axes(axs, 5, loc=1)  # zoom=6
            axins.plot(target[:, 0], target[:, 1], color=cm(0), label='Истинная')
            axins.plot(predicted_states[:, 0], predicted_states[:, 1], color=cm(1), label='Предсказанная нейронной сетью')
            axins.plot(ekf_ext_states[:, 0], ekf_ext_states[:, 1], color=cm(2), label='Предсказанная расширенным ФК')
            x1, x2, y1, y2 = 170, 230, 440, 500
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            # draw a bbox of the region of the inset axes in the parent axes and
            # connecting lines between the bbox and the inset axes area
            mark_inset(axs, axins, loc1=2, loc2=3, fc="none", ec="0.5")

            plt.show()

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            plt.tight_layout()

            axs[0].set_title('Ошибка по координате x')
            axs[0].plot(target[:, 0] - predicted_states[1:, 0], color=cm(1), label='Ошибка нейронной сети')
            axs[0].plot(target[:, 0] - ekf_states[1:, 0], color=cm(2), label='Ошибка ФК')
            axs[0].plot(target[:, 0] - ekf_ext_states[1:, 0], color=cm(3), label='Ошибка расширенного ФК')
            # plot_sigma(axs[0], torch.sqrt(P_history[:, 0, 0]), cm(1), 'нейронной сети')
            # plot_sigma(axs[0], torch.sqrt(torch.from_numpy(ekf_P_history[:, 0, 0])), cm(2), 'ФК')
            # plot_sigma(axs[0], torch.sqrt(torch.from_numpy(ekf_ext_P_history[:, 0, 0])), cm(3), 'расширенного ФК')
            axs[0].axhline(y=0, color='gray')
            axs[0].set_yscale('symlog')
            axs[0].legend()

            axs[1].set_title('Ошибка по координате y')
            axs[1].plot(target[:, 1] - predicted_states[1:, 1], color=cm(1), label='Ошибка нейронной сети')
            axs[1].plot(target[:, 1] - ekf_states[1:, 1], color=cm(2), label='Ошибка ФК')
            axs[1].plot(target[:, 1] - ekf_ext_states[1:, 1], color=cm(3), label='Ошибка расширенного ФК')
            # plot_sigma(axs[1], torch.sqrt(P_history[:, 1, 1]), cm(1), 'нейронной сети')
            # plot_sigma(axs[1], torch.sqrt(torch.from_numpy(ekf_P_history[:, 1, 1])), cm(2), 'ФК')
            # plot_sigma(axs[1], torch.sqrt(torch.from_numpy(ekf_ext_P_history[:, 1, 1])), cm(3), 'расширенного ФК')
            axs[1].axhline(y=0, color='gray')
            axs[1].set_yscale('symlog')
            axs[1].legend()

            axs[2].set_title('Ошибка по углу')
            axs[2].plot(target[:, 2] - predicted_states[1:, 2], color=cm(1), label='Ошибка нейронной сети')
            axs[2].plot(target[:, 2] - ekf_states[1:, 2], color=cm(2), label='Ошибка ФК')
            axs[2].plot(target[:, 2] - ekf_ext_states[1:, 2], color=cm(3), label='Ошибка расширенного ФК')
            # plot_sigma(axs[2], torch.sqrt(P_history[:, 2, 2]), cm(1), 'нейронной сети')
            # plot_sigma(axs[2], torch.sqrt(torch.from_numpy(ekf_P_history[:, 2, 2])), cm(2), 'ФК')
            # plot_sigma(axs[2], torch.sqrt(torch.from_numpy(ekf_ext_P_history[:, 2, 2])), cm(3), 'расширенного ФК')
            axs[2].axhline(y=0, color='gray')
            axs[2].legend()

            plt.show()
