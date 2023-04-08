import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == '__main__':
    errors = {}
    for error in ('real', 'cov'):
        errors[error] = {}
        for model in ('nn', 'kf', 'kf_ext'):
            errors[error][model] = torch.load(f'errors/{model}_{error}_errors.pt')
            print(
                f'{model} {error} errors: {errors[error][model].mean((0, 1))} +- {errors[error][model].std(1).mean(0)}')

    cm = plt.get_cmap('tab10')
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    axs[0].set_title('Ошибки координаты x')
    axs[0].set_xlabel('Время, с')
    axs[0].set_ylabel('Ошибка, м$^2$')
    axs[1].set_title('Ошибки координаты y')
    axs[1].set_xlabel('Время, с')
    axs[1].set_ylabel('Ошибка, м$^2$')
    axs[2].set_title('Ошибки курса')
    axs[2].set_xlabel('Время, с')
    axs[2].set_ylabel('Ошибка, град$^2$')
    for j, error in enumerate(('real', 'cov')):
        for i in range(3):
            for k, (model, name) in enumerate((('nn', 'Нейронная сеть'), ('kf', 'Расширенный ФК'))):
                linestyle = '-'
                lw = 1.5
                if error == 'cov':
                    linestyle = '--'
                    if model == 'nn':
                        linestyle = '-.'
                        # lw = 2
                axs[i].plot(errors[error][model].mean(0)[:, i] * (180 / np.pi if i == 2 else 1), color=cm(k),
                            label=f'{name} - {"расчетная ошибка" if error == "cov" else "действительная ошибка"}', linestyle=linestyle, lw=lw)
                axs[i].set_yscale('log')
            axs[i].legend()
            axs[i].grid()
    plt.show()
