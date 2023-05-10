import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from matplotlib import ticker as mticker

if __name__ == '__main__':
    errors = {}
    for error in ('real', 'cov'):
        errors[error] = {}
        for model in ('nn', 'kf_ext'):
            errors[error][model] = torch.load(f'errors/{model}_{error}_errors.pt')
            print(
                f'{model} {error} errors: {errors[error][model].mean((0, 1))} +- {errors[error][model].std(1).mean(0)}')

    cm = plt.get_cmap('tab10')
    fig, axs = plt.subplots(3, 2, figsize=(6, 8))
    plt.tight_layout(pad=2)
    plt.subplots_adjust(hspace=0.2)
    for k in range(2):
        axs[0, k].set_title('Ошибки координаты $x_1$')
        axs[1, k].set_title('Ошибки координаты $x_2$')
        axs[2, k].set_title('Ошибки курса')
        axs[2, k].set_xlabel('Время, с')
    axs[0, 0].set_ylabel('Ошибка, м$^2$')
    axs[1, 0].set_ylabel('Ошибка, м$^2$')
    axs[2, 0].set_ylabel('Ошибка, град$^2$')
    for i in range(3):
        for k, (model, name) in enumerate((('nn', 'Нейронная сеть'), ('kf_ext', 'Расширенный ФК'))):
            for j, error in enumerate(('real', 'cov')):
                linestyle = '-'
                lw = 1.5
                if error == 'cov':
                    linestyle = '--'
                    if model == 'nn':
                        linestyle = '-.'
                        # lw = 2
                err = errors[error][model].mean(0)[:, i] * (180 / np.pi if i == 2 else 1) ** 2
                axs[i, k].plot(err, color=cm(k),
                            # label=f'{name} - {"расчетная ошибка" if error == "cov" else "действительная ошибка"}', linestyle=linestyle, lw=lw)
                            label=f'{"Расчетная ошибка" if error == "cov" else "Действительная ошибка"}', linestyle=linestyle, lw=lw)
                axs[i, k].set_yscale('log')
            axs[i, k].legend()
            axs[i, k].grid()
    for i in range(3):
        ylim = (min(axs[i, 0].get_ylim()[0], axs[i, 1].get_ylim()[0]), max(axs[i, 0].get_ylim()[1], axs[i, 1].get_ylim()[1]))
        for k in range(2):
            axs[i, k].set_ylim(ylim)
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(5, 8))
    plt.tight_layout(pad=3)
    axs[0].set_ylabel('Ошибка, м$^2$')
    axs[1].set_ylabel('Ошибка, м$^2$')
    axs[2].set_ylabel('Ошибка, град$^2$')
    for i, name in enumerate(['координаты $x_1$', 'координаты $x_2$', 'курса']):
        axs[i].set_title(f'Ошибки {name}')
        sns.violinplot((np.log10(errors['real']['nn'].mean(1)[:, i]) - (1 if i != 2 else 0), np.log10(errors['real']['kf'].mean(1)[:, i])), ax=axs[i])
        axs[i].yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        ymin, ymax = axs[i].get_ylim()
        tick_range = np.arange(np.floor(ymin), ymax)
        axs[i].yaxis.set_ticks(tick_range)
        axs[i].yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)
        axs[i].xaxis.set_ticks([0, 1], ["Разработанный алгоритм", "Расширенный ФК"])
    plt.show()
