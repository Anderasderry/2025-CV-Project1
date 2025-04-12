import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def plotter(history, trunc=0, savepath=None):
    val_acc = history.get('val_acc', [])
    val_loss = history.get('val_loss', [])
    train_loss = history.get('loss', [])

    if val_acc:
        print(f"Final Validation Accuracy = {val_acc[-1] * 100:.2f}%")

    def sample_curve(data, num_points=10):
        if len(data) <= num_points:
            return list(range(len(data))), data
        idx = np.linspace(0, len(data) - 1, num_points).astype(int)
        return idx, [data[i] for i in idx]

    plt.figure(figsize=(15, 7))

    # ======= LOSS PLOT =======
    plt.subplot(1, 2, 1)
    plt.title('Loss')

    idx_train, sampled_train = sample_curve(train_loss[trunc:])
    idx_val, sampled_val = sample_curve(val_loss)

    plt.plot(idx_train, sampled_train, '-o', label='Training Loss', linewidth=1)
    plt.plot(idx_val, sampled_val, '-ro', label='Validation Loss', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ======= ACCURACY PLOT =======
    plt.subplot(1, 2, 2)
    plt.title('Validation Accuracy')

    idx_acc, sampled_acc = sample_curve(val_acc)
    plt.plot(idx_acc, sampled_acc, '-o', label='Val Accuracy')

    if val_acc:
        final_acc = val_acc[-1]
        plt.plot(
            np.linspace(-2, idx_acc[-1] + 2, 500),
            np.full(500, final_acc),
            '--',
            color='gray',
            label='Final Val Acc'
        )

        # 设置 y 轴刻度覆盖 final acc，并保留两位小数
        ymin = min(sampled_acc)
        ymax = max(max(sampled_acc), final_acc)
        yticks = np.round(np.linspace(ymin, ymax, 10), 2)
        plt.yticks(yticks)

    plt.xlim(-.5, idx_acc[-1] + 0.5)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))  # 两位小数格式
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if savepath:
        plt.savefig(savepath)
    plt.show()
