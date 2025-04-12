import numpy as np
import argparse
import csv
from neural_net import NeuralNetwork
from train import load_cifar10, one_hot

def grid_search(params):
    # 读取数据
    x_train, y_train, x_test, y_test = load_cifar10(params['data_dir'])
    y_train_oh = one_hot(y_train)

    # 拆分出验证集
    split = int(0.9 * len(x_train))
    x_val, y_val = x_train[split:], y_train_oh[split:]
    x_train, y_train = x_train[:split], y_train_oh[:split]

    best_acc = 0
    best_params = None

    # 保存训练记录的 CSV 文件
    with open('hyperparameter_search_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Learning Rate', 'Regularization Lambda', 'Hidden Size', 'Validation Accuracy', 'Test Accuracy'])

        # 进行网格搜索
        for lr in params['lr']:
            for reg_lambda in params['reg_lambda']:
                for hidden_size in params['hidden_size']:
                    print(f"\n Training with lr={lr}, reg_lambda={reg_lambda}, hidden_size={hidden_size}...")
                    model = NeuralNetwork(
                        hidden_sizes=[3072, hidden_size, 10],
                        activations=['relu', 'softmax'],
                        learning_rate=lr,
                        reg_lambda=reg_lambda
                    )

                    # 训练模型
                    history = model.fit(
                        x=x_train,
                        y=y_train,
                        epochs=params['epochs'],
                        batch_size=params['batch_size'],
                        x_val=x_val,
                        y_val=y_val,
                        decay_every=params['decay_every'],
                        decay_factor=params['decay_factor']
                    )

                    # 评估模型
                    y_pred = model.predict(x_test)
                    test_acc = np.mean(y_pred == y_test)
                    val_acc = history['val_acc'][-1] if 'val_acc' in history else 0  # 获取最后一个验证集准确度

                    # 写入每次超参数组合和相应的结果
                    writer.writerow([lr, reg_lambda, hidden_size, val_acc, test_acc])
                    print(f"Test Accuracy: {test_acc * 100:.2f}%, Val Accuracy: {val_acc * 100:.2f}%")

                    # 更新最佳准确率
                    if test_acc > best_acc:
                        best_acc = test_acc
                        best_params = (lr, reg_lambda, hidden_size)
                        print(f"Best model updated with test_acc: {test_acc * 100:.2f}%")

    print(f"\n Best Hyperparameters: lr={best_params[0]}, reg_lambda={best_params[1]}, hidden_size={best_params[2]}")
    print(f"Best Test Accuracy: {best_acc * 100:.2f}%")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='cifar-10-batches-py')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--decay_every', type=int, default=5)
    parser.add_argument('--decay_factor', type=float, default=0.9)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # 定义参数网格
    param_grid = {
        'lr': [0.001, 0.01, 0.1],  # 学习率
        'reg_lambda': [1e-4, 1e-3, 1e-2],  # 正则化强度
        'hidden_size': [128, 256, 512],  # 隐藏层大小
        'epochs': 10,  # 训练周期
        'batch_size': 64,  # 批次大小
        'decay_every': 5,  # 学习率衰减间隔
        'decay_factor': 0.9,  # 学习率衰减因子
        'data_dir': 'cifar-10-batches-py',  # 数据集目录
    }

    grid_search(param_grid)