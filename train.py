import pickle
import numpy as np
import argparse
import os
from neural_net import NeuralNetwork
from plot import plotter

def load_batch(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        x = data[b'data'].astype(np.float32) / 255.0
        y = np.array(data[b'labels'])
        return x, y

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

def load_cifar10(data_dir='cifar-10-batches-py'):
    x_train, y_train = [], []
    for i in range(1, 6):
        x, y = load_batch(os.path.join(data_dir, f'data_batch_{i}'))
        x_train.append(x)
        y_train.append(y)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test, y_test = load_batch(os.path.join(data_dir, 'test_batch'))
    return x_train, y_train, x_test, y_test

def train(args):
    np.random.seed(42)

    # 1. Load data
    x_train, y_train, x_test, y_test = load_cifar10(args.data_dir)
    y_train_oh = one_hot(y_train)

    # 2. Split val set from training data
    split = int(0.9 * len(x_train))
    x_val, y_val = x_train[split:], y_train_oh[split:]
    x_train, y_train = x_train[:split], y_train_oh[:split]

    # 3. Define model
    model = NeuralNetwork(
        hidden_sizes = [3072, args.hidden_size, 10],
        activations = ['relu', 'softmax'],
        learning_rate = args.lr,
        reg_lambda = args.reg
    )

    # 4. Train model
    history = model.fit(
        x = x_train,
        y = y_train,
        epochs = args.epochs,
        batch_size = args.batch_size,
        x_val = x_val,
        y_val = y_val,
        decay_every = args.decay_every,
        decay_factor = args.decay_factor,
        save_path = "best_model.txt"
    )

    # 5. Evaluate best saved model on test set
    best_model = NeuralNetwork.load("best_model.txt")
    y_pred = best_model.predict(x_test)
    acc = np.mean(y_pred == y_test)
    print(f"\n Final Test Accuracy of best model: {acc * 100:.2f}%")

    plotter(history, savepath="training_plot.png")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='cifar-10-batches-py')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--reg', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--decay_every', type=int, default=5)
    parser.add_argument('--decay_factor', type=float, default=0.9)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)