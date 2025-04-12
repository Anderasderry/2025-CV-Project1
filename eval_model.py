import numpy as np
from neural_net import NeuralNetwork
from train import load_cifar10
import argparse

def evaluate(model_path, data_dir='cifar-10-batches-py'):
    _, _, x_test, y_test = load_cifar10(data_dir)

    model = NeuralNetwork.load(model_path)
    print(f"Model loaded from: {model_path}")

    y_pred = model.predict(x_test)
    acc = np.mean(y_pred == y_test)

    print(f"Test Accuracy: {acc * 100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.txt', help='Path to saved model')
    parser.add_argument('--data_dir', type=str, default='cifar-10-batches-py', help='CIFAR-10 data directory')
    args = parser.parse_args()

    evaluate(args.model_path, args.data_dir)