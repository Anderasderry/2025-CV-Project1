"""
Author: Xiaorui Jin 23307110107
Homepage: https://github.com/Anderasderry/2025-CV-Project1

"""

import numpy as np
from tqdm import tqdm

class Activator:
    @classmethod
    def activate(cls, name='relu'):
        if name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-np.clip(x, -50, 50)))
        elif name == 'tanh':
            return lambda x: np.tanh(x)
        elif name == 'softmax':
            return lambda x: np.exp(x - np.max(x, axis=1, keepdims=True)) / \
                             np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True)

    @classmethod
    def derivative(cls, name='relu'):
        if name == 'relu':
            return lambda x: (x > 0).astype(float)
        elif name == 'sigmoid':
            sig = cls.activate('sigmoid')
            return lambda x: sig(x) * (1 - sig(x))
        elif name == 'tanh':
            return lambda x: 1 - np.tanh(x) ** 2
        else:
            raise ValueError(f"No derivative for activation: {name}")

class NeuralNetwork:
    def __init__(self, hidden_sizes, activations, learning_rate=0.01, reg_lambda=0.0):
        self.layers = len(hidden_sizes) - 1
        self.learning_rate = learning_rate
        self.activations = activations
        self.weights = []
        self.biases = []
        self.reg_lambda = reg_lambda
        self.hidden_size = hidden_sizes
        self.activators = activations

        for i in range(self.layers):
            in_dim, out_dim = hidden_sizes[i], hidden_sizes[i + 1]
            # He initialization
            self.weights.append(np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim))
            self.biases.append(np.zeros((1, out_dim)))

        self.outputs = [None] * (self.layers + 1)
        self.z_values = [None] * (self.layers)

    def forward(self, x):
        self.outputs[0] = x
        for i in range(self.layers):
            z = np.dot(self.outputs[i], self.weights[i]) + self.biases[i]
            self.z_values[i] = z
            self.outputs[i + 1] = Activator.activate(self.activations[i])(z)
        return self.outputs[-1]

    def backward(self, y_true):
        grads_w, grads_b = [None] * self.layers, [None] * self.layers
        delta = self.outputs[-1] - y_true  # softmax + cross entropy

        for i in reversed(range(self.layers)):
            grads_w[i] = np.dot(self.outputs[i].T, delta) / y_true.shape[0]
            grads_b[i] = np.mean(delta, axis=0, keepdims=True)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * \
                        Activator.derivative(self.activations[i - 1])(self.z_values[i - 1])
        return grads_w, grads_b

    def update_params(self, grads_w, grads_b):
        for i in range(self.layers):
            self.weights[i] -= self.learning_rate * (grads_w[i] + self.reg_lambda * self.weights[i])
            self.biases[i] -= self.learning_rate * grads_b[i]

    def compute_loss(self, pred, target):
        m = target.shape[0]
        log_likelihood = -np.log(pred + 1e-9) * target
        loss = np.sum(log_likelihood) / m
        reg_loss = 0.5 * self.reg_lambda * sum(np.sum(w**2) for w in self.weights)
        return loss + reg_loss

    def fit(self, x, y, epochs=10, batch_size=64, x_val=None, y_val=None,
            decay_every=None, decay_factor=0.5, save_path=None):
        n = x.shape[0]
        history = {'loss': [], 'val_loss': [], 'val_acc': []}
        best_acc = 0.0

        for epoch in range(epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)

            with tqdm(range(0, n, batch_size), desc = f"Epoch {epoch + 1}") as pbar:
                for i in pbar:
                    batch_idx = indices[i:i + batch_size]
                    x_batch, y_batch = x[batch_idx], y[batch_idx]

                    self.forward(x_batch)
                    grads_w, grads_b = self.backward(y_batch)
                    self.update_params(grads_w, grads_b)

            # Epoch completed, compute training loss
            y_pred = self.forward(x)
            loss = self.compute_loss(y_pred, y)
            history['loss'].append(loss)

            # Validation check
            if x_val is not None and y_val is not None:
                y_val_pred = self.forward(x_val)
                val_loss = self.compute_loss(y_val_pred, y_val)
                val_acc = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                print(f"Validation Accuracy: {val_acc * 100:.2f}%, Loss: {val_loss:.4f}")

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch + 1
                    if save_path:
                        self.save(save_path)
                    print(f"Best model saved at epoch {best_epoch} with val_acc: {val_acc * 100:.2f}%")

            # Decay learning rate
            if decay_every and (epoch + 1) % decay_every == 0:
                self.learning_rate *= decay_factor
                print(f"Learning rate decayed to {self.learning_rate:.6f}")

        return history

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)


    @classmethod
    def load(cls, path):
        def arraying(x, f):
            for i in range(x.shape[0]):
                x[i] = np.array([float(i) for i in f.readline().strip().split()])

        with open(path, 'r') as f:
            hidden_size = [int(i) for i in f.readline().strip().split()]
            acts = f.readline().strip().split()
            mod = NeuralNetwork(hidden_size, acts)
            for i in range(len(hidden_size) - 1):
                arraying(mod.weights[i], f)
                arraying(mod.biases[i], f)
        return mod

    def save(self, path):
        def saving(x, f):
            for line in x:
                f.write('\n' + ' '.join(['%.6f' % value for value in line]))

        with open(path, 'w') as f:
            f.write(' '.join([str(i) for i in self.hidden_size]))
            f.write('\n' + ' '.join([str(i) for i in self.activators]))
            for i in range(len(self.weights)):
                saving(self.weights[i], f)
                saving(self.biases[i], f)


if __name__ == '__main__':
    pass