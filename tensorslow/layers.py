# Layers
import numpy as np

class ReLU:
    def __init__(self):
        pass
    def forward(self, x):
        return (x > 0) * x

class Softmax:
    def __init__(self):
        pass
    def forward(self, x):
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1, keepdims=True)

class Dense:
    def __init__(self, n, activation=None):
        self.n = n
        self.neurons = np.random.rand(n)
        self.W = None
        self.b = None
        if activation=="relu":
            self.activation = ReLU()
        else:
            self.activation = None

    def forward(self, x):
        if self.W is None:
            input_dim = x.shape[1]
            self.W = np.random.randn(input_dim, self.n)
            self.b = np.random.randn(1, self.n)

        z = x @ self.W + self.b

        if self.activation:
            return self.activation.forward(z)
        return z


class Flatten:
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def forward(self, x):
        assert x.shape[1:] == self.input_shape
        return x.reshape(x.shape[0], -1)

