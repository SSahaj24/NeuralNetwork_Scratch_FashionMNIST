# Layers
import numpy as np

class ReLU:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x>0)
        return self.mask * x
    def backward(self, grad):
        return self.mask * grad

class Softmax:
    def __init__(self):
        return
    def forward(self, x):
        # e_x = np.exp(x)
        # return e_x / e_x.sum(axis=1, keepdims=True) # axis=0 is the batch
        e_x = np.exp(x - x.max(axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    # def backward(self,

class Dense:
    def __init__(self, n, activation=None):
        self.n = n
        self.neurons = np.random.rand(n)
        self.W = None
        self.b = None
        self.x = None
        if activation=="relu":
            self.activation = ReLU()
        else:
            self.activation = None

    def forward(self, x):
        self.x = x
        if self.W is None:
            input_dim = x.shape[1]
            self.W = np.random.randn(input_dim, self.n)
            self.b = np.random.randn(1, self.n)

        z = x @ self.W + self.b

        if self.activation:
            return self.activation.forward(z)
        return z

    def backward(self, grad):
        if self.activation:
            grad = self.activation.backward(grad)
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0, keepdims=True)
        return grad @ self.W.T


class Flatten:
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def forward(self, x):
        assert x.shape[1:] == self.input_shape
        return x.reshape(x.shape[0], -1)
    def backward(self, grad):
        return grad.reshape(-1, *self.input_shape) # Need to unpack 
