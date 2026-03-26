# Layers
import numpy as np
import tensorslow as ts

# TODO - isinstance checking

class Dense:
    def __init__(self, n, activation=None, kernel_initializer=None, kernel_regularizer=None):
        self.n = n
        self.W = None
        self.b = None
        self.x = None
        self.reg_loss_W = 0
        if activation=="relu":
            self.activation = ts.activations.ReLU()
        elif activation=="tanh":
            self.activation = ts.activations.tanh()
        else:
            self.activation = None

        if kernel_initializer=="xavier":
            self.initializer = ts.initializers.XavierUniformInitializer()
        else:
            self.initializer = ts.initializers.RandomNormalInitializer()
        
        if isinstance(kernel_regularizer, str):
            raise TypeError("Please pass the regularizer object")
        else:
            self.regularizer = kernel_regularizer

    def forward(self, x):
        self.x = x
        if self.W is None:
            input_dim = x.shape[1]

            if self.initializer:
                # self.W = np.random.randn(input_dim, self.n) * np.sqrt(2. / input_dim) # TODO - BETTER INITIALIZATION?
                self.W = self.initializer(input_dim, self.n)
                self.b = self.initializer(1, self.n)
        if self.regularizer:
            self.reg_loss_W = self.regularizer.loss(self.W)
        z = x @ self.W + self.b

        if self.activation:
            return self.activation.forward(z)
        return z

    def backward(self, grad):
        self.grad = grad
        if self.activation:
            self.grad = self.activation.backward(self.grad)
        batch_size = self.x.shape[0]
        self.dW = self.x.T @ self.grad / batch_size
        self.db = self.grad.sum(axis=0, keepdims=True) / batch_size
        return self.grad @ self.W.T


class Flatten:
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def forward(self, x):
        assert x.shape[1:] == self.input_shape
        return x.reshape(x.shape[0], -1)
    def backward(self, grad):
        return grad.reshape(-1, *self.input_shape) # Need to unpack 
