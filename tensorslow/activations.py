import numpy as np

class ReLU:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x>0)
        return self.mask * x
    def backward(self, grad):
        return self.mask * grad

class tanh:
    def __init__(self):
        self.out=None
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    def backward(self, grad):
        return grad * (1.0 - np.square(self.out))

class Softmax:
    def __init__(self):
        return
    def forward(self, x):
        e_x = np.exp(x - x.max(axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    # Backward is complicated; integrate with lossfunction instead
