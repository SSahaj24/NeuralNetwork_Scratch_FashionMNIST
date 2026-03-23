import numpy as np
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def step(self, model):
        for layer in model.layers:
            if hasattr(layer, 'dW'): # To skip layers without params
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db
