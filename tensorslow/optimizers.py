import numpy as np
from collections import defaultdict
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def step(self, model):
        for layer in model.layers:
            if hasattr(layer, 'dW'): # To skip layers without params
                # dW = layer.x.T @ layer.grad
                # db = layer.grad.sum(axis=0, keepdims=True)
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

class MomentumGD:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.u_W = defaultdict(lambda:0)
        self.u_b = defaultdict(lambda:0)
        
    def step(self, model):
        for layer in model.layers:
            if hasattr(layer, 'W') and layer.W is not None:
                self.u_W[layer] = self.beta * self.u_W[layer] + layer.dW
                self.u_b[layer] = self.beta * self.u_b[layer] + layer.db
                layer.W -= self.lr * self.u_W[layer]
                layer.b -= self.lr * self.u_b[layer]

class NesterovGD:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.u_W = defaultdict(lambda:0)
        self.u_b = defaultdict(lambda:0)

    def step(self, model):
        for layer in model.layers:
            if hasattr(layer, 'dW'):
                self.u_W[layer] = self.beta * self.u_W[layer] - self.lr * layer.dW
                self.u_b[layer] = self.beta * self.u_b[layer] - self.lr * layer.db

                layer.W += self.beta * self.u_W[layer] - self.lr * layer.dW
                layer.b += self.beta * self.u_b[layer] - self.lr * layer.db


class RMSprop:
    def __init__(self, lr=0.01, beta=0.999):
        self.lr = lr
        self.beta = beta
        self.v = 0
        self.epsilon = 1e-8
    def step(self, model):
        for layer in model.layers:
            if hasattr(layer, 'dW'):
                self.v = self.beta * self.v + (1-self.beta) * np.square(layer.dW)
                self.w = self.w - self.eta / (np.sqrt(self.v + self.epsilon)) * layer.dW

class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_W = defaultdict(lambda:0) 
        self.m_b = defaultdict(lambda:0)
        self.v_W = defaultdict(lambda:0)
        self.v_b = defaultdict(lambda:0)
        self.t = 0
        self.epsilon = 1e-8
    def step(self, model):
        self.t += 1
        for layer in model.layers:
            if hasattr(layer, 'dW'):
                self.m_W[layer] = self.beta1 * self.m_W[layer] + (1-self.beta1) * layer.dW
                self.v_W[layer] = self.beta2 * self.v_W[layer] + (1-self.beta2) * (layer.dW)**2 
                self.m_b[layer] = self.beta1 * self.m_b[layer] + (1-self.beta1) * layer.db
                self.v_b[layer] = self.beta2 * self.v_b[layer] + (1-self.beta2) * (layer.db)**2
                
                self.v_b_cap = self.v_b[layer] / (1 - self.beta2**self.t)
                self.m_W_cap = self.m_W[layer] / (1 - self.beta1**self.t)
                self.v_W_cap = self.v_W[layer] / (1 - self.beta2**self.t)
                self.m_b_cap = self.m_b[layer] / (1 - self.beta1**self.t)

                layer.W -= self.lr / (np.sqrt(self.v_W_cap) + self.epsilon) * self.m_W_cap
                layer.b -= self.lr / (np.sqrt(self.v_b_cap) + self.epsilon) * self.m_b_cap


class Nadam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_W = defaultdict(lambda:0) 
        self.m_b = defaultdict(lambda:0)
        self.v_W = defaultdict(lambda:0)
        self.v_b = defaultdict(lambda:0)
        self.t = 0
        self.epsilon = 1e-8
    def step(self, model):
        self.t += 1
        for layer in model.layers:
            if hasattr(layer, 'dW'):
                self.m_W[layer] = self.beta1 * self.m_W[layer] + (1-self.beta1) * layer.dW
                self.v_W[layer] = self.beta2 * self.v_W[layer] + (1-self.beta2) * (layer.dW)**2 
                self.m_b[layer] = self.beta1 * self.m_b[layer] + (1-self.beta1) * layer.db
                self.v_b[layer] = self.beta2 * self.v_b[layer] + (1-self.beta2) * (layer.db)**2
                
                self.v_b_cap = self.v_b[layer] / (1 - self.beta2**self.t)
                self.m_W_cap = self.m_W[layer] / (1 - self.beta1**self.t)
                self.v_W_cap = self.v_W[layer] / (1 - self.beta2**self.t)
                self.m_b_cap = self.m_b[layer] / (1 - self.beta1**self.t)

                layer.W -= self.lr / (np.sqrt(self.v_W_cap) + self.epsilon) * (self.beta1 * self.m_W_cap + (1-self.beta1) * layer.dW / (1-self.beta1**(self.t+1)))
                layer.b -= self.lr / (np.sqrt(self.v_b_cap) + self.epsilon) * (self.beta1 * self.m_b_cap + (1-self.beta1) * layer.db / (1-self.beta1**(self.t+1)))
