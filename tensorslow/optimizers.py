import numpy as np
from collections import defaultdict
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def step(self, model):
        for layer in model.layers:
            if hasattr(layer, 'dW'): # To skip layers without params
                if layer.regularizer:
                    regularization_W = layer.regularizer(layer.W)
                else:
                    regularization_W = 0
                dW = layer.dW + regularization_W
                db = layer.db
                layer.W -= self.lr * dW

class MomentumGD:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.u_W = defaultdict(lambda:0)
        self.u_b = defaultdict(lambda:0)
        
    def step(self, model):
        for layer in model.layers:
            if hasattr(layer, 'W') and layer.W is not None:
                if layer.regularizer:
                    regularization_W = layer.regularizer(layer.W)
                else:
                    regularization_W = 0

                dW = layer.dW + regularization_W
                db = layer.db

                self.u_W[layer] = self.beta * self.u_W[layer] + dW
                layer.W -= self.lr * self.u_W[layer]

class NesterovGD:
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0005):
        self.lr = lr
        self.beta = beta
        self.u_W = defaultdict(lambda:0)
        self.u_b = defaultdict(lambda:0)

    def step(self, model):
        for layer in model.layers:
            if hasattr(layer, 'dW'):
                if layer.regularizer:
                    regularization_W = layer.regularizer(layer.W)
                else:
                    regularization_W = 0
                
                dW = layer.dW + regularization_W
                db = layer.db

                self.u_W[layer] = self.beta * self.u_W[layer] - self.lr * dW
                self.u_b[layer] = self.beta * self.u_b[layer] - self.lr * db
                layer.W += self.beta * self.u_W[layer] - (self.lr * dW)
                layer.b += self.beta * self.u_b[layer] - (self.lr * db)


class RMSprop:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v_W = defaultdict(lambda:0)
        self.v_b = defaultdict(lambda:0)
        self.epsilon = 1e-8
    def step(self, model):
        for layer in model.layers:
            if hasattr(layer, 'dW'):
                if layer.regularizer:
                    regularization_W = layer.regularizer(layer.W)
                else:
                    regularization_W = 0
                
                dW = layer.dW + regularization_W
                db = layer.db

                self.v_W[layer] = self.beta * self.v_W[layer] + (1-self.beta) * dW**2 
                self.v_b[layer] = self.beta * self.v_b[layer] + (1-self.beta) * db**2
                layer.W -= self.lr / (np.sqrt(self.v_W[layer]) + self.epsilon) * dW
                layer.b -= self.lr / (np.sqrt(self.v_b[layer]) + self.epsilon) * db

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
                if layer.regularizer:
                    regularization_W = layer.regularizer(layer.W)
                else:
                    regularization_W = 0
                
                dW = layer.dW + regularization_W
                db = layer.db 

                self.m_W[layer] = self.beta1 * self.m_W[layer] + (1-self.beta1) * dW
                self.v_W[layer] = self.beta2 * self.v_W[layer] + (1-self.beta2) * dW**2 
                self.m_b[layer] = self.beta1 * self.m_b[layer] + (1-self.beta1) * db
                self.v_b[layer] = self.beta2 * self.v_b[layer] + (1-self.beta2) * db**2
                
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
                if layer.regularizer:
                    regularization_W = layer.regularizer(layer.W)
                else:
                    regularization_W = 0
                dW = layer.dW + regularization_W
                db = layer.db

                self.m_W[layer] = self.beta1 * self.m_W[layer] + (1-self.beta1) * dW
                self.v_W[layer] = self.beta2 * self.v_W[layer] + (1-self.beta2) * dW**2 
                self.m_b[layer] = self.beta1 * self.m_b[layer] + (1-self.beta1) * db
                self.v_b[layer] = self.beta2 * self.v_b[layer] + (1-self.beta2) * db**2
                
                self.v_b_cap = self.v_b[layer] / (1 - self.beta2**self.t)
                self.m_W_cap = self.m_W[layer] / (1 - self.beta1**self.t)
                self.v_W_cap = self.v_W[layer] / (1 - self.beta2**self.t)
                self.m_b_cap = self.m_b[layer] / (1 - self.beta1**self.t)

                layer.W -= self.lr / (np.sqrt(self.v_W_cap) + self.epsilon) * (self.beta1 * self.m_W_cap + (1-self.beta1) * layer.dW / (1-self.beta1**(self.t+1)))
                layer.b -= self.lr / (np.sqrt(self.v_b_cap) + self.epsilon) * (self.beta1 * self.m_b_cap + (1-self.beta1) * layer.db / (1-self.beta1**(self.t+1)))
