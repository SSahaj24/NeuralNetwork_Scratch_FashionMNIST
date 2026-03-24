import numpy as np

class RandomUniformInitializer:
    def __init__(self):
        return
    def __call__(self, n_in, n_out):
        return np.random.randn(n_in, n_out)

class XavierUniformInitializer:
    def __init__(self):
        return
    def __call__(self, n_in, n_out):
        limit = np.sqrt(6.0 / (n_in + n_out))
        return np.random.uniform(low=-limit, high=limit, size=(n_in, n_out))
