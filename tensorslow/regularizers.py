import numpy as np

class L1:
    def __init__(self, l1=0):
        self.l1 = l1
    def __call__(self, x):
        return self.l1 * np.sign(x)

class L2:
    def __init__(self, l2=0):
        self.l2 = l2

    def __call__(self, x):
        return 2 * self.l2 * x

class L1L2:
    def __init__(self, l1=0, l2=0):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, x):
        return self.l1 * np.sign(x) + (2 * self.l2) * x
