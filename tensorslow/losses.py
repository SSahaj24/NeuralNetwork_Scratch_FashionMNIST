import numpy as np

class CategoricalCrossEntropy:
    def __init__(self, label_smoothing=None):
        self.label_smoothing = label_smoothing
        self.model = None # Set at instance creation
    def __call__(self, y_true, y_pred):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1-eps)
        if self.label_smoothing:
            y_true = y_true * (1-self.label_smoothing) + (self.label_smoothing / y_true.shape[1])
        loss = -np.sum( y_true * np.log(y_pred), axis = 1)
        return np.mean(loss)

    def backward(self, y_true, y_pred):
        if self.model.layers[-1].__class__.__name__ == "Softmax":
            return (y_pred - y_true)
        else:
            raise NotImplementedError("CCE backward is only implemented for Softmax output")
