import numpy as np

class CategoricalCrossEntropy:
    def __init__(self, label_smoothing=None):
        self.label_smoothing = label_smoothing
        self.model = None # Set at instance creation
        self.y_true = None
        self.y_pred = None
    def __call__(self, y_true, y_pred):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1-eps)
        if self.label_smoothing:
            y_true = y_true * (1-self.label_smoothing) + (self.label_smoothing / y_true.shape[1])
        self.y_true = y_true
        self.y_pred = y_pred

        loss = -np.sum( y_true * np.log(y_pred), axis = 1)
        return np.mean(loss)

    def backward(self):
        if self.model.layers[-1].__class__.__name__ == "Softmax":
            return (self.y_pred - self.y_true)
        else:
            raise NotImplementedError("CCE backward is only implemented for Softmax output")
