# model.fit(train_images, train_labels, epochs=10)
from tensorslow import losses, optimizers, metrics
import numpy as np

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.loss = None
        self.optim = None
        self.metric = None
    
    def compile(self, loss=None, optim=None, metric=None):
        if isinstance(loss, str):
            if loss=='CategoricalCrossEntropy':
                self.loss = losses.CategoricalCrossEntropy()
            else:
                raise NotImplementedError(f"{loss} not implemented")
        else:
            self.loss = loss
        if self.loss:
            self.loss.model = self

        if isinstance(optim, str):
            if optim=='SGD':
                self.optimizer = optimizers.SGD()
            elif optim=='NesterovGD':
                self.optimizer = optimizers.NesterovGD(lr=0.01)
            elif optim=="Adam":
                self.optimizer = optimizers.Adam()
            elif optim=="Nadam":
                self.optimizer = optimizers.Nadam()
            else:
                raise NotImplementedError(f"{optim} not implemented")
        else:
            self.optimizer = optim
    
        if isinstance(metric, str):
            if metric=='accuracy':
                self.metric = metrics.Accuracy()
            else:
                raise NotImplementedError(f"{metric} not implemented")
        else:
            self.metric = metric

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true, y_pred):
        grad = self.loss.backward(y_true, y_pred)
        for layer in self.layers[-2::-1]:
            grad = layer.backward(grad)
        return grad

    def fit(self, train_x, train_y, epochs, batch_size=32):
        for epoch in range(epochs):
            # shuffle
            idx = np.random.permutation(train_x.shape[0])
            train_x = train_x[idx]
            train_y = train_y[idx]

            epoch_loss = 0
            epoch_acc = 0
            for i in range(0, train_x.shape[0], batch_size):
                x_batch = train_x[i:i+batch_size]
                y_batch = train_y[i:i+batch_size]

                y_pred = self.forward(x_batch)
                epoch_loss += self.loss(y_batch, y_pred)
                if self.metric:
                    epoch_acc += self.metric(y_batch, y_pred)

                self.backward(y_batch, y_pred)
                self.optimizer.step(self)
            # num_batches = train_x.shape[0] // batch_size
            num_batches = np.ceil(train_x.shape[0] / batch_size)
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss/num_batches:.4f} - acc: {epoch_acc/num_batches:.4f}")
    
    def predict(self, x):
        return self.forward(x)

    def evaluate(self, x, y):
        y_pred = self.forward(x)
        loss = self.loss(y, y_pred)
        metric = self.metric(y, y_pred) if self.metric else None
        print(f"{self.loss.__class__.__name__} Loss: {loss:.4f} - {self.metric.__class__.__name__} Metric: {metric:.4f}")
        return loss, metric
