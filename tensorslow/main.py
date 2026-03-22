# model.fit(train_images, train_labels, epochs=10)
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def fit(self, train_x, train_y, epochs):
        pass
