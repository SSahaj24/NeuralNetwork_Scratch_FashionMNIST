import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

import tensorslow as ts
# from tensorslow.losses import CategoricalCrossEntropy

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print("Train Dim", train_images.shape)
print("Train Label Count", len(train_labels))
print("Train Labels", train_labels)
print("Test Dim", test_images.shape)
print("Test Label Count", len(test_labels))

def to_one_hot(y, num_classes=10):
    return np.eye(num_classes)[y] # y'th row of I 
train_labels = to_one_hot(train_labels)
test_labels = to_one_hot(test_labels)
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = ts.Sequential([
    ts.layers.Flatten(input_shape=(28,28)),
    ts.layers.Dense(128, activation="tanh", kernel_initializer="xavier"),
    ts.layers.Dense(128, activation="tanh", kernel_initializer="xavier"),
    ts.layers.Dense(128, activation="tanh", kernel_initializer="xavier"),
    ts.layers.Dense(10, activation="tanh"),
    ts.activations.Softmax()
])

# print("Input Shape: ", train_images[:32].shape)
# output = model.forward(train_images[:32])
# print("Output Shape: ", output.shape)
# print("Output: ", output)

loss = ts.losses.CategoricalCrossEntropy(label_smoothing=0)

# optim = ts.optimizers.SGD(lr=0.01)
# optim = ts.optimizers.MomentumGD(lr=0.01, beta=0.9)
# optim = ts.optimizers.NesterovGD(lr=0.01, beta=0.9)
# optim = ts.optimizers.RMSprop(lr=0.01, beta=0.99)
# optim = ts.optimizers.Adam(lr=0.01, beta1=0.9, beta2=0.999)
optim = ts.optimizers.Nadam(lr=0.0005, beta1=0.9, beta2=0.999)

model.compile(loss=loss, optim=optim, metric='accuracy')
model.fit(train_images, train_labels, epochs=10, batch_size=32)

model.evaluate(test_images, test_labels)
