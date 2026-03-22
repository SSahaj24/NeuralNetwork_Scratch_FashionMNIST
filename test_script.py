import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

import tensorslow as ts

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print("Train Dim", train_images.shape)
print("Train Label Count", len(train_labels))
print("Train Labels", train_labels)
print("Test Dim", test_images.shape)
print("Test Label Count", len(test_labels))

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
    ts.layers.Dense(128, activation='relu'),
    ts.layers.Dense(10)
])

print("Input Shape: ", train_images[:32].shape)
print("Layers: Flatten(28,28), Dense(128, ReLU), Dense(10)")
output = model.forward(train_images[:32])
print("Output Shape: ", output.shape)
print("Output: ", output)
