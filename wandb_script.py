import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import wandb
import tensorslow as ts
# from tensorslow.losses import CategoricalCrossEntropy

def build_optimizer(cfg):
    if cfg.optimizer == "SGD":
        return ts.optimizers.SGD(lr=cfg.lr)
    elif cfg.optimizer == "Adam":
        return ts.optimizers.Adam(lr=cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2)
    elif cfg.optimizer == "Nadam":
        return ts.optimizers.Nadam(lr=cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2)
    elif cfg.optimizer == "RMSprop":
        return ts.optimizers.RMSprop(lr=cfg.lr, beta=cfg.beta2)
    elif cfg.optimizer == "MomentumGD":
        return ts.optimizers.MomentumGD(lr=cfg.lr, beta=cfg.beta1)
    else:
        raise NotImplementedError(f"Unknown optimizer: {cfg.optimizer}")

wandb.init(
    project="fashionmnist-tensorslow",
    config={
        "lr": 0.0005,
        "beta1": 0.9,
        "beta2": 0.999,
        "optimizer": "Nadam",
        "batch_size": 32,
        "epochs": 10,
        "hidden_units": 128,
        "activation": "tanh",
        "kernel_initializer": "xavier",
        "label_smoothing": 0,
    }
)
cfg = wandb.config  # use this everywhere below


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def to_one_hot(y, num_classes=10):
    return np.eye(num_classes)[y] # y'th row of I 
train_labels = to_one_hot(train_labels)
test_labels = to_one_hot(test_labels)
train_images = train_images / 255.0
test_images = test_images / 255.0

split = int(0.9 * len(train_images))
val_images, val_labels = train_images[split:], train_labels[split:]
train_images, train_labels = train_images[:split], train_labels[:split]



model = ts.Sequential([
    ts.layers.Flatten(input_shape=(28,28)),
    ts.layers.Dense(cfg.hidden_units, activation=cfg.activation, kernel_initializer=cfg.kernel_initializer),
    ts.layers.Dense(cfg.hidden_units, activation=cfg.activation, kernel_initializer=cfg.kernel_initializer),
    ts.layers.Dense(cfg.hidden_units, activation=cfg.activation, kernel_initializer=cfg.kernel_initializer),
    ts.layers.Dense(10),
    ts.activations.Softmax()
], use_wandb=True)

loss = ts.losses.CategoricalCrossEntropy(label_smoothing=cfg.label_smoothing)
optim = build_optimizer(wandb.config)
model.compile(loss=loss, optim=optim, metric='accuracy')
model.fit(train_images, train_labels, epochs=cfg.epochs, batch_size=cfg.batch_size, val_data  = (val_images, val_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
wandb.finish()

