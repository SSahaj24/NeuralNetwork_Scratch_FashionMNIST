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
    elif cfg.optimizer == "NesterovGD":
        return ts.optimizers.NesterovGD(lr=cfg.lr, beta=cfg.beta1, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError(f"Unknown optimizer: {cfg.optimizer}")

def build_regularizer(cfg):
    if cfg.weight_decay_type == "L1":
        return ts.regularizers.L1(l1=cfg.weight_decay)
    elif cfg.weight_decay_type == "L2":
        return ts.regularizers.L2(l2=cfg.weight_decay)
    elif cfg.weight_decay_type == "L1L2":
        return ts.regularizers.L1L2(l1=cfg.weight_decay, l2=cfg.weight_decay)
    else:
        return None

wandb.init(
    project="fashionmnist-tensorslow",
    config={
        "lr": 0.0005,
        "beta1": 0.9,
        "beta2": 0.999,
        "optimizer": "Nadam",
        "batch_size": 32,
        "epochs": 10,
        "num_layers": 3,
        "hidden_units": 128,
        "activation": "tanh",
        "kernel_initializer": "xavier",
        "label_smoothing": 0.1,
        "weight_decay": 0.0005,
        "weight_decay_type": "L2",
    }
)
cfg = wandb.config


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

def to_one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]
train_labels = to_one_hot(train_labels)
test_labels = to_one_hot(test_labels)
train_images = train_images / 255.0
test_images = test_images / 255.0

split = int(0.9 * len(train_images))
val_images, val_labels = train_images[split:], train_labels[split:]
train_images, train_labels = train_images[:split], train_labels[:split]

regularizer = build_regularizer(cfg)

layers = [ts.layers.Flatten(input_shape=(28,28))]
for _ in range(cfg.num_layers):
    layers.append(ts.layers.Dense(cfg.hidden_units, 
                                  activation=cfg.activation, 
                                  kernel_initializer=cfg.kernel_initializer,
                                  kernel_regularizer=regularizer))
layers.append(ts.layers.Dense(10))
layers.append(ts.activations.Softmax())

model = ts.Sequential(layers, use_wandb=True)

loss = ts.losses.CategoricalCrossEntropy(label_smoothing=cfg.label_smoothing)
optim = build_optimizer(cfg)
model.compile(loss=loss, optim=optim, metric='accuracy')
model.fit(train_images, train_labels, epochs=cfg.epochs, batch_size=cfg.batch_size, val_data  = (val_images, val_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
wandb.finish()

