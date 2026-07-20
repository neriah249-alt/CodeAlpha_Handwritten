import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    print(f"Données chargées : {x_train.shape[0]} entraînement, {x_test.shape[0]} test.")
    return (x_train, y_train), (x_test, y_test)