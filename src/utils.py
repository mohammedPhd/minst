# src/utils.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical

def load_and_prepare_for_cnn():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1) # (N,28,28,1)
    x_test = np.expand_dims(x_test, -1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def load_and_prepare_for_lstm():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    # On considère chaque ligne de 28 pixels comme un pas de temps
    # shape finale (N, timesteps=28, features=28)
    x_train = x_train.reshape((-1, 28, 28))
    x_test = x_test.reshape((-1, 28, 28))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def plot_history(history, out_prefix):
    # history : tf.keras.callbacks.History
    import os
    os.makedirs('outputs/figures', exist_ok=True)
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outputs/figures/{out_prefix}_accuracy.png')
    plt.close()
    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outputs/figures/{out_prefix}_loss.png')
    plt.close()

def save_results(name, train_time, test_loss, test_acc):
    import os
    os.makedirs('outputs', exist_ok=True)
    path = 'outputs/results.csv'
    df = pd.DataFrame([[name, train_time, test_loss, test_acc]],
    columns=['model','train_time_s','test_loss','test_acc'])
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode='a', header=False, index=False)