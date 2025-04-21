"""
RNN models and training routines.
"""
import tensorflow as tf
import numpy as np
from typing import Tuple
from .metrics import smape, wape, relative_mse


def build_rnn(window: int,
              hidden_units: int,
              horizon: int,
              activation: str,
              dropout_rate: float = 0.2) -> tf.keras.Model:
    """Builds and compiles a simple RNN forecasting model."""
    initializer = tf.keras.initializers.HeNormal()
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(hidden_units,
                                  activation=activation,
                                  input_shape=(window, 1),
                                  kernel_initializer=initializer),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(horizon)
    ])
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['mae']
    )
    return model


def train_and_extract(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int
) -> Tuple[tf.keras.callbacks.History, np.ndarray]:
    """
    Trains `model` and returns training history + activations on validation set.
    Activations extracted from RNN layer at each epoch.
    """
    # store weights callback
    weights_per_epoch = []
    def on_epoch_end(epoch, logs):
        weights_per_epoch.append(model.layers[0].get_weights())

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)]
    )
    # extract activations
    activations = []
    rnn_layer = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(model.layers[0].units,
                                  return_sequences=True,
                                  activation=model.layers[0].activation,
                                  kernel_initializer='zeros')
    ])
    for w in weights_per_epoch:
        rnn_layer.layers[0].set_weights(w)
        activations.append(rnn_layer.predict(X_val))

    return history, np.array(activations)


def plot_loss(history, save_path: str) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()