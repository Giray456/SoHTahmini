# src/model_lstm.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout


def build_lstm_model(input_shape):
    """
    input_shape: (sequence_length, n_features)
    """

    model = Sequential(
        [
            # Uyarıyı çözmek için Input layer kullanıyoruz
            Input(shape=input_shape),

            # 1. katman: Bidirectional LSTM (sequence döndürür)
            Bidirectional(
                LSTM(64, return_sequences=True),
                name="bilstm_1"
            ),
            Dropout(0.2),

            # 2. katman: Bidirectional LSTM (son state)
            Bidirectional(
                LSTM(32, return_sequences=False),
                name="bilstm_2"
            ),
            Dropout(0.2),

            # Tam bağlantılı katmanlar
            Dense(32, activation="relu", name="dense_1"),
            Dense(16, activation="relu", name="dense_2"),

            # Çıkış: SoH (normalize tek değer)
            Dense(1, activation="linear", name="output"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    model.summary()
    return model
