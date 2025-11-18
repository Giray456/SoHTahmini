# src/train_model.py

import joblib
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .preprocess import train_test_split_dataset
from .model_lstm import build_lstm_model
from .utils import MODELS_DIR


def train_lstm_model(seq_len: int = 100, epochs: int = 50, batch_size: int = 32):
    # Veri setini hazırla
    X_train, X_test, y_train, y_test, scaler = train_test_split_dataset(
        seq_len=seq_len
    )

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    # Callback'ler: erken durdurma + öğrenme oranı azaltma
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # Model ve scaler'ı kaydet
    model_path = MODELS_DIR / "lstm_soh_model.h5"
    scaler_path = MODELS_DIR / "feature_scaler.pkl"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"[INFO] Model kaydedildi: {model_path}")
    print(f"[INFO] Scaler kaydedildi: {scaler_path}")

    # Test performansı
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Test Loss (MSE): {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    return history, (test_loss, test_mae)
