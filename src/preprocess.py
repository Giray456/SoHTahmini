# src/preprocess.py
from typing import List, Tuple, Dict, Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .load_data import load_all_batteries


def resample_series(series: np.ndarray, target_len: int = 100) -> np.ndarray:
    """
    Zaman serisini (N,) boyutundan (target_len,) boyutuna lineer olarak yeniden örnekler.
    """
    n = len(series)
    if n == target_len:
        return series

    old_idx = np.linspace(0, 1, n)
    new_idx = np.linspace(0, 1, target_len)
    return np.interp(new_idx, old_idx, series)


def build_dataset(
    target_len: int = 100,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Tüm bataryalardan deşarj döngülerini alır,
    (num_samples, target_len, 3) şeklinde X,
    (num_samples,) şeklinde y (SoH) döndürür.
    """
    raw_cycles = load_all_batteries()

    capacities = np.array([c["capacity"] for c in raw_cycles], dtype=np.float32)
    first_capacity = capacities.max()  # kabaca yeni pil kapasitesi gibi düşün
    soh = capacities / first_capacity

    X_list = []
    for c in raw_cycles:
        v = resample_series(c["voltage"], target_len)
        i = resample_series(c["current"], target_len)
        t = resample_series(c["temperature"], target_len)

        stacked = np.vstack([v, i, t]).T  # (target_len, 3)
        X_list.append(stacked)

    X = np.stack(X_list, axis=0)  # (num_samples, target_len, 3)

    # Özellikleri 0-1 aralığına çekelim
    num_samples, seq_len, num_features = X.shape
    scaler = MinMaxScaler()
    X_reshaped = X.reshape(-1, num_features)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(num_samples, seq_len, num_features)

    y = soh.astype(np.float32)

    return X_scaled, y, scaler


def train_test_split_dataset(
    test_size: float = 0.2, random_state: int = 42, seq_len: int = 100
):
    X, y, scaler = build_dataset(target_len=seq_len)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    return X_train, X_test, y_train, y_test, scaler
