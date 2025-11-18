# src/evaluate_model.py

import os
import numpy as np

import matplotlib
# Tk penceresiyle uğraşmamak için non-GUI backend kullanıyoruz
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from .preprocess import train_test_split_dataset
from .utils import PROJECT_ROOT


def evaluate_and_plot(
    seq_len: int = 100,
    model_path: str = "models/lstm_soh_model.h5",
    n_points: int = 300,
):
    """
    Eğitilmiş modeli test verisi üzerinde değerlendirir,
    metrikleri hesaplar ve:
      - Gerçek vs Tahmin SoH grafiğini
      - Tahmin hatası (Gerçek - Tahmin) profilini
    PNG olarak kaydeder.
    """

    print(f"[INFO] Model yükleniyor: {model_path}")
    model = load_model(model_path, compile=False)

    print("[INFO] Veri seti yeniden hazırlanıyor...")
    X_train, X_test, y_test_input, y_test, _ = train_test_split_dataset(
        seq_len=seq_len
    )

    print("[INFO] Test verisi için tahmin alınıyor...")
    y_pred = model.predict(X_test, verbose=0)

    # Tüm test seti için metrikler
    mse_all = float(np.mean((y_test - y_pred) ** 2))
    mae_all = float(np.mean(np.abs(y_test - y_pred)))

    print(f"[RESULT] Tüm test seti için MSE: {mse_all:.4f}, MAE: {mae_all:.4f}")

    # Plot için ilk n_points
    n_points = min(n_points, len(y_test))
    y_true_plot = y_test[:n_points]
    y_pred_plot = y_pred[:n_points]

    mse_plot = float(np.mean((y_true_plot - y_pred_plot) ** 2))
    mae_plot = float(np.mean(np.abs(y_true_plot - y_pred_plot)))

    print(
        f"[RESULT] Plot aralığı (ilk {n_points} örnek) için MSE: "
        f"{mse_plot:.4f}, MAE: {mae_plot:.4f}"
    )

    # Sonuçları kaydedeceğimiz klasör
    results_dir = PROJECT_ROOT / "results"
    os.makedirs(results_dir, exist_ok=True)

    # 1) Gerçek vs Tahmin SoH grafiği
    plt.figure(figsize=(12, 5))
    plt.plot(
        y_true_plot,
        label="Gerçek SoH",
        linewidth=1.7,
    )
    plt.plot(
        y_pred_plot,
        label="Tahmin SoH",
        linestyle="--",
        linewidth=1.7,
    )
    plt.xlabel("Örnek İndeksi", fontsize=11)
    plt.ylabel("SoH (normalize)", fontsize=11)
    plt.title(
        f"SoH Tahmini - İlk {n_points} Örnek\n"
        f"(MAE={mae_plot:.3f}, MSE={mse_plot:.3f})",
        fontsize=13,
    )
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pred_path = results_dir / "soh_prediction.png"
    plt.savefig(pred_path, dpi=300)
    plt.close()

    # 2) Hata profili grafiği (Gerçek - Tahmin)
    errors = (y_true_plot - y_pred_plot).ravel()

    plt.figure(figsize=(12, 4))
    plt.plot(errors, linewidth=1.5)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Örnek İndeksi", fontsize=11)
    plt.ylabel("Hata (Gerçek - Tahmin)", fontsize=11)
    plt.title("Tahmin Hatası Profili", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    err_path = results_dir / "soh_error_profile.png"
    plt.savefig(err_path, dpi=300)
    plt.close()

    print(f"[INFO] Grafikler kaydedildi:")
    print(f"       - {pred_path}")
    print(f"       - {err_path}")
