from src.train_model import train_lstm_model
from src.evaluate_model import evaluate_and_plot

if __name__ == "__main__":
    # Eğit
    train_lstm_model(seq_len=100, epochs=30, batch_size=32)

    # Değerlendir + plot
    evaluate_and_plot(seq_len=100, model_path="models/lstm_soh_model.h5", n_points=300)
