import os
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model import Autoencoder


X_TRAIN_PATH = "data/processed/X_train.csv"
X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"
MODEL_PATH = "models/autoencoder_model.pth"
RESULT_PATH = "results/metrics/threshold_experiment_results.txt"


def load_data():
    """
    Eğitim ve test verilerini okur.
    """

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    return X_train, X_test, y_test


def load_model(device):
    """
    Eğitilmiş Autoencoder modelini yükler.
    """

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    input_dim = checkpoint["input_dim"]

    model = Autoencoder(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def calculate_errors(model, X, device):
    """
    Verilen veri için reconstruction error hesaplar.
    """

    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed = model(X_tensor)

        errors = torch.mean(
            (X_tensor - reconstructed) ** 2,
            dim=1
        ).cpu().numpy()

    return errors


def evaluate_threshold(y_true, test_errors, threshold):
    """
    Belirli bir threshold değeri için model performansını hesaplar.
    """

    predictions = (test_errors > threshold).astype(int)

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    cm = confusion_matrix(y_true, predictions)

    return accuracy, precision, recall, f1, cm


def main():
    """
    Farklı threshold değerlerini karşılaştırır.
    """

    os.makedirs("results/metrics", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    X_train, X_test, y_test = load_data()
    model = load_model(device)

    train_errors = calculate_errors(model, X_train, device)
    test_errors = calculate_errors(model, X_test, device)

    percentiles = [90, 95, 97, 98, 99, 99.5]

    lines = []
    lines.append("Threshold Experiment Results")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Percentile | Threshold | Accuracy | Precision | Recall | F1-score | Confusion Matrix")
    lines.append("-" * 80)

    for percentile in percentiles:
        threshold = np.percentile(train_errors, percentile)

        accuracy, precision, recall, f1, cm = evaluate_threshold(
            y_true=y_test,
            test_errors=test_errors,
            threshold=threshold
        )

        result_line = (
            f"{percentile:>10} | "
            f"{threshold:.8f} | "
            f"{accuracy:.6f} | "
            f"{precision:.6f} | "
            f"{recall:.6f} | "
            f"{f1:.6f} | "
            f"{cm.tolist()}"
        )

        lines.append(result_line)

        print(result_line)

    with open(RESULT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print(f"\nThreshold deneme sonuçları kaydedildi: {RESULT_PATH}")


if __name__ == "__main__":
    main() 