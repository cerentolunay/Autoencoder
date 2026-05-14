import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from model import Autoencoder


# İşlenmiş test verilerinin yolları
X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"

# Eğitilmiş model ve threshold dosyalarının yolları
MODEL_PATH = "models/autoencoder_model.pth"
THRESHOLD_PATH = "models/threshold.txt"

# Sonuçların kaydedileceği yollar
METRICS_SAVE_PATH = "results/metrics/evaluation_results.txt"
ERROR_DISTRIBUTION_PATH = "results/figures/reconstruction_error_distribution.png"
CONFUSION_MATRIX_PATH = "results/figures/confusion_matrix.png"


def load_test_data():
    """
    Test için hazırlanmış X_test ve y_test dosyalarını okur.

    X_test:
        Ölçeklendirilmiş test feature verileri.

    y_test:
        Gerçek sınıf etiketleri.
        0 → Normal
        1 → Fraud / Anomali
    """

    if not os.path.exists(X_TEST_PATH):
        raise FileNotFoundError(
            f"X_test dosyası bulunamadı: {X_TEST_PATH}\n"
            "Önce src/data_preprocessing.py dosyasını çalıştırmalısın."
        )

    if not os.path.exists(Y_TEST_PATH):
        raise FileNotFoundError(
            f"y_test dosyası bulunamadı: {Y_TEST_PATH}\n"
            "Önce src/data_preprocessing.py dosyasını çalıştırmalısın."
        )

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    return X_test, y_test


def load_threshold():
    """
    Eğitim aşamasında hesaplanan threshold değerini dosyadan okur.
    """

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(
            f"Threshold dosyası bulunamadı: {THRESHOLD_PATH}\n"
            "Önce src/train.py dosyasını çalıştırmalısın."
        )

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold = float(file.read().strip())

    return threshold


def load_trained_model(device: torch.device):
    """
    Eğitilmiş Autoencoder modelini yükler.

    Modeli yeniden oluşturabilmek için train.py içinde kaydedilen input_dim değeri kullanılır.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {MODEL_PATH}\n"
            "Önce src/train.py dosyasını çalıştırmalısın."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    input_dim = checkpoint["input_dim"]

    model = Autoencoder(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model yüklendi: {MODEL_PATH}")
    print(f"Input dimension: {input_dim}")

    return model


def calculate_reconstruction_errors(
    model: Autoencoder,
    X_test: pd.DataFrame,
    device: torch.device
):
    """
    Test verisi için reconstruction error değerlerini hesaplar.

    Her satır için:
        error = mean((orijinal veri - yeniden oluşturulan veri)^2)

    Bu değer yüksekse model o örneği iyi yeniden oluşturamamış demektir.
    """

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed = model(X_test_tensor)

        reconstruction_errors = torch.mean(
            (X_test_tensor - reconstructed) ** 2,
            dim=1
        ).cpu().numpy()

    return reconstruction_errors


def generate_predictions(reconstruction_errors: np.ndarray, threshold: float):
    """
    Reconstruction error değerlerini threshold ile karşılaştırarak tahmin üretir.

    Eğer reconstruction error threshold değerinden büyükse:
        1 → Anomali

    Aksi halde:
        0 → Normal
    """

    predictions = (reconstruction_errors > threshold).astype(int)
    return predictions


def evaluate_model(y_test, predictions):
    """
    Modelin tahminlerini gerçek etiketlerle karşılaştırır
    ve temel sınıflandırma metriklerini hesaplar.
    """

    accuracy = accuracy_score(y_test, predictions)

    # zero_division=0, bazı durumlarda hiç pozitif tahmin olmazsa hata vermemesi için kullanılır.
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    cm = confusion_matrix(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report
    }

    return metrics


def save_evaluation_results(
    metrics: dict,
    threshold: float,
    reconstruction_errors: np.ndarray
):
    """
    Hesaplanan metrikleri txt dosyasına kaydeder.
    """

    os.makedirs("results/metrics", exist_ok=True)

    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as file:
        file.write("Autoencoder Anomaly Detection Evaluation Results\n")
        file.write("=" * 55 + "\n\n")

        file.write(f"Threshold: {threshold:.8f}\n")
        file.write(f"Ortalama Reconstruction Error: {np.mean(reconstruction_errors):.8f}\n")
        file.write(f"Minimum Reconstruction Error: {np.min(reconstruction_errors):.8f}\n")
        file.write(f"Maximum Reconstruction Error: {np.max(reconstruction_errors):.8f}\n\n")

        file.write(f"Accuracy:  {metrics['accuracy']:.6f}\n")
        file.write(f"Precision: {metrics['precision']:.6f}\n")
        file.write(f"Recall:    {metrics['recall']:.6f}\n")
        file.write(f"F1-score:  {metrics['f1_score']:.6f}\n\n")

        file.write("Confusion Matrix:\n")
        file.write(str(metrics["confusion_matrix"]))
        file.write("\n\n")

        file.write("Classification Report:\n")
        file.write(metrics["classification_report"])

    print(f"Değerlendirme sonuçları kaydedildi: {METRICS_SAVE_PATH}")


def plot_reconstruction_error_distribution(
    reconstruction_errors: np.ndarray,
    y_test,
    threshold: float
):
    """
    Reconstruction error dağılımını grafik olarak kaydeder.

    Normal ve anomali örneklerinin hata dağılımını ayrı gösterir.
    """

    os.makedirs("results/figures", exist_ok=True)

    y_test_array = np.array(y_test)

    normal_errors = reconstruction_errors[y_test_array == 0]
    anomaly_errors = reconstruction_errors[y_test_array == 1]

    plt.figure(figsize=(10, 6))

    plt.hist(
        normal_errors,
        bins=50,
        alpha=0.7,
        label="Normal"
    )

    plt.hist(
        anomaly_errors,
        bins=50,
        alpha=0.7,
        label="Anomaly"
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label="Threshold"
    )

    plt.xlabel("Reconstruction Error")
    plt.ylabel("Sample Count")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.grid(True)

    plt.savefig(ERROR_DISTRIBUTION_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Reconstruction error grafiği kaydedildi: {ERROR_DISTRIBUTION_PATH}")


def plot_confusion_matrix(cm: np.ndarray):
    """
    Confusion matrix grafiğini kaydeder.
    """

    os.makedirs("results/figures", exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    class_names = ["Normal", "Anomaly"]

    plt.xticks(np.arange(len(class_names)), class_names)
    plt.yticks(np.arange(len(class_names)), class_names)

    # Hücrelerin içine değerleri yazdırıyoruz.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar()
    plt.tight_layout()

    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix grafiği kaydedildi: {CONFUSION_MATRIX_PATH}")


def main():
    """
    Anomali tespiti ve model değerlendirme sürecini başlatır.
    """

    # GPU varsa CUDA, yoksa CPU kullanılır.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    # Test verilerini yükle
    X_test, y_test = load_test_data()

    # Modeli ve threshold değerini yükle
    model = load_trained_model(device)
    threshold = load_threshold()

    print(f"Threshold değeri: {threshold:.8f}")

    # Reconstruction error hesapla
    reconstruction_errors = calculate_reconstruction_errors(
        model=model,
        X_test=X_test,
        device=device
    )

    # Threshold ile karşılaştırıp prediction üret
    predictions = generate_predictions(
        reconstruction_errors=reconstruction_errors,
        threshold=threshold
    )

    # Model performansını değerlendir
    metrics = evaluate_model(
        y_test=y_test,
        predictions=predictions
    )

    # Sonuçları terminale yazdır
    print("\nModel Değerlendirme Sonuçları")
    print("-" * 35)
    print(f"Accuracy:  {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall:    {metrics['recall']:.6f}")
    print(f"F1-score:  {metrics['f1_score']:.6f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    # Sonuçları dosyaya kaydet
    save_evaluation_results(
        metrics=metrics,
        threshold=threshold,
        reconstruction_errors=reconstruction_errors
    )

    # Grafikleri kaydet
    plot_reconstruction_error_distribution(
        reconstruction_errors=reconstruction_errors,
        y_test=y_test,
        threshold=threshold
    )

    plot_confusion_matrix(metrics["confusion_matrix"])

    print("\nAnomali tespiti tamamlandı.")


if __name__ == "__main__":
    main()