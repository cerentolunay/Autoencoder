import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from model import Autoencoder


# İşlenmiş eğitim verisinin yolu
TRAIN_DATA_PATH = "data/processed/X_train.csv"

# Eğitilen modelin kaydedileceği yol
MODEL_SAVE_PATH = "models/autoencoder_model.pth"

# Threshold değerinin kaydedileceği yol
THRESHOLD_SAVE_PATH = "models/threshold.txt"

# Eğitim loss grafiğinin kaydedileceği yol
LOSS_FIGURE_PATH = "results/figures/training_loss.png"


def load_training_data(file_path: str = TRAIN_DATA_PATH) -> pd.DataFrame:
    """
    Ön işleme aşamasında oluşturulan X_train.csv dosyasını okur.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Eğitim verisi bulunamadı: {file_path}\n"
            "Önce src/data_preprocessing.py dosyasını çalıştırmalısın."
        )

    X_train = pd.read_csv(file_path)
    return X_train


def create_dataloader(X_train: pd.DataFrame, batch_size: int = 256) -> DataLoader:
    """
    Pandas DataFrame formatındaki eğitim verisini PyTorch DataLoader formatına çevirir.
    """

    # DataFrame'i float32 Tensor formatına çeviriyoruz.
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)

    # Autoencoder'da giriş ve hedef aynı veridir.
    # Yani model X'i alır, yine X'i üretmeye çalışır.
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader


def train_model(
    model: Autoencoder,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    learning_rate: float = 0.001
):
    """
    Autoencoder modelini eğitir.

    Model sadece normal işlem verileriyle eğitilir.
    Amaç, normal veriyi yeniden oluşturmayı öğrenmesidir.
    """

    # Reconstruction error için MSELoss kullanıyoruz.
    criterion = nn.MSELoss()

    # Optimizer olarak Adam kullanıyoruz.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_losses = []

    model.to(device)

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0.0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            # Modelden yeniden oluşturulmuş çıktı alınır.
            outputs = model(batch_features)

            # Giriş verisi ile model çıktısı arasındaki fark hesaplanır.
            loss = criterion(outputs, batch_targets)

            # Önce eski gradient değerleri sıfırlanır.
            optimizer.zero_grad()

            # Backpropagation yapılır.
            loss.backward()

            # Model ağırlıkları güncellenir.
            optimizer.step()

            epoch_loss += loss.item()

        # Epoch başına ortalama loss hesaplanır.
        avg_epoch_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_epoch_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_epoch_loss:.6f}")

    return model, training_losses


def calculate_threshold(
    model: Autoencoder,
    X_train: pd.DataFrame,
    device: torch.device,
    percentile: int = 95
) -> float:
    """
    Eğitim verisi üzerinden reconstruction error hesaplar
    ve threshold değerini belirler.

    Burada percentile yöntemi kullanıyoruz.
    Örneğin 95 percentile:
    Normal verilerin reconstruction error değerlerinin %95'i bu değerin altındadır.
    Bu değerin üstüne çıkan örnekler test aşamasında anomali kabul edilebilir.
    """

    model.eval()

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed = model(X_train_tensor)

        # Her satır için ayrı reconstruction error hesaplanır.
        reconstruction_errors = torch.mean(
            (X_train_tensor - reconstructed) ** 2,
            dim=1
        ).cpu().numpy()

    threshold = np.percentile(reconstruction_errors, percentile)

    print(f"Threshold ({percentile}. percentile): {threshold:.6f}")

    return threshold


def save_model(model: Autoencoder, input_dim: int, threshold: float):
    """
    Eğitilen modeli ve threshold değerini kaydeder.
    """

    os.makedirs("models", exist_ok=True)

    # Model state_dict ile kaydedilir.
    # input_dim de kaydediliyor çünkü modeli tekrar yüklerken lazım olacak.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim
        },
        MODEL_SAVE_PATH
    )

    with open(THRESHOLD_SAVE_PATH, "w", encoding="utf-8") as file:
        file.write(str(threshold))

    print(f"Model kaydedildi: {MODEL_SAVE_PATH}")
    print(f"Threshold kaydedildi: {THRESHOLD_SAVE_PATH}")


def save_loss_plot(training_losses):
    """
    Eğitim loss değerlerini grafik olarak kaydeder.
    """

    os.makedirs("results/figures", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(training_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(LOSS_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Training loss grafiği kaydedildi: {LOSS_FIGURE_PATH}")


def main():
    """
    Eğitim sürecini başlatan ana fonksiyon.
    """

    # GPU varsa CUDA, yoksa CPU kullanılır.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    # Eğitim verisini oku
    X_train = load_training_data()

    # Feature sayısı otomatik alınır.
    # Credit Card Fraud veri setinde bu değer genellikle 29 olur.
    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")

    # DataLoader oluştur
    train_loader = create_dataloader(X_train, batch_size=256)

    # Autoencoder modelini oluştur
    model = Autoencoder(input_dim=input_dim)

    # Modeli eğit
    model, training_losses = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=50,
        learning_rate=0.001
    )

    # Threshold hesapla
    threshold = calculate_threshold(
        model=model,
        X_train=X_train,
        device=device,
        percentile=95
    )

    # Modeli ve threshold değerini kaydet
    save_model(
        model=model,
        input_dim=input_dim,
        threshold=threshold
    )

    # Eğitim loss grafiğini kaydet
    save_loss_plot(training_losses)

    print("Eğitim süreci tamamlandı.")


if __name__ == "__main__":
    main() 