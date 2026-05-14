import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Ham veri dosyasının yolu
RAW_DATA_PATH = "data/raw/creditcard.csv"

# İşlenmiş verilerin kaydedileceği klasör
PROCESSED_DATA_DIR = "data/processed"

# Eğitilen scaler nesnesinin kaydedileceği dosya yolu
SCALER_PATH = "models/scaler.pkl"


def load_dataset(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Ham kredi kartı dolandırıcılık veri setini okur.

    Beklenen dosya konumu:
        data/raw/creditcard.csv
    """

    # Dosya belirtilen konumda yoksa kullanıcıya anlaşılır hata mesajı verir.
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Veri seti bulunamadı: {file_path}\n"
            "Lütfen Credit Card Fraud Detection veri setini indirip "
            "data/raw/creditcard.csv konumuna yerleştir."
        )

    df = pd.read_csv(file_path)
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Veri setini Autoencoder tabanlı anomali tespiti için hazırlar.

    Yapılan işlemler:
    - Class sütununa göre normal ve fraud verilerini ayırır.
    - Gerekli olmayan sütunları çıkarır.
    - Eğitim verisini sadece normal işlemlerden oluşturur.
    - Test verisini normal + fraud işlemlerden oluşturur.
    - Özellikleri StandardScaler ile ölçeklendirir.
    """

    # Bu veri setinde mutlaka bulunması gereken sütunlar
    required_columns = {"Class", "Amount"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Veri setinde şu sütunlar bulunmalıdır: {required_columns}"
        )

    # Class = 0 → Normal işlem
    # Class = 1 → Dolandırıcılık / anomali işlemi
    normal_data = df[df["Class"] == 0].copy()
    fraud_data = df[df["Class"] == 1].copy()

    print(f"Normal örnek sayısı: {len(normal_data)}")
    print(f"Fraud/anomali örnek sayısı: {len(fraud_data)}")

    # Time sütunu varsa çıkarılır.
    # Bu projede işlem zamanını doğrudan modellemeyeceğiz.
    # Ama istenirse daha sonra zaman tabanlı analiz için tekrar eklenebilir.
    if "Time" in df.columns:
        normal_data = normal_data.drop(columns=["Time"])
        fraud_data = fraud_data.drop(columns=["Time"])

    # Etiketleri ayırıyoruz.
    # Eğitimde y_train kullanılmayacak çünkü Autoencoder gözetimsiz mantıkla çalışacak.
    normal_labels = normal_data["Class"]
    fraud_labels = fraud_data["Class"]

    # Class sütunu hedef/etiket olduğu için feature'lardan çıkarılır.
    normal_features = normal_data.drop(columns=["Class"])
    fraud_features = fraud_data.drop(columns=["Class"])

    # Normal verinin %80'i eğitim, %20'si test için ayrılır.
    # Model sadece normal eğitim verisiyle eğitilecektir.
    X_train_normal, X_test_normal, y_train_normal, y_test_normal = train_test_split(
        normal_features,
        normal_labels,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # Test verisi normal test örnekleri + tüm fraud örneklerinden oluşur.
    # Böylece modelin hem normal hem de anormal veriler üzerindeki davranışı ölçülebilir.
    X_test = pd.concat([X_test_normal, fraud_features], axis=0)
    y_test = pd.concat([y_test_normal, fraud_labels], axis=0)

    # Test verisini karıştırıyoruz.
    # Böylece test sırasında önce tüm normaller, sonra tüm fraudlar gelmez.
    X_test, y_test = shuffle_data(X_test, y_test)

    # Özellikleri ölçeklendirmek için StandardScaler kullanıyoruz.
    # Scaler sadece eğitim verisi üzerinde fit edilir.
    # Bu doğru yaklaşımdır çünkü test verisinden bilgi sızdırmamış oluruz.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test)

    # NumPy array çıktısını tekrar DataFrame'e çeviriyoruz.
    # Böylece sütun isimleri korunur ve debug etmek kolaylaşır.
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train_normal.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    # y_test indexlerini sıfırlıyoruz.
    y_test = y_test.reset_index(drop=True)

    return X_train_scaled, X_test_scaled, y_test, scaler


def shuffle_data(X: pd.DataFrame, y: pd.Series):
    """
    Feature ve label verilerini birlikte karıştırır.

    X ve y ayrı ayrı karıştırılırsa etiketler yanlış örneklerle eşleşebilir.
    Bu yüzden önce birleştirip, sonra birlikte karıştırıyoruz.
    """

    combined = pd.concat(
        [X.reset_index(drop=True), y.reset_index(drop=True)],
        axis=1
    )

    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    X_shuffled = combined.drop(columns=["Class"])
    y_shuffled = combined["Class"]

    return X_shuffled, y_shuffled


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: StandardScaler
):
    """
    İşlenmiş eğitim/test verilerini ve scaler nesnesini kaydeder.
    """

    # Klasörler yoksa otomatik oluşturulur.
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # İşlenmiş veri dosyalarını kaydet
    X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_test.csv"), index=False)

    # Scaler nesnesini kaydet
    # Daha sonra yeni veriler geldiğinde aynı scaler ile dönüştürme yapılmalıdır.
    joblib.dump(scaler, SCALER_PATH)

    print("İşlenmiş veriler başarıyla kaydedildi.")
    print(f"X_train boyutu: {X_train.shape}")
    print(f"X_test boyutu: {X_test.shape}")
    print(f"y_test boyutu: {y_test.shape}")
    print(f"Scaler kaydedildi: {SCALER_PATH}")


def main():
    """
    Veri ön işleme sürecini başlatan ana fonksiyon.
    """

    df = load_dataset()
    X_train, X_test, y_test, scaler = preprocess_data(df)
    save_processed_data(X_train, X_test, y_test, scaler)


if __name__ == "__main__":
    main()