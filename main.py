import subprocess
import sys


def run_command(command):
    """
    Verilen komutu çalıştırır.
    Eğer hata oluşursa programı durdurur.
    """

    print("\n" + "=" * 60)
    print(f"Çalıştırılıyor: {command}")
    print("=" * 60)

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"Hata oluştu: {command}")
        sys.exit(result.returncode)


def main():
    """
    Projenin tüm akışını sırasıyla çalıştırır.

    1. Veri ön işleme
    2. Model eğitimi
    3. Anomali tespiti ve değerlendirme
    """

    print("Autoencoder Anomaly Detection Pipeline Başlatıldı")

    run_command("python src/data_preprocessing.py")
    run_command("python src/train.py")
    run_command("python src/detect_anomaly.py")

    print("\nTüm süreç başarıyla tamamlandı.")


if __name__ == "__main__":
    main()