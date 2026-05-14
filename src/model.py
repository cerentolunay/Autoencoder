import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Autoencoder modeli.

    Bu model anomali tespiti için kullanılacaktır.
    Modelin amacı, verilen giriş verisini yeniden oluşturmaktır.

    Eğitim sırasında model sadece normal verilerle eğitilir.
    Test sırasında anormal veriler geldiğinde model bu verileri iyi yeniden oluşturamaz.
    Bu durumda reconstruction error yükselir.
    """

    def __init__(self, input_dim: int):
        """
        Autoencoder modelini oluşturur.

        Parametre:
            input_dim: Veri setindeki feature sayısı.
                       Credit Card Fraud veri setinde Time ve Class çıkarıldıktan sonra
                       genellikle 29 feature kalır.
        """

        super(Autoencoder, self).__init__()

        # Encoder kısmı:
        # Giriş verisini kademeli olarak daha küçük boyuta sıkıştırır.
        # Bu sıkıştırılmış temsil, verinin özünü öğrenmeye çalışır.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            # Bottleneck katmanı
            # Verinin en sıkıştırılmış temsilidir.
            nn.Linear(8, 4),
            nn.ReLU()
        )

        # Decoder kısmı:
        # Bottleneck'ten gelen düşük boyutlu temsili tekrar giriş boyutuna döndürür.
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),

            nn.Linear(8, 16),
            nn.ReLU(),

            nn.Linear(16, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            # Çıkış boyutu input_dim ile aynı olmalıdır.
            # Çünkü Autoencoder giriş verisini yeniden üretmeye çalışır.
            nn.Linear(64, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Modelin ileri yayılım fonksiyonu.

        Parametre:
            x: Modele verilen giriş verisi.

        Dönüş:
            reconstructed: Modelin yeniden oluşturduğu veri.
        """

        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)

        return reconstructed


def test_model():
    """
    Modelin doğru çalışıp çalışmadığını test etmek için basit kontrol fonksiyonu.

    Burada örnek olarak input_dim = 29 kullanıyoruz.
    Çünkü Credit Card Fraud veri setinde Time ve Class çıkarıldıktan sonra
    geriye V1-V28 ve Amount sütunları kalır.
    """

    input_dim = 29
    batch_size = 5

    model = Autoencoder(input_dim=input_dim)

    # 5 satırlık, 29 feature'lı sahte giriş verisi oluşturuyoruz.
    sample_input = torch.randn(batch_size, input_dim)

    # Modelden çıktı alıyoruz.
    output = model(sample_input)

    print("Model başarıyla oluşturuldu.")
    print(f"Giriş boyutu: {sample_input.shape}")
    print(f"Çıkış boyutu: {output.shape}")

    # Autoencoder'da giriş ve çıkış boyutları aynı olmalıdır.
    assert sample_input.shape == output.shape, "Giriş ve çıkış boyutları eşleşmiyor!"

    print("Test başarılı: Giriş ve çıkış boyutları eşleşiyor.")
    print(model)


if __name__ == "__main__":
    test_model()