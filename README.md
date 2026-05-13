# Autoencoder
# 🕵️‍♂️ Deep Anomaly Detection Using Autoencoders

Bu proje, **PyTorch** kullanılarak geliştirilecek bir **Autoencoder tabanlı anomali tespiti** uygulamasıdır. Projenin temel amacı, yapay sinir ağlarının yalnızca **normal verilerden öğrenerek** normal dışı davranışları, hataları veya saldırı benzeri durumları nasıl tespit edebileceğini göstermektir.

Autoencoder modeli, klasik sınıflandırma modellerinden farklı olarak girdiyi doğrudan `normal` veya `anomali` olarak sınıflandırmaz. Bunun yerine, verilen girdiyi yeniden oluşturmaya çalışır. Model sadece normal verilerle eğitildiği için normal verileri düşük hata ile yeniden oluşturabilir. Ancak anormal veriler modele verildiğinde model bu verileri başarılı şekilde yeniden oluşturamaz ve **reconstruction error**, yani yeniden inşa hatası yükselir.

Bu hata belirlenen eşik değerinin üzerine çıkarsa ilgili veri **anomali** olarak kabul edilir.

---

## 📌 Proje Durumu

Bu proje henüz geliştirme aşamasındadır. README dosyası, proje geliştirilmeden önce sistemin amacını, kullanılacak mimariyi, uygulanacak yöntemi ve planlanan dosya yapısını açıklamak için hazırlanmıştır.

Proje tamamlandığında aşağıdaki çıktılar hedeflenmektedir:

- PyTorch ile Autoencoder modeli geliştirme
- Normal veri ile model eğitimi
- Reconstruction error hesaplama
- Threshold belirleme
- Test verisi üzerinde anomali tespiti
- Grafiklerle hata dağılımı analizi
- Model sonuçlarının değerlendirilmesi

---

## 🎯 Projenin Amacı

Bu projenin amacı, Autoencoder mimarisini kullanarak normal veri davranışını öğrenen ve normal dışı örnekleri tespit edebilen bir sistem geliştirmektir.

Proje kapsamında aşağıdaki sorulara cevap aranacaktır:

- Bir Autoencoder modeli normal veri yapısını öğrenebilir mi?
- Model, daha önce görmediği anormal verilerde daha yüksek hata üretir mi?
- Reconstruction error kullanılarak anomali tespiti yapılabilir mi?
- Threshold seçimi sistem başarısını nasıl etkiler?
- Autoencoder modeli siber güvenlik, finans veya endüstriyel verilerde nasıl kullanılabilir?

---

## 🧠 Autoencoder Nedir?

Autoencoder, giriş verisini önce daha küçük boyutlu bir temsile sıkıştıran, daha sonra bu sıkıştırılmış temsilden orijinal veriyi yeniden üretmeye çalışan bir yapay sinir ağı modelidir.

Autoencoder mimarisi genel olarak iki bölümden oluşur:

### 1. Encoder

Encoder, yüksek boyutlu veriyi daha düşük boyutlu bir temsile dönüştürür.

Örneğin:

```txt
Input Data → 64 → 32 → 16 → 8
```
Bu bölümde model, verinin en önemli özelliklerini öğrenmeye çalışır.

### 2. Bottleneck / Latent Space

Bottleneck katmanı, verinin en sıkıştırılmış halidir.

Örneğin:

8 → 4

Bu katman, modelin veriyi daha küçük ve anlamlı bir temsil ile ifade etmeye çalıştığı bölümdür.

### 3. Decoder

Decoder, sıkıştırılmış veriyi tekrar orijinal giriş boyutuna döndürmeye çalışır.

Örneğin:

4 → 8 → 16 → 32 → 64 → Output

Modelin amacı, giriş verisi ile çıkış verisi arasındaki farkı minimuma indirmektir.

---

## 📌 Anomali Tespiti Mantığı

Bu projede Autoencoder modeli sadece normal verilerle eğitilecektir.

Model, normal verilerin yapısını öğrendikten sonra test aşamasında hem normal hem de anormal veriler modele verilecektir.

Eğer model bir veriyi başarılı şekilde yeniden oluşturabiliyorsa bu veri normal kabul edilir.

Eğer model veriyi iyi şekilde yeniden oluşturamıyorsa hata değeri yükselir ve veri anomali olarak değerlendirilir.

Normal Veri
    ↓
Autoencoder
    ↓
Düşük Reconstruction Error
    ↓
Normal
Anormal Veri
    ↓
Autoencoder
    ↓
Yüksek Reconstruction Error
    ↓
Anomali

---

## 🏗️ Planlanan Model Mimarisi

Bu projede PyTorch ile simetrik bir Autoencoder mimarisi kullanılacaktır.

Örnek mimari:

Input Layer
    ↓
Encoder
    ↓
Linear(input_dim, 64)
ReLU
    ↓
Linear(64, 32)
ReLU
    ↓
Linear(32, 16)
ReLU
    ↓
Linear(16, 8)
ReLU
    ↓
Bottleneck
Linear(8, 4)
ReLU
    ↓
Decoder
    ↓
Linear(4, 8)
ReLU
    ↓
Linear(8, 16)
ReLU
    ↓
Linear(16, 32)
ReLU
    ↓
Linear(32, 64)
ReLU
    ↓
Linear(64, input_dim)
    ↓
Output Layer

Burada input_dim, veri setindeki özellik sayısını ifade eder.

Örneğin veri setinde 30 sütun varsa:

input_dim = 30

Modelin çıkış boyutu da giriş boyutuyla aynı olacaktır. Çünkü Autoencoder modelinin amacı girdiyi yeniden üretmektir.


---

## 📐 Kullanılacak Kayıp Fonksiyonu

Modelin eğitiminde Mean Squared Error, yani MSE kullanılacaktır.

MSE, orijinal veri ile modelin yeniden oluşturduğu veri arasındaki farkı ölçer.

MSE = (1 / n) * Σ(xᵢ - x̂ᵢ)²

Burada:

xᵢ: Orijinal giriş verisi
x̂ᵢ: Modelin yeniden oluşturduğu veri
n: Özellik sayısı

MSE değeri düşükse model veriyi başarılı şekilde yeniden oluşturmuştur.

MSE değeri yüksekse veri normal davranıştan farklı olabilir.

---

## 🚨 Threshold Mantığı

Model eğitildikten sonra normal verilerin reconstruction error değerleri hesaplanacaktır.

Daha sonra bir eşik değeri belirlenecektir.

Planlanan threshold yöntemi:

Threshold = Ortalama Reconstruction Error + 3 * Standart Sapma

Anomali kararı şu şekilde verilecektir:

Reconstruction Error > Threshold  → Anomali
Reconstruction Error <= Threshold → Normal

Bu yöntem sayesinde model, normal verilerden öğrendiği hata aralığının dışına çıkan örnekleri anomali olarak işaretleyebilir.

---

## 🚀 Proje Geliştirme Adımları

Proje aşağıdaki adımlar takip edilerek geliştirilecektir.

### 1. Veri Setinin Belirlenmesi

Öncelikle kullanılacak veri seti seçilecektir.

Bu proje için kullanılabilecek veri seti türleri:

Ağ trafiği verisi
Kredi kartı işlem verisi
Sensör verisi
Sistem log verisi
Normal/anormal etiketli örnek veri setleri

Proje başlangıcında örnek olması açısından küçük ve anlaşılır bir veri seti tercih edilebilir.

### 2. Veri Ön İşleme

Model eğitilmeden önce veri seti ön işleme adımlarından geçirilecektir.

Planlanan işlemler:

Eksik değer kontrolü
Sayısal olmayan değerlerin dönüştürülmesi
Normal ve anomali verilerinin ayrılması
Verilerin normalize edilmesi
Eğitim ve test verisinin ayrılması

Özellikle Autoencoder modellerinde veri ölçeklendirme önemlidir. Bu nedenle StandardScaler veya MinMaxScaler kullanılacaktır.

### 3. Sadece Normal Veri ile Eğitim

Autoencoder modeli sadece normal verilerle eğitilecektir.

Bu sayede model, yalnızca normal davranış örüntülerini öğrenmiş olacaktır.

Anomali verileri eğitim sırasında kullanılmayacaktır.

### 4. Reconstruction Error Hesaplama

Model eğitildikten sonra giriş verileri tekrar modele verilecektir.

Modelin ürettiği çıktı ile orijinal giriş arasındaki fark hesaplanacaktır.

Bu fark, her örnek için reconstruction error değerini verecektir.

### 5. Threshold Belirleme

Normal verilerin reconstruction error dağılımı incelenecektir.

Ortalama ve standart sapmaya göre bir threshold değeri belirlenecektir.

### 6. Test Aşaması

Test verileri modele verilecektir.

Her örnek için reconstruction error hesaplanacaktır.

Threshold değerini geçen örnekler anomali olarak işaretlenecektir.

### 7. Sonuçların Değerlendirilmesi

Modelin başarısı aşağıdaki metriklerle değerlendirilecektir:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Reconstruction Error Distribution
🛠️ Kullanılacak Teknolojiler

Bu projede aşağıdaki teknolojiler kullanılacaktır:

Python
PyTorch
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn

---

## 📁 Planlanan Proje Dosya Yapısı

Proje için önerilen klasör yapısı aşağıdaki gibidir:
```bash
deep-anomaly-detection-autoencoder/
│
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   │
│   └── processed/
│       ├── train_data.csv
│       └── test_data.csv
│
├── notebooks/
│   └── autoencoder_experiment.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── detect_anomaly.py
│   └── utils.py
│
├── results/
│   ├── figures/
│   │   ├── training_loss.png
│   │   ├── reconstruction_error.png
│   │   └── confusion_matrix.png
│   │
│   └── metrics/
│       └── evaluation_results.txt
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

📂 Dosyaların Görevleri
```
data/raw/
```
Ham veri setinin tutulacağı klasördür.
```
data/processed/
```
Ön işleme yapılmış veri dosyalarının tutulacağı klasördür.
```
notebooks/
```
Deneysel çalışmaların ve ilk model denemelerinin yapılacağı Jupyter Notebook dosyalarını içerir.
```
src/data_preprocessing.py
```
Veri temizleme, dönüştürme, normalizasyon ve train-test ayrımı işlemlerini içerir.
```
src/model.py
```
PyTorch Autoencoder model mimarisinin tanımlandığı dosyadır.
```
src/train.py
```
Modelin eğitildiği dosyadır.
```
src/detect_anomaly.py
```
Eğitilen model kullanılarak test verisi üzerinde anomali tespiti yapılan dosyadır.
```
src/utils.py
```
Yardımcı fonksiyonların tutulacağı dosyadır.

Örneğin:

Grafik çizdirme
Model kaydetme
Threshold hesaplama
Metrik hesaplama
main.py

Tüm süreci tek bir dosyadan çalıştırmak için kullanılacaktır.

---

## 💻 Planlanan PyTorch Modeli

Aşağıdaki yapı projede kullanılacak Autoencoder modelinin temel halidir.
```bash

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),

            nn.Linear(8, 16),
            nn.ReLU(),

            nn.Linear(16, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
``` 
Modelin son katmanında aktivasyon fonksiyonu kullanılmamıştır. Bunun nedeni, proje kapsamında verilerin büyük ihtimalle StandardScaler ile ölçeklendirilecek olmasıdır.

Eğer veri MinMaxScaler ile 0-1 aralığına çekilirse, son katmanda Sigmoid aktivasyonu da tercih edilebilir.

---

## 🏋️ Planlanan Eğitim Süreci

Modelin eğitiminde şu yapı kullanılacaktır:
```bash
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import Autoencoder


input_dim = X_train.shape[1]

model = Autoencoder(input_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)

epochs = 50

for epoch in range(epochs):
    model.train()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")
```
Bu eğitim sürecinde model, giriş verisini tekrar üretmeye çalışacaktır.

Loss değeri azaldıkça model normal veriyi daha iyi öğrenmiş olacaktır.

---

## 🧪 Threshold Hesaplama

Eğitim tamamlandıktan sonra threshold değeri hesaplanacaktır.
```
import numpy as np
import torch


model.eval()

with torch.no_grad():
    train_outputs = model(X_train_tensor)
    train_errors = torch.mean((X_train_tensor - train_outputs) ** 2, dim=1).numpy()

threshold = np.mean(train_errors) + 3 * np.std(train_errors)

print("Threshold:", threshold)
```
Bu threshold değeri test aşamasında anomali kararını vermek için kullanılacaktır.

---

## 🚨 Anomali Tespiti

Test verisi modele verilecek ve her örnek için reconstruction error hesaplanacaktır.
```bash
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

model.eval()

with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_errors = torch.mean((X_test_tensor - test_outputs) ** 2, dim=1).numpy()

predictions = test_errors > threshold

for i, error in enumerate(test_errors):
    if predictions[i]:
        print(f"Sample {i}: Anomaly detected! Error: {error:.6f}")
    else:
        print(f"Sample {i}: Normal. Error: {error:.6f}")
```

---

## 📊 Beklenen Çıktılar

Proje tamamlandığında aşağıdaki çıktılar alınması hedeflenmektedir:

Eğitim kaybı grafiği
Reconstruction error dağılımı
Threshold değeri
Normal/anomali tahminleri
Confusion matrix
Accuracy, Precision, Recall ve F1-score değerleri

Örnek çıktı:

Threshold: 0.035421

Sample 0: Normal. Error: 0.012391
Sample 1: Normal. Error: 0.018827
Sample 2: Anomaly detected! Error: 0.091442
Sample 3: Normal. Error: 0.014528

---

## 📈 Görselleştirme Planı

Reconstruction error dağılımı grafikle gösterilecektir.

import matplotlib.pyplot as plt

plt.hist(test_errors, bins=50)
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.show()

Bu grafik sayesinde normal ve anormal verilerin hata dağılımları daha net görülebilecektir.

---

## ⚙️ Kurulum

Projeyi çalıştırmak için önce gerekli kütüphaneler kurulmalıdır.

pip install -r requirements.txt
📦 requirements.txt

Proje için kullanılacak temel Python kütüphaneleri:

numpy
pandas
matplotlib
seaborn
scikit-learn
torch
torchvision

---

## ▶️ Çalıştırma Planı

Model eğitimi için:

python src/train.py

Anomali tespiti için:

python src/detect_anomaly.py

Tüm proje akışını çalıştırmak için:

python main.py
🧩 Kullanım Alanları

Autoencoder tabanlı anomali tespiti birçok farklı alanda kullanılabilir.

Siber Güvenlik

Ağ trafiğinde saldırı, izinsiz erişim veya olağan dışı davranışları tespit etmek için kullanılabilir.

Finans

Kredi kartı işlemlerinde dolandırıcılık tespiti için kullanılabilir.

Endüstriyel Sistemler

Sensör verileri analiz edilerek makine arızaları önceden tahmin edilebilir.

Sistem Logları

Sunucu veya uygulama loglarında normal dışı olaylar tespit edilebilir.

Sağlık Verileri

Hasta takip sistemlerinde normal dışı ölçümler tespit edilebilir.

---

## ✅ Projenin Avantajları

Bu yaklaşımın bazı avantajları şunlardır:

Etiketli anomali verisine her zaman ihtiyaç duymaz.
Model sadece normal verilerle eğitilebilir.
Farklı veri türlerine uyarlanabilir.
Karmaşık veri desenlerini öğrenebilir.
Siber güvenlik ve finans gibi kritik alanlarda kullanılabilir.

---

## ⚠️ Projenin Sınırlamaları

Bu yöntemin bazı sınırlamaları da vardır:

Threshold değeri yanlış seçilirse hatalı tahminler yapılabilir.
Eğitim verisinde anomali bulunursa model anomalileri de normal gibi öğrenebilir.
Çok gürültülü verilerde reconstruction error güvenilir olmayabilir.
Modelin karar verme süreci klasik yöntemlere göre daha zor yorumlanabilir.
Veri ölçeklendirme doğru yapılmazsa model başarısı düşebilir.

---

## 🧪 Değerlendirme Metrikleri

Proje tamamlandığında model başarısı aşağıdaki metriklerle değerlendirilecektir.

Accuracy

Tüm tahminler içinde doğru tahmin oranını gösterir.

Precision

Anomali olarak tahmin edilen örneklerin ne kadarının gerçekten anomali olduğunu gösterir.

Recall

Gerçek anomalilerin ne kadarının doğru yakalandığını gösterir.

F1-score

Precision ve Recall değerlerinin dengeli ortalamasıdır.

Confusion Matrix

Modelin normal ve anomali sınıflarındaki doğru ve yanlış tahminlerini tablo halinde gösterir.

---

## 🔮 Geliştirme Planı

Proje ilerleyen aşamalarda şu özelliklerle geliştirilebilir:

Farklı Autoencoder mimarilerinin denenmesi
Denoising Autoencoder kullanılması
LSTM Autoencoder ile zaman serisi anomalisi tespiti
Farklı threshold belirleme yöntemlerinin karşılaştırılması
Gerçek siber güvenlik veri setleriyle test yapılması
Web arayüzü veya basit dashboard eklenmesi

---

## 📌 Sonuç

Bu proje, Autoencoder mimarisinin anomali tespiti için nasıl kullanılabileceğini göstermeyi amaçlamaktadır.

Model, yalnızca normal verilerle eğitilecek ve normal veri yapısını öğrenmeye çalışacaktır. Daha sonra test verileri üzerinde reconstruction error hesaplanarak normal dışı örnekler tespit edilecektir.

PyTorch kullanılarak geliştirilecek bu proje, derin öğrenme tabanlı anomali tespiti mantığını anlamak ve uygulamak için temel bir örnek olacaktır.
