# Autoencoder
# 🕵️‍♂️ Deep Anomaly Detection Using Autoencoders

Bu proje, **PyTorch** kullanılarak geliştirilmiş bir **Autoencoder tabanlı anomali tespiti** uygulamasıdır. Projenin amacı, yapay sinir ağlarının yalnızca **normal verilerden öğrenerek** normal dışı davranışları veya dolandırıcılık işlemlerini nasıl tespit edebileceğini göstermektir.

Bu çalışmada **Credit Card Fraud Detection** veri seti kullanılmıştır. Model yalnızca normal kredi kartı işlemleriyle eğitilmiş, test aşamasında ise hem normal hem de fraud işlemler üzerinde değerlendirilmiştir.

Autoencoder modeli, klasik sınıflandırma modellerinden farklı olarak girdiyi doğrudan `normal` veya `anomali` olarak sınıflandırmaz. Bunun yerine, verilen girdiyi yeniden oluşturmaya çalışır. Model sadece normal verilerle eğitildiği için normal verileri düşük hata ile yeniden oluşturabilir. Ancak fraud/anomali verileri modele verildiğinde reconstruction error yükselir.

Bu hata belirlenen threshold değerinin üzerine çıkarsa ilgili işlem **anomali** olarak kabul edilir.

---

## 📌 Proje Durumu

Proje başarıyla tamamlanmış ve uçtan uca çalıştırılmıştır.

Tamamlanan aşamalar:

- Credit Card Fraud Detection veri setinin hazırlanması
- Normal ve fraud işlemlerin ayrılması
- `Time` ve `Class` sütunlarının uygun şekilde çıkarılması
- `Amount` ve diğer sayısal feature değerlerinin ölçeklendirilmesi
- Modelin sadece normal işlemlerle eğitilmesi
- PyTorch ile Autoencoder modelinin oluşturulması
- Reconstruction error hesaplanması
- Farklı threshold değerlerinin denenmesi
- Final threshold değerinin seçilmesi
- Test verisi üzerinde anomali tespiti yapılması
- Confusion matrix, Precision, Recall ve F1-score değerlerinin hesaplanması
- Sonuçların grafik ve metrik dosyaları olarak kaydedilmesi

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

## 🏗️ Model Mimarisi

Bu projede PyTorch ile simetrik bir Autoencoder mimarisi kullanılmıştır.


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

Bu projede `Time` ve `Class` sütunları çıkarıldıktan sonra geriye `V1-V28` ve `Amount` sütunları kalmıştır. Bu nedenle modelin giriş boyutu `29` olmuştur.

Modelin çıkış boyutu da giriş boyutuyla aynı olacaktır. Çünkü Autoencoder modelinin amacı girdiyi yeniden üretmektir.


---

## 📐 Kayıp Fonksiyonu

Modelin eğitiminde Mean Squared Error, yani MSE kullanılmıştır.

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

Model eğitildikten sonra normal eğitim verileri üzerinde reconstruction error değerleri hesaplanmıştır.

Bu projede farklı percentile değerleri denenerek threshold seçiminin model performansına etkisi incelenmiştir.

Denemelerde şu percentile değerleri karşılaştırılmıştır:

- 90
- 95
- 97
- 98
- 99
- 99.5

Daha düşük threshold değerleri daha fazla fraud işlemi yakalasa da çok fazla false positive üretmiştir. Daha yüksek threshold değerleri ise false positive sayısını azaltmış, ancak bazı fraud işlemlerinin kaçırılmasına neden olmuştur.

Final modelde en dengeli sonuç verdiği için **99.5 percentile** threshold değeri seçilmiştir.

Anomali kararı şu şekilde verilmiştir:

```txt
Reconstruction Error > Threshold  → Anomali
Reconstruction Error <= Threshold → Normal
```

---

## 🚀 Proje Geliştirme Adımları

Proje aşağıdaki adımlar takip edilerek geliştirilmiştir.

### 1. Veri Setinin Hazırlanması

Credit Card Fraud Detection veri seti `data/raw/creditcard.csv` konumuna yerleştirilmiştir.

### 2. Veri Ön İşleme

`src/data_preprocessing.py` dosyası ile aşağıdaki işlemler yapılmıştır:

- `creditcard.csv` dosyası okunmuştur.
- `Class` sütununa göre normal ve fraud işlemler ayrılmıştır.
- `Class = 0` olan veriler normal işlem olarak alınmıştır.
- `Class = 1` olan veriler fraud/anomali işlem olarak alınmıştır.
- `Time` sütunu çıkarılmıştır.
- `Class` sütunu model girişinden çıkarılmıştır.
- Feature değerleri `StandardScaler` ile ölçeklendirilmiştir.
- Eğitim verisi sadece normal işlemlerden oluşturulmuştur.
- Test verisi normal + fraud işlemlerden oluşturulmuştur.

### 3. Model Eğitimi

`src/train.py` dosyası ile Autoencoder modeli eğitilmiştir.

Eğitimde:

- Loss fonksiyonu olarak `MSELoss`
- Optimizer olarak `Adam`
- Epoch sayısı olarak `50`
- Batch size olarak `256`

kullanılmıştır.

Model sadece normal işlem verileriyle eğitilmiştir.

### 4. Threshold Belirleme

Eğitim verisi üzerinde reconstruction error değerleri hesaplanmıştır. Daha sonra farklı percentile değerleri denenmiş ve final threshold olarak `99.5 percentile` seçilmiştir.

### 5. Anomali Tespiti

`src/detect_anomaly.py` dosyası ile test verileri üzerinde reconstruction error hesaplanmıştır. Threshold değerini geçen örnekler anomali olarak işaretlenmiştir.

### 6. Sonuçların Değerlendirilmesi

Model başarısı aşağıdaki metriklerle değerlendirilmiştir:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Reconstruction Error Distribution

---

## 📁 Proje Dosya Yapısı

```bash
Autoencoder/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       └── y_test.csv
│
├── models/
│   ├── autoencoder_model.pth
│   ├── threshold.txt
│   └── scaler.pkl
│
├── results/
│   ├── figures/
│   │   ├── training_loss.png
│   │   ├── reconstruction_error_distribution.png
│   │   └── confusion_matrix.png
│   │
│   └── metrics/
│       ├── evaluation_results.txt
│       └── threshold_experiment_results.txt
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── detect_anomaly.py
│   ├── threshold_experiment.py
│   └── utils.py
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

## 📊 Çıktılar

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

## 📈 Görselleştirme

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
torchaudio
joblib

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

## 📊 Deneysel Sonuçlar

Model yalnızca normal kredi kartı işlemleriyle eğitilmiştir. Test aşamasında normal ve fraud işlemler birlikte kullanılmıştır.

Final threshold değeri, eğitim verisinin reconstruction error dağılımındaki `99.5 percentile` değeri olarak seçilmiştir.

| Metrik | Değer |
|---|---:|
| Threshold | 2.96167900 |
| Accuracy | 0.993287 |
| Precision | 0.577649 |
| Recall | 0.808943 |
| F1-score | 0.674005 |

Confusion Matrix:

```txt
[[56572   291]
 [   94   398]]
```

Bu sonuçlara göre model, toplam 492 fraud işlemin 398 tanesini doğru şekilde yakalamıştır. 94 fraud işlemi kaçırılmıştır. Ayrıca 291 normal işlem yanlışlıkla anomali olarak sınıflandırılmıştır.

Bu sonuç, Autoencoder modelinin normal işlem desenlerini öğrenerek fraud işlemleri belirli bir başarıyla ayırt edebildiğini göstermektedir.

---

## 🧪 Threshold Deneyi

Threshold seçiminin model performansı üzerindeki etkisini görmek için farklı percentile değerleri denenmiştir.

| Percentile | Threshold | Accuracy | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|---:|
| 90 | 0.82548052 | 0.900671 | 0.071181 | 0.878049 | 0.131687 |
| 95 | 1.11640763 | 0.948531 | 0.127724 | 0.857724 | 0.222339 |
| 97 | 1.38761914 | 0.968686 | 0.194757 | 0.845528 | 0.316591 |
| 98 | 1.66717970 | 0.978659 | 0.265385 | 0.841463 | 0.403509 |
| 99 | 2.33922410 | 0.988179 | 0.406439 | 0.821138 | 0.543742 |
| 99.5 | 3.52127695 | 0.993305 | 0.581325 | 0.784553 | 0.667820 |

Threshold yükseldikçe model daha seçici davranmıştır. Bunun sonucunda false positive sayısı azalmış ve precision artmıştır. Ancak threshold yükseldikçe bazı fraud işlemleri kaçırıldığı için recall değeri düşmüştür.

Final eğitim çalıştırmasında `99.5 percentile` kullanılmış ve aşağıdaki sonuç elde edilmiştir:

```txt
Threshold: 2.96167900
Accuracy:  0.993287
Precision: 0.577649
Recall:    0.808943
F1-score:  0.674005 
``` 

---

## 📌 Sonuç

Bu proje, Autoencoder mimarisinin anomali tespiti için nasıl kullanılabileceğini göstermektedir.

Model yalnızca normal kredi kartı işlemleriyle eğitilmiş ve normal işlem desenlerini öğrenmiştir. Daha sonra test verileri üzerinde reconstruction error hesaplanarak fraud/anomali işlemler tespit edilmiştir.

Deneysel sonuçlara göre final model:

- `0.993287` accuracy
- `0.577649` precision
- `0.808943` recall
- `0.674005` F1-score

değerlerine ulaşmıştır.

Threshold denemeleri, anomali tespitinde eşik değerinin model performansı üzerinde önemli etkisi olduğunu göstermiştir. Daha düşük threshold değerleri daha fazla fraud yakalarken çok fazla false positive üretmiştir. Daha yüksek threshold değerleri ise false positive sayısını azaltmış ancak bazı fraud işlemlerinin kaçırılmasına neden olmuştur.

Bu nedenle final modelde precision-recall dengesini daha iyi sağladığı için `99.5 percentile` threshold değeri kullanılmıştır.

Sonuç olarak bu proje, PyTorch ile geliştirilen Autoencoder mimarisinin finansal dolandırıcılık tespiti gibi dengesiz veri problemlerinde kullanılabileceğini göstermektedir.
