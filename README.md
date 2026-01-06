# NeuralWall
# 🔐 Firewall Trafik Sınıflandırma Projesi (PyTorch)

Bu proje, **Internet Firewall Data Seti** kullanılarak ağ trafiğinin
makine öğrenmesi ve derin öğrenme yöntemleriyle sınıflandırılmasını
amaçlamaktadır.  
Çalışma, **FSA131 – Mühendisliğin Uygulama İlkeleri** dersi kapsamında
hazırlanmıştır.

---

## 📌 Projenin Amacı

Bu projenin temel amacı:

- Firewall üzerinden geçen ağ trafiğini analiz etmek
- Trafik türlerini (`Action`) sınıflandırmak
- Çok Katmanlı Yapay Sinir Ağı (MLP) kullanarak tahmin yapmak
- Dengesiz veri problemiyle başa çıkmak için **class weighting** uygulamak

---

## 📊 Kullanılan Veri Seti

- **Veri Seti Adı:** Internet Firewall Data
- **Hedef Değişken:** `Action`
- **Özellikler (Features):**
  - Source Port
  - Destination Port
  - NAT Source Port
  - NAT Destination Port
  - Bytes
  - Bytes Sent
  - Bytes Received
  - Packets
  - Elapsed Time (sec)
  - pkts_sent
  - pkts_received

Veri setinde sınıflar dengesiz olduğu için model eğitimi sırasında
bu durum dikkate alınmıştır.

---

## ⚙️ Kullanılan Teknolojiler

- **Python 3**
- **PyTorch**
- **NumPy & Pandas**
- **Scikit-learn**
- **Jupyter Notebook**

---

## 🧠 Kullanılan Yöntemler

### 🔹 Veri Ön İşleme
- Hedef değişken (`Action`) **LabelEncoder** ile sayısal hale getirilmiştir
- Özellikler **StandardScaler** ile ölçeklendirilmiştir
- Veri seti %80 eğitim, %20 test olarak ayrılmıştır (`stratify` kullanılmıştır)

### 🔹 Model
- Çok Katmanlı Algılayıcı (MLP)
- ReLU aktivasyon fonksiyonu
- Batch Normalization
- Dropout (overfitting’i önlemek için)

### 🔹 Eğitim
- Kayıp fonksiyonu: `CrossEntropyLoss`
- **Class Weighting** kullanılarak dengesiz sınıflar dengelenmiştir
- Optimizasyon algoritması: Adam
- Epoch sayısı: 20

---

## 📈 Model Değerlendirme

Model performansı aşağıdaki metrikler ile değerlendirilmiştir:

- Confusion Matrix
- Precision
- Recall
- F1-Score
- Classification Report

`UndefinedMetricWarning` uyarıları, nadir sınıflara ait tahmin
yapılmamasından kaynaklanmaktadır ve `zero_division=0` parametresi ile
kontrol altına alınmıştır.

---

## 🧪 Çalıştırma Adımları

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install numpy pandas scikit-learn torch
