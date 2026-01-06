## 🤖 Models

Bu projede, firewall trafiğini sınıflandırmak amacıyla
**Çok Katmanlı Yapay Sinir Ağı (MLP)** modeli kullanılmıştır.

### Model Mimarisi
- Girdi katmanı: Özellik sayısı kadar nöron
- Gizli katmanlar:
  - 128 nöron (ReLU + Batch Normalization + Dropout)
  - 64 nöron (ReLU + Batch Normalization + Dropout)
- Çıkış katmanı: Sınıf sayısı kadar nöron

### Model Eğitimi
- Kayıp fonksiyonu: CrossEntropyLoss
- Optimizasyon algoritması: Adam
- Epoch sayısı: 20
- Dengesiz veri problemi için **class weighting** uygulanmıştır

### Model Kaydı
Eğitilen model, tekrar kullanılabilmesi için
`models/mlp_firewall_model.pt` dosyasına kaydedilmiştir.

Bu sayede model, yeniden eğitilmeden doğrudan yüklenip
test veya tahmin işlemlerinde kullanılabilmektedir.
