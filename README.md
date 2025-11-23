# NeuralWall

# Ağ Trafiği Sınıflandırma Projesi (PyTorch)

Bu proje, ağ trafiğini (network traffic) kullanarak firewall’ın vereceği
**allow – deny – drop** aksiyonlarını derin öğrenme (MLP – PyTorch) ile tahmin etmeyi amaçlar.

##  Veri Seti
Veri; port, paket, byte, süre gibi trafik özelliklerini içerir.
Hedef sütun: **Action**
- allow
- deny
- drop  
Nadir sınıf **reset-both**, dengesizliği azaltmak için *deny* ile birleştirilmiştir.

##  Model
- Tabular Neural Network (MLP)
- Katmanlar: 64 → 32 → Output  
- BatchNorm + Dropout
- CrossEntropyLoss + class weight
- Train / Val / Test ayrımı

##  Sonuçlar
- **Accuracy:** 0.993  
- **Macro F1:** 0.991  
- Üç sınıfta da F1 ≈ **0.99**

##  Kullanım
1. Veriyi klasöre ekle: `data/mobile_device_usage.csv`
2. Gereksinimleri kur:
   ```bash
   pip install -r requirements.txt
3. Eğitimi çalıştır:
   ```bash
   python src/train.py

##  Amaç
Ağ trafiği özelliklerinden firewall aksiyonunu otomatik sınıflandırmak.




