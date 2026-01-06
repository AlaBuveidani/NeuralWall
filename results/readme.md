## 📊 Results (Sonuçlar)

Bu bölümde, geliştirilen Çok Katmanlı Yapay Sinir Ağı (MLP) modelinin
test verisi üzerindeki performans sonuçları sunulmaktadır.

Model, eğitim sürecinden sonra daha önce görülmemiş test verisi
(%20) üzerinde değerlendirilmiştir.

---

### 🔹 Confusion Matrix

Confusion Matrix, modelin her bir sınıf için yaptığı doğru ve yanlış
tahminleri göstermektedir.  

- Diyagonal elemanlar doğru tahminleri
- Diyagonal dışı elemanlar yanlış sınıflandırmaları temsil eder

Bu matris yardımıyla modelin hangi sınıfları daha iyi öğrendiği ve
hangi sınıflarda hata yaptığı analiz edilmiştir.

---

### 🔹 Classification Report

Model performansı aşağıdaki metrikler kullanılarak değerlendirilmiştir:

- **Precision:** Modelin belirli bir sınıf için yaptığı tahminlerin ne kadarının doğru olduğunu gösterir
- **Recall:** Gerçek sınıf örneklerinin ne kadarının doğru tahmin edildiğini gösterir
- **F1-Score:** Precision ve Recall değerlerinin harmonik ortalamasıdır
- **Support:** Her bir sınıfa ait örnek sayısını ifade eder

Dengesiz veri yapısından dolayı bazı sınıflar için precision veya
recall değerlerinin düşük olduğu gözlemlenmiştir.  
Bu durum, nadir sınıflara ait örnek sayısının az olmasından kaynaklanmaktadır.

---

### 🔹 UndefinedMetricWarning Açıklaması

Bazı sınıflar için `UndefinedMetricWarning` uyarısı alınmıştır.
Bu uyarı, modelin ilgili sınıf için test setinde herhangi bir tahmin
yapmaması durumunda ortaya çıkmaktadır.

Bu durum:
- Bir hata değildir
- Veri setindeki sınıf dengesizliğinin doğal bir sonucudur

Uyarı, değerlendirme sırasında `zero_division=0` parametresi kullanılarak
kontrol altına alınmıştır.

---

### 🔹 Genel Değerlendirme

- Model, baskın sınıflar üzerinde başarılı sonuçlar üretmiştir
- Nadir sınıfların performansını artırmak için **class weighting**
  yöntemi uygulanmıştır
- Elde edilen sonuçlar, firewall trafiği sınıflandırma problemi için
  derin öğrenme tabanlı yaklaşımın uygulanabilir olduğunu göstermektedir

---

### 🔹 İyileştirme Önerileri

Model performansını daha da artırmak için aşağıdaki çalışmalar önerilmektedir:

- Daha fazla veri ile modelin yeniden eğitilmesi
- Alternatif modellerin (Logistic Regression, Random Forest vb.) denenmesi
- Hiperparametre optimizasyonu
- Veri dengesizliği için farklı örnekleme tekniklerinin kullanılması
