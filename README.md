# 🧠 ANN Tabanlı Meme Kanseri Teşhisi

Bu proje, **Yapay Sinir Ağları (YSA)** dersi kapsamında geliştirilmiş olup, **Wisconsin Breast Cancer Diagnostic** veri seti kullanılarak meme kanserinin **iyi huylu (Benign)** veya **kötü huylu (Malignant)** olarak sınıflandırılmasını amaçlamaktadır.

Proje; veri analizi, model eğitimi, model karşılaştırması ve kullanıcı etkileşimli bir arayüzü kapsayan **uçtan uca (end-to-end)** bir makine öğrenmesi uygulamasıdır.

---

## 📌 Projenin Amacı

* Meme kanseri teşhisinde yapay sinir ağlarının etkinliğini incelemek
* Sayısal özelliklerden oluşan bir veri seti üzerinde ANN modeli geliştirmek
* ANN modelini klasik makine öğrenmesi algoritmalarıyla karşılaştırmak
* Model sonuçlarını görselleştirmek ve kullanıcı dostu bir arayüz sunmak

---

## 📊 Kullanılan Veri Seti

**Wisconsin Breast Cancer Diagnostic Dataset**

* Toplam örnek sayısı: **569**
* Özellik sayısı: **30** (sayısal)
* Sınıf sayısı: **2**

  * Benign (0)
  * Malignant (1)

Özellikler; hücre çekirdeği ölçümlerine dayalı olarak hesaplanan yarıçap, çevre, alan, doku, simetri gibi istatistiksel değerleri içermektedir.

---

## ⚙️ Kullanılan Teknolojiler

* **Python 3.11**
* **scikit-learn**
* **pandas / numpy**
* **matplotlib**
* **Streamlit**
* **joblib**

---

## 🧠 Model Mimarisi (ANN)

Projede sınıflandırma için **MLPClassifier (Artificial Neural Network)** kullanılmıştır.

* Giriş katmanı: 30 nöron
* Gizli katmanlar:

  * 16 nöron
  * 8 nöron
* Aktivasyon fonksiyonu: ReLU
* Optimizasyon algoritması: Adam
* Maksimum epoch: 300

Model, ölçeklendirilmiş veriler üzerinde eğitilmiştir.

---

## 📈 Model Karşılaştırması

ANN modeli aşağıdaki algoritmalarla karşılaştırılmıştır:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Decision Tree
* Artificial Neural Network (ANN)

Karşılaştırma sonucunda ANN modeli en yüksek doğruluk oranını elde etmiştir.

---

## 📊 Performans Değerlendirmesi

* Accuracy (Doğruluk): ≈ **%98**
* Confusion Matrix analizi yapılmıştır
* Loss grafiği ile eğitim süreci incelenmiştir
* Train vs Test accuracy karşılaştırması ile overfitting analizi yapılmıştır

Modelin eğitim ve test performansları birbirine yakın olup, ezberleme (overfitting) gözlemlenmemiştir.

---

## 🖥️ Kullanıcı Arayüzü (Streamlit)

Proje kapsamında geliştirilen Streamlit arayüzü ile:

* Kullanıcı manuel hasta verisi girebilir
* Test verisinden rastgele örnek seçilebilir
* Model tahmini ve gerçek sonuç birlikte gösterilir
* Birden fazla deneme geçmişi tutulur
* Deneme geçmişi CSV olarak indirilebilir
* Model performans grafikleri ayrı bir sekmede sunulur

---

## 📂 Proje Klasör Yapısı

```
YSA_Meme_Kanseri_Teshisi/
│
├── data/
│   └── data.csv
│
├── outputs/
│   ├── ann_model.pkl
│   ├── X_test.csv
│   ├── y_test.csv
│   ├── ann_confusion_matrix.png
│   ├── figure_loss.png
│   ├── figure_train_test_accuracy.png
│   ├── figure_accuracy_learning_curve.png
│   ├── model_comparison_all_models.png
│   └── results.txt
│
├── ann_model.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## ▶️ Kurulum ve Çalıştırma

### 1️⃣ Gerekli kütüphaneleri yükleyin

```bash
pip install -r requirements.txt
```

### 2️⃣ Modeli eğitin (Notebook)

```bash
jupyter notebook ann_model.ipynb
```

### 3️⃣ Arayüzü çalıştırın

```bash
streamlit run app.py
```

---

## 🎓 Akademik Not

Bu proje, Yapay Sinir Ağları dersinde ANN mimarisinin:

* sayısal veriler üzerinde uygulanmasını,
* genelleme yeteneğini,
* model karşılaştırmasını

göstermek amacıyla hazırlanmıştır. Modelin test verilerinde zaman zaman yanlış tahmin yapması, sistemin ezberleme yapmadığını ve gerçekçi sonuçlar ürettiğini göstermektedir.

---

## 👩‍🎓 Hazırlayan

**Semanur Yıldırım**
Yapay Sinir Ağları Dersi Projesi





