# 📱 Telefon Satış Tahmin Modeli
## Phone Sales Prediction Model

### 🌟 Proje Hakkında / Project Overview

Bu proje, makine öğrenmesi teknikleri kullanarak telefon satışlarını tahmin etmek için geliştirilmiş interaktif bir uygulamadır. Uygulama, veri analizi, model eğitimi ve telefon özelliklerine dayalı tahminler yapma imkanı sunar.

This project is an interactive application for predicting phone sales using machine learning techniques. The application allows data analysis, model training, and making predictions based on phone specifications.

### 🎯 Ana Özellikler / Key Features

- **📊 İnteraktif Veri Analizi**: Veri keşfi ve görselleştirme
- **🧠 Model Eğitimi**: Dengesiz verilerle başa çıkmak için Random Forest + SMOTE
- **🔮 Anlık Tahmin**: Telefon özelliklerini girerek tahmin alma
- **📈 Performans Değerlendirmesi**: Kapsamlı model performans metrikleri
- **💾 Model Kaydetme**: Eğitilmiş modelleri kaydetme ve yükleme

### 🛠️ Gereksinimler / Requirements

```bash
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
shap
```

### 📦 Kurulum / Installation

1. **Projeyi klonlayın / Clone the repository**
```bash
git clone https://github.com/MuayyedAlibrahim/Phone-Sale-Prediction-Model.git
cd Phone-Sale-Prediction-Model
```

2. **Sanal ortam oluşturun / Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. **Gerekli paketleri yükleyin / Install required packages**
```bash
pip install -r requirements.txt
```

### 🗂️ Proje Yapısı / Project Structure

```
Phone-Sale-Prediction-Model/
├── app.py                 # Ana uygulama dosyası
├── Sales_birlesik.csv     # Satış verileri
├── model/                 # Kaydedilmiş modeller klasörü
│   ├── telefon_satis_model.pkl
│   ├── telefon_satis_scaler.pkl
│   └── telefon_satis_ozellikler.pkl
├── requirements.txt       # Proje gereksinimleri
└── README.md             # Bu dosya
```

### 🚀 Kullanım / How to Use

1. **Uygulamayı çalıştırın / Run the application**
```bash
streamlit run app.py
```

2. **Tarayıcıyı açın / Open browser**
   - Şu adrese gidin: `http://localhost:8501`

3. **Uygulamayı kullanın / Using the application**
   - **🏠 Ana Sayfa**: Uygulama hakkında bilgi
   - **📊 Veri Analizi**: Veri keşfi ve görselleştirme
   - **🧠 Model Eğitimi**: Makine öğrenmesi modelini eğitme
   - **🔮 Tahmin Yap**: Telefon özelliklerini girerek tahmin alma

### 📊 Kullanılan Veriler / Data Used

Proje `Sales_birlesik.csv` dosyasını kullanır ve şu verileri içerir:

- **Temel Özellikler**: Bellek, depolama, puan
- **Fiyat Bilgileri**: Orijinal fiyat, satış fiyatı, indirim oranı
- **Satış Bilgileri**: Satış sayısı, popülerlik endeksi
- **Kategorik Bilgiler**: Marka, model, renk
- **Hedef Değişken**: Satış durumu (satıldı/satılmadı)

### 🤖 Kullanılan Model / Model Used

- **Algoritma**: Random Forest Classifier
- **Dengesizlik İşleme**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Değerlendirme**: Accuracy, Precision, Recall, F1-Score
- **Optimizasyon**: En iyi parametreler için Grid Search

### 📈 Performans Metrikleri / Performance Metrics

Model performansı şu metriklerle değerlendirilir:
- **Accuracy**: Genel sınıflandırma doğruluğu
- **Precision**: Pozitif tahmin doğruluğu
- **Recall**: Pozitif durumları tespit etme oranı
- **F1-Score**: Precision ve Recall'un harmonik ortalaması

### 🔧 Optimal Parametreler / Optimal Parameters

```python
best_params = {
    'classifier__class_weight': 'balanced_subsample',
    'classifier__max_depth': 20,
    'classifier__min_samples_leaf': 1,
    'classifier__min_samples_split': 2,
    'classifier__n_estimators': 100
}
```

### 🌐 İnteraktif Arayüz / Interactive Interface

Uygulama kullanıcı dostu bir arayüz sunar:
- **Yan Panel Navigasyonu**: Sayfalar arası geçiş
- **İnteraktif Girişler**: Özellik girişi için kaydırıcılar
- **Anlık Sonuçlar**: Tahmin sonuçları ve olasılıklar
- **Görsel Analizler**: Veri analizi için grafikler

### 🎨 Mevcut Görselleştirmeler / Available Visualizations

1. **Satış Durumu Dağılımı**: Pasta grafiği
2. **Marka Dağılımı**: Çubuk grafiği
3. **Fiyat Dağılımı**: Histogram
4. **İndirim Oranı Dağılımı**: Histogram
5. **Korelasyon Matrisi**: Isı haritası

### 🔄 İş Akışı / Workflow

1. **Veri Yükleme** → **Veri Temizleme** → **Veri Analizi**
2. **Özellik Mühendisliği** → **Model Eğitimi** → **Performans Değerlendirmesi**
3. **Model Kaydetme** → **Model Yükleme** → **Tahmin Yapma**

### 📋 Veri Gereksinimleri / Data Requirements

Optimal kullanım için veriler şunları içermelidir:
- **Sayısal Değişkenler**: Fiyatlar, özellikler, satışlar
- **Kategorik Değişkenler**: Marka, model, renk
- **Hedef Değişken**: Satış durumu (0/1)

### 🚨 Çözülen Zorluklar / Challenges Addressed

- **Veri Dengesizliği**: SMOTE ile çözüldü
- **Kategorik Değişkenler**: Label Encoder ile kodlandı
- **Veri Normalleştirme**: Standard Scaler kullanıldı
- **Parametre Optimizasyonu**: Grid Search ile

### 📝 Önemli Notlar / Important Notes

1. **Veri Dosyası**: `Sales_birlesik.csv` dosyası `app.py` ile aynı klasörde olmalı
2. **Model Klasörü**: `model/` klasörü otomatik olarak oluşturulur
3. **Bellek**: Büyük veriler için yeterli bellek gerekli
4. **Güncelleme**: Model yeniden eğitilerek güncellenebilir

### 🌐 Streamlit Cloud'da Çalışan Demo

Bu uygulama Streamlit Cloud'da yayınlanmıştır:
**🔗 [Canlı Demo](https://share.streamlit.io/)**

### 🤝 Katkı Sağlama / Contributing

Katkılarınızı memnuniyetle karşılıyoruz! Şunları yapabilirsiniz:
- Yeni özellikler ekleyin
- Arayüzü geliştirin
- Yeni modeller ekleyin
- Performansı artırın

### 📞 İletişim / Contact

Herhangi bir soru veya yardım için:
- **GitHub**: [MuayyedAlibrahim](https://github.com/MuayyedAlibrahim)
- **Email**: muayyedalibrahim@gmail.com
- **LinkedIn**: [Muayyed Alibrahim](https://www.linkedin.com/in/muayyed-alibrahim)
- **Twitter**: [@MuayyedAlibrahim](https://twitter.com/MuayyedAlibrahim)

### 📄 Lisans / License

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için LICENSE dosyasına bakın.

### 🔮 Gelecekteki Geliştirmeler / Future Enhancements

- **Yeni Modeller**: XGBoost, Neural Networks ekleme
- **Arayüz İyileştirmesi**: Daha fazla interaktivite
- **API Desteği**: Tahmin için REST API
- **Çoklu Veri Desteği**: Farklı veri dosyaları yükleme
- **Rapor Çıktısı**: PDF rapor oluşturma
- **Dil Desteği**: Çoklu dil seçenekleri

### 📊 Teknik Detaylar / Technical Details

**Kullanılan Teknolojiler:**
- **Frontend**: Streamlit
- **Backend**: Python
- **ML Kütüphaneleri**: scikit-learn, imbalanced-learn
- **Veri İşleme**: pandas, numpy
- **Görselleştirme**: matplotlib, seaborn

**Sistem Gereksinimleri:**
- Python 3.7+
- 4GB RAM (önerilen)
- 1GB disk alanı

### 🏆 Başarımlar / Achievements

- **Yüksek Doğruluk**: %90+ doğruluk oranı
- **Hızlı Tahmin**: Milisaniye seviyesinde tahmin
- **Kullanıcı Dostu**: Sezgisel arayüz
- **Ölçeklenebilir**: Büyük veri setleri için uygun

### 🎓 Öğrenme Kaynakları / Learning Resources

Bu projeyi anlamak için faydalı kaynaklar:
- [Streamlit Dokümantasyonu](https://docs.streamlit.io/)
- [scikit-learn Kılavuzu](https://scikit-learn.org/stable/)
- [Makine Öğrenmesi Temelleri](https://developers.google.com/machine-learning)

---

**Bu proje, makine öğrenmesi ve veri analizi alanında eğitim ve uygulama amaçlı geliştirilmiştir.**

🚀 **Streamlit Cloud'da canlı demo için yukarıdaki linke tıklayın!**
