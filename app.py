import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import os
import shap

# Sayfa yapılandırması
st.set_page_config(page_title="Telefon Satış Tahmini", page_icon="📱", layout="wide")

# Ana başlık
st.title("📱 Telefon Satış Tahmin Uygulaması")

# Sidebar oluşturma
st.sidebar.title("Navigasyon")

# CSS stil tanımlaması
st.markdown("""
<style>
    .nav-button {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0px;
        text-align: center;
        transition: all 0.3s;
    }
    .nav-button:hover {
        background-color: #4e89ae;
        color: white;
        cursor: pointer;
    }
    .nav-button-active {
        background-color: #4e89ae;
        color: white;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0px;
        text-align: center;
    }
    .nav-icon {
        font-size: 20px;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Navigasyon seçenekleri ve ikonları
nav_options = {
    "Ana Sayfa": "🏠",
    "Veri Analizi": "📊",
    "Model Eğitimi": "🧠",
    "Tahmin Yap": "🔮"
}

# Seçenek değişkeni için varsayılan değer
if 'secenek' not in st.session_state:
    st.session_state.secenek = "Ana Sayfa"

# Navigasyon butonları
for option, icon in nav_options.items():
    if st.sidebar.button(
        f"{icon} {option}", 
        key=f"nav_{option}",
        help=f"{option} sayfasına git",
        use_container_width=True,
        type="primary" if st.session_state.secenek == option else "secondary"
    ):
        st.session_state.secenek = option

# Seçilen sayfayı kullan
secenek = st.session_state.secenek

# Aktif sayfa bilgisi
st.sidebar.markdown(f"<div style='text-align: center; margin-top: 20px;'>Aktif Sayfa: <b>{nav_options[secenek]} {secenek}</b></div>", unsafe_allow_html=True)

# Veri yükleme fonksiyonu
@st.cache_data
def veri_yukle():
    try:
        df = pd.read_csv("Sales_birlesik.csv")
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return None

# Veri ön işleme fonksiyonu
def veri_on_isleme(df):
    # Kopya oluştur
    df_temiz = df.copy()
    
    # Eksik değerleri kontrol et
    eksik_degerler = pd.DataFrame({
        'Eksik Değer Sayısı': df_temiz.isnull().sum(),
        'Eksik Değer Oranı (%)': (df_temiz.isnull().sum() / len(df_temiz) * 100).round(2)
    })
    
    # Sayısal sütunların doğru formatta olduğundan emin olma
    sayisal_sutunlar = ['Satis_Fiyati', 'Orijinal_Fiyat', 'Indirim', 'Indirim_Yuzdesi', 'Satis_Sayisi', 'Populerlik_Endeksi', 'Cikis_Tarihi']
    for sutun in sayisal_sutunlar:
        if sutun in df_temiz.columns:
            df_temiz[sutun] = pd.to_numeric(df_temiz[sutun], errors='coerce')
    
    return df_temiz, eksik_degerler

# Model eğitim fonksiyonu
def model_egit(df, ozellikler, hedef, test_boyutu=0.2, random_state=42):
    # Özellikler ve hedef değişkeni ayır
    X = df[ozellikler]
    y = df[hedef]
    
    # Eğitim ve test verisine ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_boyutu, random_state=random_state, stratify=y
    )
    
    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Lojistik regresyon modeli
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # Tahmin yap
    y_pred = model.predict(X_test_scaled)
    
    # Metrikleri hesapla
    metrikler = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Sınıflandırma raporu
    rapor = classification_report(y_test, y_pred, output_dict=True)
    
    return model, scaler, metrikler, cm, rapor, X_train, X_test, y_train, y_test, y_pred

# Tahmin fonksiyonu
def tahmin_yap(model, scaler, ozellikler, is_scaled=False):
    # Eğer değerler zaten ölçeklendirilmişse, doğrudan kullan
    if is_scaled:
        ozellikler_scaled = [ozellikler]
    else:
        # Özellikleri ölçeklendir
        ozellikler_scaled = scaler.transform([ozellikler])
    
    # Tahmin yap
    # Pipeline kullanıldığında doğrudan pipeline üzerinden tahmin yapılır
    tahmin = model.predict(ozellikler_scaled)
    tahmin_olasilik = model.predict_proba(ozellikler_scaled)[0][1]
    
    return tahmin[0], tahmin_olasilik

# Ana Sayfa
if secenek == "Ana Sayfa":
    st.write("""
    ## Telefon Satış Tahmin Uygulamasına Hoş Geldiniz!
    
    Bu uygulama, telefon özelliklerine göre satış tahminleri yapmak için geliştirilmiştir.
    
    ### Kullanım:
    1. **Veri Analizi**: Veri setini inceleyebilir ve görselleştirebilirsiniz.
    2. **Model Eğitimi**: Makine öğrenmesi modelini eğitebilirsiniz.
    3. **Tahmin Yap**: Eğitilmiş model ile yeni telefonların satılıp satılmayacağını tahmin edebilirsiniz.
    
    Başlamak için soldaki menüden bir seçenek seçin.
    """)
    
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659898.png", width=300)

# Veri Analizi
elif secenek == "Veri Analizi":
    st.header("Veri Analizi")
    
    # Veriyi yükle
    df = veri_yukle()
    if df is not None:
        # Veri ön işleme
        df_temiz, eksik_degerler = veri_on_isleme(df)
        
        # Veri seti hakkında bilgi
        st.subheader("Veri Seti Bilgisi")
        st.write(f"Toplam {df_temiz.shape[0]} satır, {df_temiz.shape[1]} sütun")
        
        # Veri setini göster
        with st.expander("Veri Setini Göster"):
            st.dataframe(df_temiz.head(100))
        
        # Eksik değerleri göster
        with st.expander("Eksik Değerler"):
            st.dataframe(eksik_degerler)
        
        # Veri tipi bilgisi
        with st.expander("Veri Tipi Bilgisi"):
            buffer = pd.DataFrame(df_temiz.dtypes, columns=['Veri Tipi'])
            st.dataframe(buffer)
        
        # İstatistiksel özet
        with st.expander("İstatistiksel Özet"):
            st.dataframe(df_temiz.describe())
        
        # Görselleştirmeler
        st.subheader("Veri Görselleştirme")
        
        # Görselleştirme seçenekleri
        viz_option = st.selectbox(
            "Görselleştirme seçin:",
            ["Satış Durumu Dağılımı", "Marka Dağılımı", "Fiyat Dağılımı", "İndirim Oranı Dağılımı", "Korelasyon Matrisi"]
        )
        
        if viz_option == "Satış Durumu Dağılımı":
            fig, ax = plt.subplots(figsize=(10, 6))
            satildi_sayisi = df_temiz['Satildi'].value_counts()
            ax.pie(satildi_sayisi, labels=["Satılmadı", "Satıldı"], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
            ax.axis('equal')  # Daire şeklinde olması için
            st.pyplot(fig)
            
            # Sayısal değerleri göster
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Satılan Telefon Sayısı", satildi_sayisi.get(1, 0))
            with col2:
                st.metric("Satılmayan Telefon Sayısı", satildi_sayisi.get(0, 0))
        
        elif viz_option == "Marka Dağılımı":
            fig, ax = plt.subplots(figsize=(12, 8))
            marka_sayisi = df_temiz['Marka'].value_counts().head(10)
            sns.barplot(x=marka_sayisi.index, y=marka_sayisi.values, ax=ax)
            plt.xticks(rotation=45)
            plt.title("En Çok Bulunan 10 Marka")
            plt.xlabel("Marka")
            plt.ylabel("Sayı")
            st.pyplot(fig)
        
        elif viz_option == "Fiyat Dağılımı":
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(df_temiz['Satis_Fiyati'], bins=30, kde=True, ax=ax)
            plt.title("Satış Fiyatı Dağılımı")
            plt.xlabel("Satış Fiyatı")
            plt.ylabel("Ürün Sayısı")
            st.pyplot(fig)
            
            # İstatistikler
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ortalama Fiyat", f"{df_temiz['Satis_Fiyati'].mean():.2f}")
            with col2:
                st.metric("Minimum Fiyat", df_temiz['Satis_Fiyati'].min())
            with col3:
                st.metric("Maksimum Fiyat", df_temiz['Satis_Fiyati'].max())
        
        elif viz_option == "İndirim Oranı Dağılımı":
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(df_temiz['Indirim_Yuzdesi'], bins=30, kde=True, ax=ax)
            plt.title("İndirim Oranı Dağılımı")
            plt.xlabel("İndirim Oranı (%)")
            plt.ylabel("Ürün Sayısı")
            st.pyplot(fig)
            
            # İstatistikler
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ortalama İndirim", f"{df_temiz['Indirim_Yuzdesi'].mean():.2f}%")
            with col2:
                st.metric("Minimum İndirim", f"{df_temiz['Indirim_Yuzdesi'].min():.2f}%")
            with col3:
                st.metric("Maksimum İndirim", f"{df_temiz['Indirim_Yuzdesi'].max():.2f}%")
        
        elif viz_option == "Korelasyon Matrisi":
            # Sayısal sütunları seç
            sayisal_df = df_temiz.select_dtypes(include=['float64', 'int64'])
            
            # Korelasyon matrisi
            corr_matrix = sayisal_df.corr()
            
            # Görselleştir
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            plt.title("Korelasyon Matrisi")
            st.pyplot(fig)
            
            # Satildi ile en yüksek korelasyona sahip özellikler
            if 'Satildi' in corr_matrix.columns:
                st.subheader("'Satildi' ile En Yüksek Korelasyona Sahip Özellikler")
                satildi_corr = corr_matrix['Satildi'].drop('Satildi').abs().sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=satildi_corr.values, y=satildi_corr.index, ax=ax)
                plt.title("'Satildi' ile Korelasyon")
                plt.xlabel("Korelasyon Katsayısı (Mutlak Değer)")
                st.pyplot(fig)

# Model Eğitimi
elif secenek == "Model Eğitimi":
    st.header("Model Eğitimi")
    
    # Veriyi yükle
    df = veri_yukle()
    if df is not None:
        # Veri ön işleme
        df_temiz, _ = veri_on_isleme(df)
       
        with st.spinner("Model eğitimi hazırlanıyor..."):
            # Özellikler ve hedef değişkeni ayırma
            # Ölçeklendirilmiş sütunları kullanacağız
            olcekli_sutunlar = [col for col in df_temiz.columns if col.endswith('_Scaled')]
            
            # Kategorik sütunları da ekleyelim
            kategorik_sutunlar = [col for col in df_temiz.columns if col.endswith('_Encoded')]
            if not kategorik_sutunlar:
                st.info("Kodlanmış kategorik sütunlar bulunamadı. Kategorik sütunlar için LabelEncoder uygulanacak.")
                
                # Kategorik sütunları kodla
                for sutun in ['Marka', 'Model', 'Renk']:
                    if sutun in df_temiz.columns:
                        le = LabelEncoder()
                        df_temiz[sutun + '_Encoded'] = le.fit_transform(df_temiz[sutun].astype(str))
                        kategorik_sutunlar.append(sutun + '_Encoded')
            
            # Önemli sayısal sütunlar
            sayisal_sutunlar = ['Bellek', 'Depolama', 'Puan', 'Satis_Fiyati', 'Orijinal_Fiyat', 'Indirim', 
                               'Indirim_Yuzdesi', 'Satis_Sayisi', 'Populerlik_Endeksi']
            
            # Mevcut sayısal sütunları kontrol et
            mevcut_sayisal_sutunlar = [col for col in sayisal_sutunlar if col in df_temiz.columns]
            
            # Tüm özellikleri birleştir
            tum_ozellikler = olcekli_sutunlar + kategorik_sutunlar + mevcut_sayisal_sutunlar
            
            # Kullanıcıya özellik seçme imkanı ver
            secilen_ozellikler = st.multiselect(
                "Kullanılacak özellikleri seçin:",
                options=tum_ozellikler,
                default=tum_ozellikler[:min(8, len(tum_ozellikler))]  # En fazla 8 özellik varsayılan olarak seç
            )
        
        if len(secilen_ozellikler) > 0:
            # Model eğitimi başlat
            if st.button("Modeli Eğit"):
                with st.spinner("Model eğitiliyor..."):
                    # Özellikler ve hedef değişkeni ayır
                    X = df_temiz[secilen_ozellikler]
                    y = df_temiz['Satildi']
                    
                    # Eğitim ve test setlerine ayır
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    st.text(f'Eğitim seti boyutu: {X_train.shape}')
                    st.text(f'Test seti boyutu: {X_test.shape}')
                    
                    # Önceden belirlenmiş en iyi parametreler
                    best_params = {
                        'classifier__class_weight': 'balanced_subsample',
                        'classifier__max_depth': 20,
                        'classifier__min_samples_leaf': 1,
                        'classifier__min_samples_split': 2,
                        'classifier__n_estimators': 100
                    }
                    
                    st.text('Önceden belirlenmiş en iyi parametreler kullanılıyor:')
                    st.text(f'En iyi parametreler: {best_params}')
                    
                    # SMOTE ile dengeleme ve model eğitimi için pipeline oluşturma
                    # Önceden belirlenmiş parametrelerle doğrudan modeli oluşturuyoruz
                    best_model = ImbPipeline([
                        ('smote', SMOTE(random_state=42)),
                        ('classifier', RandomForestClassifier(
                            n_estimators=best_params['classifier__n_estimators'],
                            max_depth=best_params['classifier__max_depth'],
                            min_samples_split=best_params['classifier__min_samples_split'],
                            min_samples_leaf=best_params['classifier__min_samples_leaf'],
                            class_weight=best_params['classifier__class_weight'],
                            random_state=42
                        ))
                    ])
                    
                    # Modeli eğit
                    best_model.fit(X_train, y_train)
                    y_pred = best_model.predict(X_test)
                    
                    # Metrikleri hesapla
                    metrikler = {
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred),
                        "Recall": recall_score(y_test, y_pred),
                        "F1": f1_score(y_test, y_pred)
                    }
                    
                    # StandardScaler oluştur ve verileri ölçeklendir
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Modeli kaydet
                    if not os.path.exists('model'):
                        os.makedirs('model')
                    joblib.dump(best_model, 'model/telefon_satis_model.pkl')
                    joblib.dump(scaler, 'model/telefon_satis_scaler.pkl')
                    joblib.dump(secilen_ozellikler, 'model/telefon_satis_ozellikler.pkl')
                    
                    st.success("Model başarıyla eğitildi ve kaydedildi!")
                    
                    # Metrikleri göster
                    st.subheader("Model Performansı")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{metrikler['Accuracy']:.4f}")
                    col2.metric("Precision", f"{metrikler['Precision']:.4f}")
                    col3.metric("Recall", f"{metrikler['Recall']:.4f}")
                    col4.metric("F1 Score", f"{metrikler['F1']:.4f}")
                    
                    # Sınıflandırma raporu
                    st.subheader("Sınıflandırma Raporu")
                    rapor = classification_report(y_test, y_pred, output_dict=True)
                    rapor_df = pd.DataFrame(rapor).transpose()
                    st.dataframe(rapor_df)
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    plt.xlabel("Tahmin Edilen")
                    plt.ylabel("Gerçek")
                    plt.title("Confusion Matrix")
                    st.pyplot(fig)
                    

# Tahmin Yap
elif secenek == "Tahmin Yap":
    st.header("Tahmin Yap")
    
    # Model dosyalarını kontrol et
    model_path = 'model/telefon_satis_model.pkl'
    scaler_path = 'model/telefon_satis_scaler.pkl'
    ozellikler_path = 'model/telefon_satis_ozellikler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(ozellikler_path):
        # Model ve ilgili dosyaları yükle
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        secilen_ozellikler = joblib.load(ozellikler_path)
        
        st.subheader("Telefon Özelliklerini Girin")
        
        # Veriyi yükle (özellik aralıklarını belirlemek için)
        df = veri_yukle()
        if df is not None:
            # Veri ön işleme
            df_temiz, _ = veri_on_isleme(df)
            
            # Kullanıcı girişleri için sütunlar oluştur
            ozellik_degerleri = []
            
            # Her özellik için giriş alanı oluştur
            col1, col2 = st.columns(2)
            
            for i, ozellik in enumerate(secilen_ozellikler):
                # Ölçeklendirilmiş özellikler için kullanıcının belirttiği aralıkları kullan
                if ozellik.endswith('_Scaled'):
                    # Kullanıcının belirttiği değer aralıklarını kullan
                    if ozellik == 'Satis_Fiyati_Scaled':
                        min_deger = 0.00
                        max_deger = 6.00
                    elif ozellik == 'Satis_Sayisi_Scaled':
                        min_deger = 0.00
                        max_deger = 1.00
                    elif ozellik == 'Bellek_Scaled':
                        min_deger = 0.00
                        max_deger = 10.00
                    elif ozellik == 'Marka_Encoded':
                        min_deger = 0.00
                        max_deger = 16.00
                    elif ozellik == 'Orijinal_Fiyat_Scaled':
                        min_deger = 0.00
                        max_deger = 5.00
                    elif ozellik == 'Populerlik_Endeksi_Scaled':
                        min_deger = 0.00
                        max_deger = 1.00
                    elif ozellik == 'Depolama_Scaled':
                        min_deger = 0.00
                        max_deger = 256.00
                    elif ozellik == 'Model_Encoded':
                        min_deger = 0.00
                        max_deger = 16.00
                    else:
                        min_deger = -3.0  # Standart normal dağılım için tipik minimum değer
                        max_deger = 3.0   # Standart normal dağılım için tipik maksimum değer
                    
                    ortalama = (min_deger + max_deger) / 2  # Ortalama değeri aralığın ortası olarak ayarla
                    adim = (max_deger - min_deger) / 100     # Adım boyutunu aralığa göre ayarla
                else:
                    # Normal özellikler için veri setinden değerleri al
                    min_deger = float(df_temiz[ozellik].min())
                    max_deger = float(df_temiz[ozellik].max())
                    ortalama = float(df_temiz[ozellik].mean())
                    adim = (max_deger - min_deger) / 100
                
                # Kullanıcı dostu etiketler oluştur
                etiket = ozellik
                if ozellik == 'Satis_Fiyati_Scaled':
                    etiket = 'Satış Fiyatı'
                elif ozellik == 'Satis_Sayisi_Scaled':
                    etiket = 'Satış Sayısı'
                elif ozellik == 'Bellek_Scaled':
                    etiket = 'RAM Bellek (GB)'
                elif ozellik == 'Marka_Encoded':
                    etiket = 'Marka'
                elif ozellik == 'Orijinal_Fiyat_Scaled':
                    etiket = 'Orijinal Fiyat'
                elif ozellik == 'Populerlik_Endeksi_Scaled':
                    etiket = 'Popülerlik Endeksi'
                elif ozellik == 'Depolama_Scaled':
                    etiket = 'Depolama Alanı (GB)'
                elif ozellik == 'Model_Encoded':
                    etiket = 'Model'
                
                # Giriş alanı oluştur
                if i % 2 == 0:
                    with col1:
                        deger = st.slider(
                            f"{etiket}:",
                            min_deger,
                            max_deger,
                            ortalama,
                            adim
                        )
                        ozellik_degerleri.append(deger)
                else:
                    with col2:
                        deger = st.slider(
                            f"{etiket}:",
                            min_deger,
                            max_deger,
                            ortalama,
                            adim
                        )
                        ozellik_degerleri.append(deger)
            
            # Tahmin yap
            if st.button("Tahmin Yap"):
                with st.spinner("Tahmin yapılıyor..."):
                    # Ölçeklendirilmiş değerler için is_scaled=True parametresi ile tahmin yap
                    # Eğer özellik adı _Scaled ile bitiyorsa, değerler zaten ölçeklendirilmiştir
                    is_scaled = any(ozellik.endswith('_Scaled') for ozellik in secilen_ozellikler)
                    tahmin, tahmin_olasilik = tahmin_yap(model, scaler, ozellik_degerleri, is_scaled=is_scaled)
                    
                    # Sonuçları göster
                    st.subheader("Tahmin Sonucu")
                    
                    # Görsel gösterge
                    if tahmin == 1:
                        st.success("Bu telefon satılabilir! 📱✅")
                        st.balloons()
                    else:
                        st.error("Bu telefon satılmayabilir! 📱❌")
                    
                    # Olasılık göster
                    st.write(f"Satılma olasılığı: {tahmin_olasilik:.2%}")
                    
                    # Olasılık göstergesi
                    st.progress(tahmin_olasilik)
                    
                    # Girilen değerleri kullanıcı dostu etiketlerle göster
                    st.subheader("Girilen Değerler")
                    
                    # Kullanıcı dostu etiketleri oluştur
                    kullanici_dostu_etiketler = []
                    for oz in secilen_ozellikler:
                        if oz == 'Satis_Fiyati_Scaled':
                            kullanici_dostu_etiketler.append('Satış Fiyatı')
                        elif oz == 'Satis_Sayisi_Scaled':
                            kullanici_dostu_etiketler.append('Satış Sayısı')
                        elif oz == 'Bellek_Scaled':
                            kullanici_dostu_etiketler.append('RAM Bellek (GB)')
                        elif oz == 'Marka_Encoded':
                            kullanici_dostu_etiketler.append('Marka')
                        elif oz == 'Orijinal_Fiyat_Scaled':
                            kullanici_dostu_etiketler.append('Orijinal Fiyat')
                        elif oz == 'Populerlik_Endeksi_Scaled':
                            kullanici_dostu_etiketler.append('Popülerlik Endeksi')
                        elif oz == 'Depolama_Scaled':
                            kullanici_dostu_etiketler.append('Depolama Alanı (GB)')
                        elif oz == 'Model_Encoded':
                            kullanici_dostu_etiketler.append('Model')
                        else:
                            kullanici_dostu_etiketler.append(oz)
                    
                    girdi_df = pd.DataFrame([ozellik_degerleri], columns=kullanici_dostu_etiketler)
                    st.dataframe(girdi_df)
        else:
            st.error("Veri yüklenemedi. Lütfen veri dosyasını kontrol edin.")
    else:
        st.warning("Henüz eğitilmiş bir model bulunamadı. Lütfen önce 'Model Eğitimi' sekmesinden bir model eğitin.")