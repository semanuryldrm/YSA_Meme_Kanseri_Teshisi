import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random
from datetime import datetime

# --------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------
st.set_page_config(
    page_title="ANN Tabanlı Meme Kanseri Teşhisi",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 ANN Tabanlı Meme Kanseri Teşhisi")

# --------------------------------------------------
# MODEL + SCALER YÜKLE
# --------------------------------------------------
@st.cache_resource
def load_model():
    data = joblib.load("outputs/ann_model.pkl")
    return data["model"], data["scaler"]

model, scaler = load_model()

# --------------------------------------------------
# TEST VERİSİ
# --------------------------------------------------
@st.cache_data
def load_test_data():
    X_test = pd.read_csv("outputs/X_test.csv")
    y_test = pd.read_csv("outputs/y_test.csv")
    return X_test, y_test

X_test_df, y_test_df = load_test_data()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "sample" not in st.session_state:
    st.session_state.sample = None
    st.session_state.sample_idx = None
    st.session_state.is_test_sample = False

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# SEKME YAPISI
# --------------------------------------------------
tab_tahmin, tab_grafikler = st.tabs(
    ["🧪 Tahmin", "📊 Model Performansı"]
)

# ==================================================
# 🧪 TAHMİN SEKME
# ==================================================
with tab_tahmin:

    st.subheader("🔍 Hasta Verisine Göre Tahmin")

    # -------------------------
    # ÖZELLİK GİRİŞLERİ
    # -------------------------
    feature_names = [
        "Radius Mean (Ortalama Yarıçap)",
        "Texture Mean (Ortalama Doku)",
        "Perimeter Mean (Ortalama Çevre)",
        "Area Mean (Ortalama Alan)",
        "Smoothness Mean (Ortalama Düzgünlük)",
        "Compactness Mean (Ortalama Kompaktlık)",
        "Concavity Mean (Ortalama İçbükeylik)",
        "Concave Points Mean (Ortalama İçbükey Nokta Sayısı)",
        "Symmetry Mean (Ortalama Simetri)",
        "Fractal Dimension Mean (Ortalama Fraktal Boyut)",

        "Radius SE (Yarıçap Standart Hatası)",
        "Texture SE (Doku Standart Hatası)",
        "Perimeter SE (Çevre Standart Hatası)",
        "Area SE (Alan Standart Hatası)",
        "Smoothness SE (Düzgünlük Standart Hatası)",
        "Compactness SE (Kompaktlık Standart Hatası)",
        "Concavity SE (İçbükeylik Standart Hatası)",
        "Concave Points SE (İçbükey Nokta Standart Hatası)",
        "Symmetry SE (Simetri Standart Hatası)",
        "Fractal Dimension SE (Fraktal Boyut Standart Hatası)",

        "Radius Worst (En Kötü Yarıçap)",
        "Texture Worst (En Kötü Doku)",
        "Perimeter Worst (En Kötü Çevre)",
        "Area Worst (En Kötü Alan)",
        "Smoothness Worst (En Kötü Düzgünlük)",
        "Compactness Worst (En Kötü Kompaktlık)",
        "Concavity Worst (En Kötü İçbükeylik)",
        "Concave Points Worst (En Kötü İçbükey Nokta Sayısı)",
        "Symmetry Worst (En Kötü Simetri)",
        "Fractal Dimension Worst (En Kötü Fraktal Boyut)"
    ]

    features = []

    for i, name in enumerate(feature_names):
        default_val = (
            float(st.session_state.sample[i])
            if st.session_state.sample is not None
            else 0.0
        )
        value = st.number_input(name, value=default_val, format="%.4f")
        features.append(value)

    st.divider()

    # -------------------------
    # BUTONLAR (ALTTA)
    # -------------------------
    col_b1, col_b2, col_b3 = st.columns(3)

    with col_b1:
        if st.button("🧪 Örnek Hasta Verisiyle Otomatik Doldur", use_container_width=True):
            try:
                # 1. Rastgele bir indeks seç
                idx = random.randint(0, len(X_test_df) - 1)
                st.session_state.sample_idx = idx
                
                # 2. Seçilen satırı al
                # values yaparak numpy array'e çeviriyoruz
                raw_sample = X_test_df.iloc[idx].values
                
                # 3. Şekil (Shape) Kontrolü ve Düzenleme
                # Eğer veride fazladan index sütunu varsa veya boyut uyumsuzsa düzelt
                expected_features = scaler.n_features_in_  # Model kaç özellik bekliyor?
                current_features = raw_sample.shape[0]     # Bizde kaç özellik var?

                if current_features != expected_features:
                    # Genelde fazladan sütun varsa sondan veya baştan kırpmak gerekebilir
                    # Ancak burada sadece kullanıcıyı uyaralım veya reshape deneyelim
                    st.error(f"⚠️ Boyut Hatası: Model {expected_features} özellik bekliyor, ancak CSV dosyasından {current_features} özellik geldi.")
                else:
                    scaled_sample = raw_sample.reshape(1, -1)
                    
                    # 4. Ölçeklemeyi GERİ AL (inverse_transform)
                    original_sample = scaler.inverse_transform(scaled_sample)
                    
                    # 5. Session State'e kaydet
                    st.session_state.sample = original_sample[0]
                    st.session_state.is_test_sample = True
                    st.success("✅ Veri başarıyla dolduruldu ve geri dönüştürüldü.")

            except Exception as e:
                st.error(f"❌ Bir hata oluştu: {e}")

    with col_b2:
        if st.button("🔄 Formu Sıfırla", use_container_width=True):
            st.session_state.sample = None
            st.session_state.sample_idx = None
            st.session_state.is_test_sample = False

    with col_b3:
        tahmin_btn = st.button("🔮 Tahmin Et", use_container_width=True)

    # -------------------------
    # TAHMİN
    # -------------------------
    if tahmin_btn:
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        sonuc = "Benign" if prediction == 0 else "Malignant"

        if prediction == 0:
            st.success("🟢 Sonuç: **Benign (İyi Huylu)**")
        else:
            st.error("🔴 Sonuç: **Malignant (Kötü Huylu)**")

        # GERÇEK SONUÇ (ÜSTTE GÖSTERİLECEK)
        if st.session_state.is_test_sample and st.session_state.sample_idx is not None:
            gerçek_deger = y_test_df.iloc[st.session_state.sample_idx].values[0]
            gercek = "Malignant" if gerçek_deger == 1 else "Benign"
            st.info(f"📌 **Gerçek Sonuç (Test Verisi): {gercek}**")
        else:
            gercek = "Bilinmiyor (Manuel Giriş)"
            st.warning("📌 **Gerçek Sonuç: Bilinmiyor (Manuel Giriş)**")

        # GEÇMİŞE EKLE
        st.session_state.history.append({
            "Zaman": datetime.now().strftime("%H:%M:%S"),
            "Tahmin": sonuc,
            "Gerçek Sonuç": gercek
        })

    # -------------------------
    # GEÇMİŞ
    # -------------------------
    st.divider()
    st.subheader("📜 Hasta Deneme Geçmişi")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Geçmişi CSV olarak indir",
            data=csv,
            file_name="tahmin_gecmisi.csv",
            mime="text/csv"
        )
    else:
        st.info("Henüz bir tahmin yapılmadı.")

# ==================================================
# 📊 GRAFİKLER SEKME
# ==================================================
with tab_grafikler:

    st.subheader("📊 Model Performans Analizi")

    col1, col2 = st.columns(2)

    with col1:
        st.image("outputs/figure_loss.png", caption="Eğitim Kayıp (Loss)", use_container_width=True)
        st.image("outputs/figure_train_test_accuracy.png", caption="Train vs Test Accuracy", use_container_width=True)

    with col2:
        st.image("outputs/figure_accuracy_learning_curve.png", caption="Learning Curve", use_container_width=True)
        st.image("outputs/ann_confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)

    st.image(
        "outputs/model_comparison_all_models.png",
        caption="Model Karşılaştırması (Accuracy)",
        use_container_width=True
    )
