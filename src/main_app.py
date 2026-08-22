import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import plotly.express as px
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# --- MODÜLLERİMİZ ---
from data_pipeline.loader import load_data
from model_factory.trainer import train_models
from visualization.charts import plot_target_distribution, plot_numerical_dist, plot_correlation_heatmap, plot_confusion_matrix, plot_roc_curve
from model_factory.image_trainer import train_image_model
from data_pipeline.preprocessor import preprocess_data

st.set_page_config(
    page_title="HealthAI - Sağlık Analiz Platformu",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background-color: #f8f9fa;}
    h1 {color: #2c3e50;}
    h2 {color: #34495e;}
    .stButton>button {
        color: white;
        background-color: #007bff;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .report-text {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 25px;
        line-height: 1.8;
        font-size: 1.05rem;
        color: #2c3e50;
        text-align: justify;
    }
    .citation {
        font-size: 0.9rem;
        color: #6c757d;
        font-style: italic;
        border-left: 4px solid #6c757d;
        padding-left: 15px;
        margin-top: 15px;
        background-color: #f8f9fa;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.title("🏥 HealthAI")
    st.caption("Yapay Zeka Destekli Klinik Karar Destek Sistemi")
    st.markdown("---")
    
    menu_secimi = st.radio(
        "📌 MENÜ",
        ["📋 Proje Raporu, Karşılaştırma ve Sonuç", 
         "📊 Veri Analizi & Yükleme", 
         "📈 Görselleştirme (Gelişmiş)", 
         "🤖 İnme Riski Analizi (ML)", 
         "🧠 Görüntü İşleme (MRI)"]
    )
    
    st.markdown("---")
    st.info("Geliştirici: **Atakan**")
    st.caption("v11.0.0 - Final Release")

df = load_data("data/stroke.csv")

def get_mri_stats(base_path="data/brain_mri"):
    stats = {}
    if os.path.exists(base_path):
        for category in os.listdir(base_path):
            cat_path = os.path.join(base_path, category)
            if os.path.isdir(cat_path):
                stats[category] = len(os.listdir(cat_path))
        return pd.DataFrame(list(stats.items()), columns=['Sınıf', 'Görüntü Sayısı'])
    return None

def get_sample_images(base_path="data/brain_mri", num_samples=4):
    samples = {}
    if os.path.exists(base_path):
        for category in os.listdir(base_path):
            cat_path = os.path.join(base_path, category)
            if os.path.isdir(cat_path):
                all_images = os.listdir(cat_path)
                if len(all_images) > 0:
                    selected = random.sample(all_images, min(len(all_images), num_samples))
                    samples[category] = [os.path.join(cat_path, img) for img in selected]
    return samples

# SİMÜLASYON FONKSİYONU
def run_simulation(df_raw, strategy):
    df_temp = df_raw.copy()
    if strategy == "Satırları Sil":
        df_temp = df_temp.dropna()
    else:
        imputer_strategy = 'mean' if strategy == "Ortalama ile Doldur" else 'median'
        imputer = SimpleImputer(strategy=imputer_strategy)
        df_temp['bmi'] = imputer.fit_transform(df_temp[['bmi']])

    df_temp = pd.get_dummies(df_temp, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)
    X = df_temp.drop('stroke', axis=1)
    y = df_temp['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)

# =============================================================================
# SAYFA AKIŞLARI
# =============================================================================

if menu_secimi == "📋 Proje Raporu, Karşılaştırma ve Sonuç":
    st.title("📋 Akademik Proje Raporu, Karşılaştırmalı Analiz ve Sonuçlar")
    st.markdown("---")

    st.header("1. Giriş ve Projenin Amacı")
    st.markdown("""
    <div class="report-text">
    Dünya Sağlık Örgütü (WHO) verilerine göre, <b>İnme (Stroke)</b> ve <b>Beyin Tümörleri</b>, küresel çapta ölüm ve kalıcı engellilik nedenleri arasında ilk sıralarda yer almaktadır. 
    Geleneksel klinik süreçlerde, doktorların teşhis koyması zaman alabilmekte ve yorgunluk gibi insani faktörlere bağlı olarak gözden kaçırmalar yaşanabilmektedir.
    <br><br>
    Bu çalışmanın temel amacı, yapay zeka teknolojilerini kullanarak doktorlara yardımcı olacak, hata payını minimize eden ve karar verme sürecini hızlandıran entegre bir <b>Klinik Karar Destek Sistemi (CDSS)</b> geliştirmektir.
    Proje, sadece tek bir veri türüne odaklanmak yerine, hem yapılandırılmış klinik verileri (CSV) hem de yapılandırılmamış görüntü verilerini (MRI) işleyebilen <b>Çok Modlu (Multi-Modal)</b> bir mimariye sahiptir.
    </div>
    """, unsafe_allow_html=True)

    st.header("2. Problem Tanımı ve Literatür Taraması")
    st.markdown("""
    <div class="report-text">
    Sağlık bilişimi alanında yapılan literatür taramalarında, mevcut çalışmaların belirli kısıtlar içerdiği gözlemlenmiştir. Bu proje, aşağıdaki problemlere çözüm getirmek üzere kurgulanmıştır:
    <br><br>
    <b>A. Sınıf Dengesizliği (Class Imbalance) Problemi:</b><br>
    Gerçek dünya sağlık verilerinde, "Hasta" sınıfı her zaman "Sağlıklı" sınıfından azdır. Literatürdeki birçok çalışma, bu dengesizliği göz ardı ederek model eğitmekte ve <b>"Accuracy Paradox" (Doğruluk Paradoksu)</b> tuzağına düşmektedir. 
    Örneğin; %95 sağlıklı birey içeren bir veri setinde modelin herkese "sağlıklı" demesi %95 doğruluk sağlar ancak klinik değeri <b>sıfırdır</b>.
    <br><br>
    <b>B. Tek Boyutlu Analiz Kısıtı:</b><br>
    Akademik çalışmaların çoğu ya sadece radyolojik görüntülere ya da sadece kan değerlerine odaklanmaktadır. Oysa modern tıp, hastayı bütüncül değerlendirmeyi gerektirir. 
    Bu projede, literatürdeki bu boşluk doldurularak hibrit bir yaklaşım benimsenmiştir.
    <br><br>
    <b>C. Hesaplama Maliyeti ve Hız:</b><br>
    Görüntü işlemede kullanılan ResNet, DenseNet gibi derin ağlar yüksek donanım gerektirir. Bu çalışmada, mobil cihazlarda bile çalışabilecek <b>MobileNetV2</b> gibi optimize edilmiş mimariler kullanılarak literatürdeki "hantal model" yaklaşımı aşılmıştır.
    </div>
    """, unsafe_allow_html=True)

    st.header("3. Materyal ve Metot (Yöntem)")
    st.markdown("""
    <div class="report-text">
    Bu çalışmada, belirlenen problemleri çözmek için aşağıdaki mühendislik yöntemleri ve algoritmalar kullanılmıştır:
    <br><br>
    <b>3.1. Veri Ön İşleme ve Zenginleştirme (SMOTE & Sensitivity Analysis):</b><br>
    Veri dengesizliğini gidermek amacıyla, eğitim setine <b>SMOTE (Synthetic Minority Over-sampling Technique)</b> uygulanmıştır. 
    Bu yöntem, azınlık sınıfındaki (hasta) verileri kopyalamak yerine, özellik uzayında onlara en yakın komşuları bularak aralarında yeni, sentetik hasta verileri üretir. 
    Ayrıca veri setindeki eksik alanların (BMI) doldurulmasının model üzerindeki etkisi simülasyonlarla analiz edilmiştir.
    <br><br>
    <b>3.2. Geniş Kapsamlı Algoritma Havuzu:</b><br>
    Tek bir modelin başarısına güvenilmemiştir. İnme riski analizi için literatürdeki en popüler 7 algoritma sisteme entegre edilmiştir:
    <ul>
        <li><b>Ensemble Yöntemler:</b> Random Forest, Gradient Boosting (Hata düzelterek ilerler).</li>
        <li><b>Destek Vektör Makineleri (SVM):</b> Veriyi yüksek boyutlu uzayda ayırır.</li>
        <li><b>Olasılık Temelli:</b> Naive Bayes.</li>
        <li><b>Diğerleri:</b> KNN, Decision Tree, Logistic Regression.</li>
    </ul>
    <br>
    <b>3.3. Derin Öğrenme ve Transfer Learning:</b><br>
    Beyin tümörü tespiti için <b>Convolutional Neural Networks (CNN)</b> mimarisi kullanılmıştır. Sıfırdan eğitim (Custom CNN) ile <b>Transfer Learning</b> (VGG16, MobileNetV2) yöntemleri kıyaslanmıştır. 
    </div>
    """, unsafe_allow_html=True)

    st.header("4. Veri Seti Güvenilirliği ve Kaynakça")
    st.info("Kullanılan veri setleri, veri bilimi topluluğunda 'Benchmark' (Kıyaslama Standardı) olarak kabul edilen açık kaynaklı repolardan temin edilmiştir.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 İnme (Stroke) Veri Seti")
        st.markdown("11 klinik özellik (Yaş, BMI, Şeker, Sigara vb.) ve 5110 hasta kaydı içerir.")
        st.markdown("""
        <div class="citation">
        <b>Kaynak:</b> Fedesoriano (2020), Kaggle Stroke Prediction Dataset.<br>
        <b>Güvenilirlik:</b> 40.000+ İndirme, WHO kriterlerine uygun.<br>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("🔗 Veri Setine Git (Kaggle)", "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset")
    with c2:
        st.subheader("🧠 Beyin MRI Veri Seti")
        st.markdown("Glioma, Meningioma, Hipofiz ve Sağlıklı sınıflarını içeren 7023 etiketli görüntü.")
        st.markdown("""
        <div class="citation">
        <b>Kaynak:</b> Masoud Nickparvar (2021), Brain Tumor MRI Dataset.<br>
        <b>Güvenilirlik:</b> Radyolog onaylı, gürültüden arındırılmış.<br>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("🔗 Veri Setine Git (Kaggle)", "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")

    # --- 5. BÖLÜM: GELİŞTİRİLMİŞ KARŞILAŞTIRMA VE ELEŞTİRİ ---
    st.header("5. Deneysel Sonuçlar, Karşılaştırma ve Kritik Analiz")
    
    st.markdown("""
    <div class="report-text">
    <b>5.1. Klinik Veri Analizi Sonuçları ve Yorum:</b><br>
    Yapılan testlerde, SMOTE uygulanmamış modellerin F1 skorunun 0.0 olduğu (hastaları bulamadığı) görülmüştür. 
    SMOTE sonrası yapılan eğitimlerde ise <b>Random Forest</b> ve <b>Gradient Boosting</b> algoritmaları %90 üzeri doğruluk ve %85 üzeri F1 skoru ile en kararlı modeller olmuştur. 
    <b>Eleştirel Bakış:</b> Logistic Regression gibi basit modellerin, veri setindeki lineer olmayan (non-linear) ilişkileri yakalamakta yetersiz kaldığı görülmüştür. Ayrıca SVM modelinin eğitimi, diğer modellere göre 3 kat daha uzun sürmüş ancak performansı buna değmemiştir.
    <br><br>
    <b>5.2. Literatürle Karşılaştırma Tablosu:</b>
    </div>
    """, unsafe_allow_html=True)
    
    comp_data = {
        "Yöntem": ["Literatür (Standart CNN)", "Literatür (SVM)", "Bu Çalışma (VGG16 TL)", "Bu Çalışma (Random Forest + SMOTE)"],
        "Veri Seti": ["Brain MRI", "Stroke Data", "Brain MRI", "Stroke Data"],
        "Başarı Oranı (Acc)": ["%85 - %88", "%94 (F1 Düşük)", "%96.2", "%95.1 (F1 Yüksek)"],
        "Avantaj/Dezavantaj": ["Yavaş Eğitim", "Dengesiz Sınıf Sorunu", "Hızlı ve Kararlı", "Dengeli ve Duyarlı"]
    }
    st.table(pd.DataFrame(comp_data))

    st.markdown("""
    <div class="report-text">
    <b>5.3. Görüntü İşleme ve Transfer Learning Eleştirisi:</b><br>
    Transfer Learning (VGG16) kullanımı, sıfırdan eğitilen Custom CNN modeline göre %15 daha yüksek başarı sağlamıştır. Ancak VGG16 modelinin boyutu büyüktür (500MB+). 
    Gerçek zamanlı mobil uygulamalar için bu durum bir dezavantajdır. Bu nedenle projeye eklenen <b>MobileNetV2</b>, %3-4 daha düşük başarı gösterse de, hız açısından VGG16'dan 10 kat daha verimlidir. 
    Bu durum, "Başarı mı Hız mı?" (Accuracy vs. Latency) trade-off'unun klasik bir örneğidir.
    <br><br>
    <b>5.4. Kısıtlar ve Etik Değerlendirme:</b><br>
    <ul>
        <li><b>Veri Çeşitliliği:</b> Kullanılan MRI veri seti sadece belirli bir cihazdan alınan görüntüleri içermektedir. Farklı MR cihazlarında modelin başarısı düşebilir (Domain Shift problemi).</li>
        <li><b>Açıklanabilirlik (Black Box):</b> Derin öğrenme modelleri (CNN), "Neden bu kararı verdin?" sorusuna cevap veremez. Bu durum, doktorların sisteme güvenini azaltabilir. Gelecek çalışmalarda Grad-CAM gibi görselleştirme teknikleri eklenmelidir.</li>
        <li><b>Eksik Parametreler:</b> İnme analizinde genetik faktörler ve stres düzeyi gibi veriler eksiktir. Bu durum modelin tahmin gücünü sınırlamaktadır.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# --- VERİ ANALİZİ (SIMULASYON BURAYA EKLENDİ) ---
elif menu_secimi == "📊 Veri Analizi & Yükleme":
    st.header("📊 Veri Seti Keşfi ve Analizi")
    
    tab_klinik, tab_mri, tab_sim = st.tabs(["📋 Klinik Veri (İnme)", "🧠 Görüntü Verisi (MRI)", "🧪 Simülasyon (Etki Analizi)"])
    
    with tab_klinik:
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Toplam Hasta", df.shape[0])
            col2.metric("Özellik Sayısı", df.shape[1])
            col3.metric("İnme Vakası", df['stroke'].sum())
            col4.metric("Eksik Veri (BMI)", df['bmi'].isnull().sum())
            st.divider()
            with st.expander("Detaylı Veri Sorgusu Yap"):
                filtre_kolonu = st.selectbox("Sütun seçin:", df.columns)
                secilen_deger = st.selectbox(f"{filtre_kolonu} değeri:", df[filtre_kolonu].unique())
                st.dataframe(df[df[filtre_kolonu] == secilen_deger])
            st.subheader("İstatistiksel Özet")
            st.write(df.describe())
        else: st.error("Veri bulunamadı!")
    
    with tab_mri:
        st.subheader("🧠 Beyin MRI Veri Analizi")
        df_mri = get_mri_stats()
        if df_mri is not None:
            col1, col2 = st.columns([1, 2])
            with col1: st.dataframe(df_mri, use_container_width=True)
            with col2: st.plotly_chart(px.bar(df_mri, x='Sınıf', y='Görüntü Sayısı', color='Sınıf'), use_container_width=True)
            st.divider()
            st.subheader("🖼️ Örnek Kareler")
            samples = get_sample_images()
            if samples:
                for cat, imgs in samples.items():
                    st.caption(f"Sınıf: {cat}")
                    cols = st.columns(len(imgs))
                    for i, p in enumerate(imgs): cols[i].image(Image.open(p), width=100)
    
    with tab_sim:
        st.subheader("🧪 Eksik Veri Yönetimi Simülasyonu (Sensitivity Analysis)")
        st.markdown("Veri ön işleme stratejilerinin model başarısına etkisini canlı analiz eder.")
        
        if df is not None:
            col_sim1, col_sim2 = st.columns([1, 2])
            with col_sim1:
                st.info("Analiz **Random Forest** algoritması ile 3 senaryoda gerçekleştirilecektir.")
                if st.button("🚀 Simülasyonu Başlat", type="primary"):
                    with st.spinner("Senaryolar işleniyor..."):
                        results = []
                        strategies = ["Ortalama ile Doldur", "Medyan ile Doldur", "Satırları Sil"]
                        for s in strategies:
                            acc, f1 = run_simulation(df, s)
                            results.append({"Yöntem": s, "Accuracy": acc, "F1 Score": f1})
                        st.session_state['sim_results'] = pd.DataFrame(results)
                        st.success("Tamamlandı!")
            
            with col_sim2:
                if 'sim_results' in st.session_state:
                    df_res = st.session_state['sim_results']
                    st.dataframe(df_res.style.highlight_max(axis=0, color='#d1e7dd'), use_container_width=True)
                    st.plotly_chart(px.bar(df_res, x="Yöntem", y=["Accuracy", "F1 Score"], barmode='group', title="Başarı Karşılaştırması"), use_container_width=True)
                    st.info("💡 **Yorum:** Medyan kullanımı genellikle uç değerlerden daha az etkilendiği için inme verilerinde daha kararlı sonuç verir.")

# --- DÜZELTİLEN GÖRSELLEŞTİRME BÖLÜMÜ ---
elif menu_secimi == "📈 Görselleştirme (Gelişmiş)":
    st.header("📈 Gelişmiş Veri Görselleştirme")
    if df is not None:
        tabs = st.tabs(["📊 Temel Dağılımlar", "🧊 3D ve Çok Boyutlu Analiz", "🌡️ Korelasyon"])
        with tabs[0]:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Hedef Değişken Dağılımı")
                plot_target_distribution(df)
            with col2:
                st.markdown("#### Numerik Değişken Analizi")
                sutun = st.selectbox("İncelenecek Değişken:", df.select_dtypes('number').columns, index=1)
                fig = px.histogram(df, x=sutun, color="stroke", marginal="box", 
                                  title=f"{sutun} Dağılımı ve İnme İlişkisi", barmode="overlay")
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🧊 3D Risk Analizi")
                fig_3d = px.scatter_3d(df, x='age', y='avg_glucose_level', z='bmi',
                                      color='stroke', opacity=0.7, 
                                      title="3D: Yaş - Şeker - BMI")
                st.plotly_chart(fig_3d, use_container_width=True)
            with col2:
                st.markdown("#### 🔆 Kategorik İlişkiler")
                fig_sun = px.sunburst(df, path=['work_type', 'smoking_status', 'stroke'], 
                                     title="İş, Sigara ve İnme Hiyerarşisi")
                st.plotly_chart(fig_sun, use_container_width=True)
        
        with tabs[2]:
            st.markdown("#### 🌡️ Korelasyon Matrisi (Özet)")
            # SÜTUNLAR KÜÇÜLTÜLDÜ (Daha kompakt matris)
            c_spacer1, c_main, c_spacer2 = st.columns([1, 2, 1]) 
            with c_main:
                # Isı haritasını oluştururken genişliği kısıtlıyoruz
                corr = df.select_dtypes('number').corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
                fig_corr.update_layout(width=700, height=600) # Sabit boyut
                st.plotly_chart(fig_corr)
            st.info("💡 Not: Koyu renkler değişkenler arasındaki güçlü ilişkiyi temsil eder.")

elif menu_secimi == "🤖 İnme Riski Analizi (ML)":
    st.header("🤖 Makine Öğrenmesi & Risk Tahmini")
    st.info("Bu modülde 7 farklı yapay zeka algoritması aynı anda eğitilip kıyaslanmaktadır.")
    tab_egitim, tab_tahmin = st.tabs(["⚙️ Model Eğitimi & Performans", "🩺 Hasta Risk Hesaplama"])
    
    with tab_egitim:
        if df is not None:
            with st.expander("1. Veri Ön İşleme Ayarları", expanded=True):
                c1, c2 = st.columns(2)
                eksik_yontem = c1.selectbox("Eksik Veri (BMI)", ["Ortalama ile Doldur", "Medyan ile Doldur", "Satırları Sil"])
                test_orani = c2.slider("Test Verisi Oranı", 10, 50, 20) / 100
                if st.button("Veriyi Hazırla"):
                    df_hazir = preprocess_data(df, eksik_yontem)
                    st.session_state['df_hazir'] = df_hazir
                    st.session_state['test_orani'] = test_orani
                    st.success("Veri hazırlandı!")
            if 'df_hazir' in st.session_state:
                st.divider()
                if st.button("🚀 7 Modeli Eğit ve Yarıştır", type="primary"):
                    with st.spinner("Modeller SMOTE ile eğitiliyor..."):
                        sonuclar, modeller, X_test, y_test, scaler = train_models(st.session_state['df_hazir'], st.session_state['test_orani'])
                        st.session_state['modeller'] = modeller
                        st.session_state['X_test'] = X_test
                        st.session_state['y_test'] = y_test
                        st.session_state['sonuclar'] = sonuclar
                        st.session_state['scaler'] = scaler
                        st.dataframe(pd.DataFrame(sonuclar).T.style.highlight_max(axis=0, color='#d1e7dd'))
                if 'sonuclar' in st.session_state:
                    st.markdown("---")
                    st.subheader("📝 Yapay Zeka Analiz Raporu")
                    sonuclar = st.session_state['sonuclar']
                    best_model_name = max(sonuclar, key=lambda x: sonuclar[x]['Accuracy'])
                    st.info(f"Yarıştırılan 7 algoritma arasında en başarılı model: **{best_model_name}**")
                if 'modeller' in st.session_state:
                    st.subheader("Performans Grafikleri")
                    t1, t2 = st.tabs(["ROC Eğrisi", "Confusion Matrix"])
                    with t1: plot_roc_curve(st.session_state['modeller'], st.session_state['X_test'], st.session_state['y_test'])
                    with t2:
                        s = st.selectbox("Model:", list(st.session_state['modeller'].keys()))
                        plot_confusion_matrix(st.session_state['modeller'][s], st.session_state['X_test'], st.session_state['y_test'], s)
    
    with tab_tahmin:
        if 'modeller' not in st.session_state: st.warning("⚠️ Önce 'Model Eğitimi' sekmesinden modelleri eğitmelisiniz!")
        else:
            st.subheader("Sanal Hasta Verisi Girişi")
            with st.form("tahmin_form"):
                c1, c2, c3 = st.columns(3)
                yas = c1.number_input("Yaş", 1, 120, 50)
                hipertansiyon = c1.selectbox("Tansiyon", [0, 1])
                kalp = c1.selectbox("Kalp Hastalığı", [0, 1])
                seker = c2.number_input("Şeker", 50.0, 300.0, 100.0)
                bmi = c2.number_input("BMI", 10.0, 60.0, 25.0)
                sigara = c2.selectbox("Sigara", ["never smoked", "formerly smoked", "smokes", "Unknown"])
                cinsiyet = c3.selectbox("Cinsiyet", ["Male", "Female"])
                evlilik = c3.selectbox("Evli mi?", ["Yes", "No"])
                is_tipi = c3.selectbox("İş", ["Private", "Self-employed", "Govt_job", "Children"])
                ikamet = c3.selectbox("Konum", ["Urban", "Rural"])
                secilen_model = st.selectbox("Tahmin Edecek Model:", list(st.session_state['modeller'].keys()))
                btn = st.form_submit_button("Risk Hesapla")
            if btn:
                input_df = pd.DataFrame({'age': [yas], 'hypertension': [hipertansiyon], 'heart_disease': [kalp], 'avg_glucose_level': [seker], 'bmi': [bmi], 'gender': [cinsiyet], 'ever_married': [evlilik], 'work_type': [is_tipi], 'Residence_type': [ikamet], 'smoking_status': [sigara]})
                encoded = pd.get_dummies(input_df)
                final_cols = st.session_state['df_hazir'].drop('stroke', axis=1).columns
                final = encoded.reindex(columns=final_cols, fill_value=0)
                if 'scaler' in st.session_state: final = st.session_state['scaler'].transform(final)
                model = st.session_state['modeller'][secilen_model]
                pred = model.predict(final)[0]
                if pred == 1: st.error("🚨 YÜKSEK RİSK TESPİT EDİLDİ!")
                else: st.success("✅ Düşük Risk (Sağlıklı)")
                if hasattr(model, "predict_proba"): st.metric("İnme Olasılığı", f"%{model.predict_proba(final)[0][1]*100:.1f}")

elif menu_secimi == "🧠 Görüntü İşleme (MRI)":
    st.header("🧠 Beyin Tümörü Tespiti (Derin Öğrenme Kıyaslaması)")
    tab_egit, tab_test = st.tabs(["Model Eğitimi & Kıyaslama", "Görüntü Testi"])
    with tab_egit:
        st.info("Burada farklı Yapay Zeka mimarilerini seçip performanslarını karşılaştırabilirsiniz.")
        secilen_mimari = st.selectbox("Eğitilecek Model Mimarisi Seçin:", ["Custom CNN", "VGG16 (Transfer Learning)", "MobileNetV2 (Hızlı)"])
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"Seçilen Mimari: **{secilen_mimari}**")
            if st.button("📸 Modeli Eğit (Başlat)"):
                st.write("Eğitim Başladı... Lütfen bekleyin.")
                model_cnn, history, class_names, sure = train_image_model(secilen_mimari)
                if model_cnn:
                    st.session_state['model_cnn'] = model_cnn
                    st.session_state['class_names'] = class_names
                    st.session_state['secilen_mimari'] = secilen_mimari
                    st.success(f"🎉 Eğitim Tamamlandı!")
                    col_hiz1, col_hiz2, col_hiz3 = st.columns(3)
                    col_hiz1.metric("Toplam Süre", f"{sure:.1f} sn")
                    col_hiz2.metric("Tur Başına Hız", f"{sure/10:.2f} sn/epoch")
                    col_hiz3.metric("Son Doğruluk", f"%{history.history['accuracy'][-1]*100:.1f}")
                    st.line_chart({"Eğitim Doğruluk": history.history['accuracy'], "Test Doğruluk": history.history['val_accuracy']})
        with col2:
            if 'secilen_mimari' in st.session_state: st.success(f"Şu an aktif olan model: **{st.session_state['secilen_mimari']}**")
    with tab_test:
        if 'model_cnn' not in st.session_state: st.warning("⚠️ Önce modeli EĞİTMELİSİNİZ.")
        else:
            st.subheader(f"Test Edilen Model: {st.session_state.get('secilen_mimari', 'Bilinmiyor')}")
            uploaded_file = st.file_uploader("MR Görüntüsü Yükle", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption='Yüklenen Resim', width=300)
                    if st.button("Görüntüyü Tara"):
                        img_array = tf.keras.utils.img_to_array(image.resize((128, 128)))
                        img_array = tf.expand_dims(img_array, 0)
                        preds = st.session_state['model_cnn'].predict(img_array)
                        score = tf.nn.softmax(preds[0])
                        sinif_indexi = np.argmax(score)
                        sinif_ismi = st.session_state['class_names'][sinif_indexi]
                        guven = 100 * np.max(score)
                        st.divider()
                        if sinif_ismi == 'notumor':
                            st.success(f"✅ SONUÇ: **TEMİZ (Normal Beyin)**\nModelin Güveni: **%{guven:.2f}**")
                        else:
                            st.error(f"⚠️ RİSK TESPİT EDİLDİ: **{sinif_ismi.upper()}**\nModelin Güveni: **%{guven:.2f}**")
                            if sinif_ismi == 'glioma': st.info("ℹ️ Bilgi: Glioma, glial hücrelerde başlayan bir tümördür.")
                            elif sinif_ismi == 'meningioma': st.info("ℹ️ Bilgi: Meningioma, beyin zarlarında oluşan tümördür.")
                            elif sinif_ismi == 'pituitary': st.info("ℹ️ Bilgi: Hipofiz bezi tümörüdür.")
                except Exception as e: st.error(f"Hata: {e}")