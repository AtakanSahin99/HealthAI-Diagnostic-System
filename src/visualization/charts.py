import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# --- YENİ EKLENEN KÜTÜPHANELER ---
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np

def plot_target_distribution(df):
    """
    Hedef değişkenin (Stroke) dağılımını pasta grafiği ile gösterir.
    Rubrik: Sınıf dengesizliği görselleştirilmiş mi? -> EVET
    """
    # İnme geçiren ve geçirmeyen sayılarını al
    counts = df['stroke'].value_counts()
    
    # Plotly ile interaktif pasta grafiği
    fig = px.pie(
        values=counts.values,
        names=['Sağlıklı (0)', 'İnme (1)'],
        title='Hedef Değişken Dağılımı (İnme Durumu)',
        color_discrete_sequence=['#66b3ff', '#ff9999'] # Mavi ve Kırmızı
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_numerical_dist(df, column):
    """
    Seçilen sayısal sütunun histogramını çizer.
    """
    fig = px.histogram(df, x=column, color="stroke", 
                       title=f"{column} Değişkeninin Dağılımı",
                       marginal="box", # Üste kutu grafiği ekle
                       hover_data=df.columns)
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_heatmap(df):
    """
    Sayısal verilerin korelasyon matrisini (ısı haritası) çizer.
    Rubrik: Korelasyon matrisi ile sütun ilişkileri gösteriliyor mu? -> EVET
    """
    # Sadece sayısal sütunları seç
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Korelasyonu hesapla
    corr = numeric_df.corr()
    
    # Matplotlib/Seaborn ile çizim
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    plt.title("Özellikler Arası Korelasyon Matrisi")
    
    # Streamlit'e gönder
    st.pyplot(fig)

# --- YENİ EKLENEN FONKSİYONLAR ---

def plot_confusion_matrix(model, X_test, y_test, model_name):
    """
    Modelin tahminlerinin ne kadar doğru olduğunu gösteren matris.
    Rubrik: Confusion Matrix görselleştirilmiş mi? -> EVET
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Matrisi görselleştir (Heatmap şeklinde)
    z = cm.tolist() # Veriyi listeye çevir
    
    # Eksen isimleri
    x = ['Tahmin: Sağlıklı', 'Tahmin: İnme']
    y = ['Gerçek: Sağlıklı', 'Gerçek: İnme']
    
    # Renkli kareler oluştur
    # colorscale='Viridis' veya 'Blues' kullanabilirsin
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
    
    fig.update_layout(title_text=f'{model_name} - Karmaşıklık Matrisi (Confusion Matrix)')
    st.plotly_chart(fig, use_container_width=True)

def plot_roc_curve(modeller, X_test, y_test):
    """
    Tüm modellerin performans eğrilerini tek grafikte karşılaştırır.
    Rubrik: Tüm modellerin ROC eğrileri üst üste çizdirilmiş mi? -> EVET
    """
    fig = go.Figure()
    
    # Referans çizgisi (Rastgele tahmin çizgisi - Kesikli çizgi)
    fig.add_shape(
        type='line', line=dict(dash='dash', color='gray'),
        x0=0, x1=1, y0=0, y1=1
    )

    for isim, model in modeller.items():
        # Olasılık değerlerini al (Modelin kendine güven oranı)
        # Bazı modeller (SVM gibi) probability=True olmadan bunu yapamaz.
        if hasattr(model, "predict_proba"):
            try:
                # 1. sınıfın (İnme olma) olasılığını al
                y_score = model.predict_proba(X_test)[:, 1]
                
                fpr, tpr, _ = roc_curve(y_test, y_score)
                auc_score = auc(fpr, tpr)
                
                # Eğriyi çiz
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{isim} (AUC={auc_score:.2f})", mode='lines'))
            except Exception as e:
                st.warning(f"{isim} için ROC çizilemedi: {e}")

    fig.update_layout(
        title='ROC Eğrileri Karşılaştırması (Sol Üste Ne Kadar Yakınsa O Kadar İyi)',
        xaxis_title='Yanlış Pozitif Oranı (False Positive)',
        yaxis_title='Doğru Pozitif Oranı (True Positive)',
        width=700, height=500,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)