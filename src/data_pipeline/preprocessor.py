import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, eksik_veri_yontemi="Ortalama ile Doldur"):
    """
    Veriyi temizler ve makine öğrenmesine hazırlar.
    Rubrik: 
    1. Eksik veri stratejisi (Ortalama/Medyan)
    2. Kategorik değişken kodlama (One-Hot Encoding)
    """
    # 1. Verinin kopyasını al (Orijinali bozulmasın)
    df_processed = df.copy()
    
    # --- EKSİK VERİLERİ DOLDURMA (BMI Sütunu) ---
    if eksik_veri_yontemi == "Ortalama ile Doldur":
        # BMI sütununun ortalamasını al ve boş yerlere yaz
        ortalama_bmi = df_processed['bmi'].mean()
        df_processed['bmi'] = df_processed['bmi'].fillna(ortalama_bmi)
        
    elif eksik_veri_yontemi == "Medyan ile Doldur":
        # Ortadaki sayıyı (medyan) al
        medyan_bmi = df_processed['bmi'].median()
        df_processed['bmi'] = df_processed['bmi'].fillna(medyan_bmi)
        
    elif eksik_veri_yontemi == "Satırları Sil":
        # Boş olan satırları komple sil
        df_processed = df_processed.dropna()

    # --- KATEGORİK (YAZI) VERİLERİ SAYIYA ÇEVİRME ---
    # Pandas'ın get_dummies fonksiyonu "One-Hot Encoding" yapar.
    # Örn: Cinsiyet sütununu -> Cinsiyet_Male (0/1), Cinsiyet_Female (0/1) diye böler.
    
    # Hangi sütunlar yazı tipinde? Onları bul.
    kategorik_sutunlar = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    # Otomatik dönüştür (drop_first=True diyerek gereksiz kolon tekrarını önlüyoruz)
    df_processed = pd.get_dummies(df_processed, columns=kategorik_sutunlar, drop_first=True)
    
    return df_processed