import pandas as pd
import streamlit as st

def load_data(file_path):
    """
    CSV dosyasını okur ve temel düzenlemeleri yapar.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Sütun isimlerindeki boşlukları temizleyelim (Opsiyonel ama iyi pratiktir)
        df.columns = df.columns.str.strip()
        
        # 'id' sütunu genelde tahmin için gereksizdir, varsa kaldıralım
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return None