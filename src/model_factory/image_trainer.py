import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks # pyright: ignore[reportMissingImports]
import streamlit as st
import os
import time

# --- CANLI TAKİP İÇİN ÖZEL ARAÇ (CALLBACK) ---
class StreamlitCallback(callbacks.Callback):
    def __init__(self, epochs):
        self.epochs = epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Her tur (epoch) bittiğinde burası çalışır
        progress = (epoch + 1) / self.epochs
        self.progress_bar.progress(progress)
        
        # Geçen süreyi hesapla
        gecen_sure = time.time() - self.start_time
        ortalama_hiz = gecen_sure / (epoch + 1)
        
        # Ekrana yaz
        self.status_text.markdown(f"""
        **Durum:** {epoch + 1}/{self.epochs}. Tur Tamamlandı
        **Anlık Doğruluk:** %{logs['accuracy']*100:.2f}
        **Ortalama Hız:** {ortalama_hiz:.2f} saniye/tur
        """)

def train_image_model(model_name="Custom CNN"):
    """
    Seçilen mimariye göre model eğitir ve süreyi ölçer.
    """
    data_dir = "data/brain_mri"
    
    if not os.path.exists(data_dir):
        st.error("HATA: 'data/brain_mri' klasörü bulunamadı!")
        return None, None, None, 0 # Süre 0 döndür

    # Resim Ayarları
    img_height = 128
    img_width = 128
    batch_size = 32 # Hızlanması için 32 yaptık

    # 1. Veriyi Yükle
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir, validation_split=0.2, subset="training", seed=123,
            image_size=(img_height, img_width), batch_size=batch_size
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir, validation_split=0.2, subset="validation", seed=123,
            image_size=(img_height, img_width), batch_size=batch_size
        )
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None, None, None, 0

    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. Model Mimarisi
    num_classes = len(class_names)
    
    if model_name == "Custom CNN":
        model = models.Sequential([
            layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])
    elif model_name == "VGG16 (Transfer Learning)":
        base_model = applications.VGG16(input_shape=(img_height, img_width, 3),
                                        include_top=False, weights='imagenet')
        base_model.trainable = False
        model = models.Sequential([
            layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            base_model,
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])
    elif model_name == "MobileNetV2 (Hızlı)":
        base_model = applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                              include_top=False, weights='imagenet')
        base_model.trainable = False
        model = models.Sequential([
            layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 3. Eğitimi Başlat (Kronometre ile)
    epochs = 10
    
    # Callback'i tanımla
    my_callback = StreamlitCallback(epochs)
    
    t0 = time.time() # Başla
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[my_callback] # Canlı takibi ekledik
    )
    t1 = time.time() # Bitir
    
    total_time = t1 - t0
    
    return model, history, class_names, total_time