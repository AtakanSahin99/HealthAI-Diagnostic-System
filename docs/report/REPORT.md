# HealthAI Diagnostic System Technical Report

HEALTHAI: YAPAY ZEKA DESTEKLİ ÇOK MODLU (MULTI-MODAL) SAĞLIK ANALİZ VE TANI DESTEK SİSTEMİ FİNAL PROJE RAPORU VE TEKNİK DOKÜMANTASYON

PROJE KİMLİK KARTI

Geliştirici

Proje Adı

Kategori

Yazılım Dili

Mimari Yaklaşım

Kütüphaneler

BİLGİLER

Atakan Şahin 23040101062

HealthAI (Health Analytics Intelligence)

Yapay Zeka, Tıbbi Görüntü İşleme, Veri Bilimi

Python 3.1x

Modüler Monolitik (Modular Monolith)

TensorFlow, Keras, Scikit-Learn, Streamlit, Pandas, SMOTE

YÖNETİCİ ÖZETİ (ABSTRACT)

Sağlık sektöründe dijitalleşmenin artmasıyla birlikte, büyük veri (Big Data) analizi ve yapay zeka uygulamaları hayati önem kazanmıştır. Dünya Sağlık Örgütü (WHO) verilerine göre, İnme (Stroke) dünya genelinde ölümlerin ve kalıcı sakatlıkların önde gelen nedenlerinden biridir. Erken teşhis, tedavi başarısını %50'nin üzerinde artırmaktadır.

Bu proje kapsamında geliştirilen HealthAI Platformu, geleneksel yöntemlerin aksine Çok Modlu (Multi-Modal) bir yaklaşım benimser. Sistem, hem yapılandırılmış klinik verileri (CSV formatındaki hasta öyküsü) hem de yapılandırılmamış görüntü verilerini (Beyin MR Taramaları) aynı anda işleyebilen hibrit bir yapay zeka motorudur. Geliştirilen prototip, dengesiz veri setlerinde dahi yüksek duyarlılıkla hastalık tespiti yapabilmekte ve hekimlere "İkinci Görüş" (Second Opinion) sunarak tanı hatalarını minimize etmeyi hedeflemektedir.

GİRİŞ VE PROBLEM TANIMI

Mevcut Durum ve Klinik Zorluklar

Modern tıpta doktorlar, sınırlı muayene sürelerinde yüzlerce parametreyi (Yaş, BMI, Şeker, Tansiyon vb.) incelemek ve aynı anda radyolojik görüntüleri yorumlamak zorundadır. Bu yoğunluk, "Algısal Körlük" ve yorgunluğa bağlı olarak yanlış teşhislere (False Negative) yol açabilmektedir. Özellikle inme gibi dakikaların önemli olduğu durumlarda, hızlı ve doğru karar vermek hayati önem taşır.

Projenin Çözüm Önerisi

HealthAI, bu darboğazları aşmak için Otomatize Edilmiş Bir Analiz Boru Hattı (Pipeline) sunar. Proje, ham veriyi alıp; ön işleme, modelleme ve görselleştirme aşamalarından geçirerek son kullanıcıya

(Doktora) "Risk Skoru" ve "Anomali Tespiti" şeklinde sunan entegre bir Karar Destek Sistemidir (CDSS).

YAZILIM MİMARİSİ VE TEKNOLOJİ YIĞINI

Proje, sürdürülebilirlik, ölçeklenebilirlik ve "Clean Code" prensiplerine uygun olarak Modüler Mimari ile tasarlanmıştır. Tek bir kod bloğu yerine, her işlevi ayrı bir birimin yönettiği katmanlı yapı kullanılmıştır.

Klasör Yapısı ve Modüller

data_pipeline/ (ETL Katmanı): Veri yükleme ve temizleme işlemlerini yönetir. Eksik veriler burada istatistiksel yöntemlerle tamamlanır.

model_factory/ (AI Motoru): Makine öğrenmesi (Trainer) ve Derin Öğrenme (CNN) modellerinin eğitildiği üretim katmanıdır.

visualization/ (Sunum Katmanı): Verilerin grafiksel dökümünü (Plotly/Seaborn) yapan görselleştirme modülüdür.

main_app.py (Orkestrasyon): Tüm modülleri yöneten ve kullanıcı arayüzünü (Streamlit UI) sunan ana dosyadır.

Kullanılan Teknolojiler

Python: Zengin veri bilimi kütüphaneleri nedeniyle tercih edilmiştir.

Streamlit: Hızlı prototipleme ve interaktif veri görselleştirme yeteneği için seçilmiştir.

TensorFlow & Keras: Görüntü işleme (CNN) tarafında endüstri standardı olduğu için kullanılmıştır.

Scikit-Learn: Klasik makine öğrenmesi algoritmalarının optimizasyonu için kullanılmıştır.

MATERYAL VE YÖNTEM (METHODOLOGY)

Veri Setleri

Projede iki farklı veri tipi kullanılarak sistemin hibrit yapısı güçlendirilmiştir:

Klinik Veri Seti (Tabular): 5000+ hastaya ait anonimleştirilmiş sağlık kayıtları (Yaş, Cinsiyet, Hipertansiyon, Kalp Hastalığı, Glukoz Seviyesi, BMI).

Görüntü Veri Seti (Imaging): Beyin MR (MRI) görüntülerinden oluşan, "Tümörlü (Yes)" ve "Sağlıklı (No)" olarak etiketlenmiş görüntü havuzu.

Veri Ön İşleme (Preprocessing)

Ham verinin modellere uygun hale getirilmesi için aşağıdaki "Veri Hijyeni" adımları uygulanmıştır:

Eksik Veri Tamamlama (Imputation): BMI sütunundaki eksik veriler, veri dağılımını bozmamak için Medyan/Ortalama değerleri ile doldurulmuştur.

Kategorik Dönüşüm (One-Hot Encoding): "Cinsiyet", "Evlilik Durumu" gibi metinsel veriler, matematiksel işlem yapılabilmesi için 0 ve 1'lerden oluşan vektörlere dönüştürülmüştür.

Görüntü Normalizasyonu: Kullanıcı tarafından yüklenen farklı formatlardaki görüntüler, sistemin kararlılığı için otomatik olarak RGB (3 Kanallı) formata ve 128x128 piksel boyutuna getirilmiştir.

Dengesiz Veri Yönetimi (SMOTE)

Sağlık verilerinde hasta sayısı, sağlıklı insan sayısına göre çok azdır. Bu durum modelin "tembel öğrenme" yapmasına neden olur. Projede SMOTE (Synthetic Minority Over-sampling Technique) kullanılarak, azınlık sınıfı (hastalar) sentetik olarak çoğaltılmış ve modelin duyarlılığı (Recall) artırılmıştır. Bu sayede modelin sadece sağlıklıları değil, hastaları da tanıması sağlanmıştır.

ALGORİTMA TASARIMI VE MODELLER

Modül 1: İnme Risk Analizi (Machine Learning)

Klinik veriler üzerinde 4 farklı algoritma yarıştırılarak en iyi sonucu veren model seçilmektedir:

Logistic Regression: Temel sınıflandırma ve olasılık hesabı.

Random Forest: Karar ağaçları topluluğu ile aşırı öğrenmeyi (Overfitting) engelleme.

Support Vector Machines (SVM): Verileri çok boyutlu uzayda ayırma.

K-Nearest Neighbors (KNN): Benzer hasta profillerini gruplama.

Modül 2: Beyin Görüntüleme (Deep Learning - CNN)

Görüntü analizi için insan beyninin görme korteksini taklit eden Convolutional Neural Networks (CNN) mimarisini kullanan özel bir model tasarlanmıştır.

Conv2D Katmanları: Görüntü üzerindeki kenar, doku ve anormallikleri (Tümör sınırları vb.) tespit eder.

MaxPooling: Görüntü boyutunu küçülterek işlem yükünü azaltır.

Dense Layer: Çıkarılan öznitelikleri kullanarak "Risk Var/Yok" kararını verir.

BULGULAR VE GÖRSELLEŞTİRME

Proje, "Explainable AI" (Açıklanabilir Yapay Zeka) prensibine uygun olarak sadece sonuç değil, nedensonuç ilişkisini gösteren grafikler sunar:

Confusion Matrix (Karmaşıklık Matrisi): Modelin doğru ve yanlış tahminlerinin şeffaf dağılımı.

ROC Eğrisi: Modelin ayırt etme gücünü gösteren performans eğrisi.

Korelasyon Isı Haritası (Heatmap): Hangi risk faktörlerinin (Örn: Yaş ve Şeker) birbiriyle ilişkili olduğunun renkli analizi.

Canlı Tahmin: Kullanıcıdan alınan anlık verilerle hesaplanan %'lik İnme Riski.

GÜVENLİK VE ETİK YAKLAŞIM

Veri Mahremiyeti (Privacy): HealthAI, "Stateless" (Durumsuz) bir mimariye sahiptir. Kullanıcıdan alınan yaş, cinsiyet veya yüklenen MR görüntüleri asla kalıcı bir veritabanına kaydedilmez. Tüm işlemler RAM (Geçici Bellek) üzerinde gerçekleşir ve oturum kapatıldığında veriler imha edilir.

Yasal Uyarı (Disclaimer): Sistem bir tıbbi cihaz değil, karar destek yazılımıdır. Üretilen sonuçlar kesin tanı yerine geçmez, hekime yardımcı tahmin niteliği taşır.

SONUÇ VE GELECEK VİZYONU

HealthAI projesi, teorik yapay zeka algoritmalarının gerçek bir halk sağlığı sorununa nasıl uygulanabileceğini gösteren somut bir mühendislik ürünüdür. Geliştirilen sistem, hem sayısal hem de görsel veriyi işleyerek "Bütüncül Tıp" anlayışına teknolojik bir altyapı sunmaktadır.

Gelecek Hedefleri:

Bulut Entegrasyonu: Sistemin AWS/Azure üzerine taşınarak hastane sistemlerine entegre edilmesi.

IoT Desteği: Akıllı saatlerden alınan anlık nabız verilerinin sisteme dahil edilmesi.

DICOM Desteği: Tıbbi standart görüntü formatlarının (DICOM) doğrudan işlenebilmesi.