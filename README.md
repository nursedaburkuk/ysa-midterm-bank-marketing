# MLP ile Sınıflandırma — Bank Marketing (Yapay Sinir Ağları Ara Sınav Projesi)

Bu depo, UCI Bank Marketing veri setinin `bank-additional.csv` dosyası üzerinde PyTorch MLP modelleriyle yapılan EDA, kat-içi ön işleme, manuel ve otomatik hiperparametre araması, nested cross-validation ve kilitli bağımsız test çalışmasını içerir. Teknik rapor `report.pdf` dosyasındadır.

## 1. Veri setinin indirilmesi

Ham veri depoya yüklenmemiştir.

1. https://archive.ics.uci.edu/dataset/222/bank+marketing adresinden `bank+marketing.zip` dosyasını indirin.
2. Zip içindeki `bank-additional.zip` dosyasını açın.
3. Yalnızca `bank-additional/bank-additional.csv` dosyasını bu deponun `data/` klasörüne kopyalayın (`data/bank-additional.csv`).

Kullanılan dosyanın SHA-256 özeti: `7e59cf650004d65d1c9d6b08553bad2ee9a9ad70d594f536e3c584ee6ed5df50` (4.119 satır, 21 sütun).

Atıf: Moro, S., Rita, P. ve Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

## 2. Kurulum

```bash
git clone https://github.com/nursedaburkuk/ysa-midterm-bank-marketing.git
cd ysa-midterm-bank-marketing
pip install -r requirements.txt
```

Google Colab'da: depoyu klonlayın, `%cd ysa-midterm-bank-marketing` ile klasöre geçin, `data/bank-additional.csv` dosyasını yükleyin ve `!pip install -r requirements.txt` komutunu çalıştırın. Notebook'lar depo kök klasöründen çalıştırılmalıdır.

## 3. Çalıştırma sırası

| Sıra | Dosya | İçerik |
|---|---|---|
| 1 | `01_eda.ipynb` | Veri kontrolü, kopya kayıt kontrolü, stratified %80/%20 bölünme (`results/dataset_split.csv`), yalnızca development kümesinde EDA ve deney hipotezleri |
| 2 | `python -m src.make_configs` | M1–M6 ve R1–R6 adaylarını, arama uzayını ve eğitim ayarlarını `configs/` klasörüne yazar (nested CV'den önce çalıştırılmış ve commit edilmiştir) |
| 3 | `02_nested_cv.ipynb` | Kat-içi ön işleme, manuel ve otomatik arama, 3×3 nested CV, referans modeller, tüm development'ta iç CV ile nihai seçim |
| 4 | `03_final_test.ipynb` | Bölüm A: 3 seed ile nihai eğitim ve etiketsiz `test_predictions_locked.csv`. Bölüm B: yalnızca kilitli dosya commit edildikten sonra test metrikleri, bootstrap güven aralıkları, grafikler ve hata analizi |
| 5 | `python -m src.make_report` | `report.pdf` raporunu sonuç dosyalarından üretir |

`03_final_test.ipynb`, kilitli tahmin dosyası git geçmişinde yoksa test etiketlerini okumadan durur. Dosya commit edildikten sonra yeniden çalıştırıldığında tahminleri yeniden üretir, kilitli dosyayla aynı olduklarını doğrular ve ancak ondan sonra etiketleri okur.

## 4. Seed politikası

Öğrenci numarasının son beş hanesi 29048'dir. Temel seed = 29048 + 2026 = **31074**. Alt seed'ler temel seed'e sabit kaydırma eklenerek türetilir (`src/project.py`, `configs/seeds.json`):

| Amaç | Seed |
|---|---|
| Python `random` | 31074 |
| NumPy | 31075 |
| PyTorch (birincil ağırlık seed'i) | 31076 |
| Train/test split | 31077 |
| Dış CV | 31078 |
| İç CV | 31079 |
| %15 erken durdurma izleme bölünmesi | 31080 |
| Random search örneklemesi (+0/+1/+2: SGD/Adam/AdamW) | 31081 |
| Bootstrap | 31082 |
| Seed kararlılığı için 2. ve 3. ağırlık seed'i | 31176, 31276 |

PyTorch'ta `torch.use_deterministic_algorithms(True)` ve tek iş parçacığı kullanılmıştır.

## 5. Sistem ve kütüphane sürümleri

Deneyler şu ortamda çalıştırılmıştır: Ubuntu 24.04.5 LTS, Intel Xeon 2.10 GHz (2 çekirdek), 7 GB RAM, GPU yok (yalnızca CPU). Python 3.13.18, PyTorch 2.15.0, scikit-learn 1.9.1, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1, matplotlib 3.11.2, joblib 1.6.0. Tam liste `requirements.txt` dosyasındadır. 02 notebook'unun toplam çalışma süresi yaklaşık 3 dakikadır.

## 6. Klasör yapısı

```
configs/   arama uzayı, M1–M6, R1–R6, eğitim ayarları, seed'ler, ek analiz adayları
results/   dataset_split.csv, EDA tabloları, manual_search_results.csv, random_search_results.csv,
           outer_cv_results.csv, final_selection.json, final_hyperparameters.json,
           test_predictions_locked.csv, test_metrics.csv, seed_stability_results.csv,
           test_bootstrap_ci.csv, logs/ (epoch bazlı eğitim kayıtları), final_models/
figures/   raporda kullanılan, kodla üretilmiş grafikler
src/       project.py (seed, veri, ön işleme, MLP, eğitim, metrikler), nested.py, make_configs.py, make_report.py
```

## 7. Zaman sıralı deney kaydı

Commit sırası: (1) bölünme ve EDA, (2) nested CV'den önce sabitlenen konfigürasyonlar, (3) nested CV sonuçları ve nihai seçim, (4) etiketsiz kilitli test tahminleri, (5) test metrikleri, (6) rapor. Veri seti kamusal olduğu için bu kayıt kör test garantisi değildir; deneyin zaman sırasını gösterir.
