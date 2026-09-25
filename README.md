# Integrasi Association Rule Mining (Apriori) dan Extreme Gradient Boosting (XGBoost) untuk Sistem Cerdas Cross-Selling pada Ritel Modern

## Abstrak

Proyek ini membangun sistem rekomendasi cross-selling untuk ritel modern dengan menggabungkan dua lapisan analitik. Association Rule Mining memakai algoritma Apriori untuk menemukan pasangan produk yang sering dibeli bersama. Extreme Gradient Boosting (XGBoost) kemudian memakai aturan tersebut, bersama perilaku belanja pelanggan, untuk memperkirakan apakah suatu produk consequent akan dibeli pada pesanan berikutnya.

Data sumber terdiri atas enam berkas transaksi historis: pesanan, isi keranjang, katalog produk, lorong, dan departemen. Keluaran akhirnya adalah model XGBoost yang dapat dipanggil secara langsung untuk memutuskan apakah sebuah aturan cross-selling layak direkomendasikan.

## Arsitektur Sistem 5-Layer

Sistem disusun sebagai lima lapisan yang berurutan. Setiap lapisan hanya menerima keluaran lapisan sebelumnya, sehingga alur dari data mentah sampai rekomendasi dapat dijelaskan satu per satu pada saat bimbingan maupun sidang.

| Layer | Nama | Masukan | Keluaran |
|---|---|---|---|
| 1 | Data Input dan Schema Validation | Enam berkas CSV mentah | Peta skema, kunci, dan relasi yang lolos pemeriksaan |
| 2 | Pre-processing dan Apriori Mining | Pesanan `prior` | `outputs/apriori_rules.csv` |
| 3 | Feature Engineering dan SMOTE | Aturan Apriori dan riwayat pengguna | `outputs/layer3_smoted_features.csv` |
| 4 | Model Training dan Benchmarking | Matriks fitur dan label | `models/xgboost_cross_sell_model.pkl` |
| 5 | Integration dan Real-time Inference | Vektor fitur pengguna baru | Probabilitas dan keputusan Ya/Tidak |

### Layer 1 — Data Input dan Schema Validation

Lapisan ini memuat enam berkas transaksi ritel dan memeriksa apakah skema serta relasinya utuh sebelum data dipakai untuk menambang aturan atau melatih model.

Berkas yang dibaca adalah `orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`, `aisles.csv`, dan `departments.csv`. Pemeriksaan mencakup jumlah baris dan kolom, tipe data, nilai kosong pada kolom kunci, serta keunikan primary key. Relasi yang divalidasi adalah `products.aisle_id` ke `aisles.aisle_id`, `products.department_id` ke `departments.department_id`, `order_id` pada isi keranjang ke `orders.order_id`, dan `product_id` pada isi keranjang ke `products.product_id`. Identitas pengguna, `user_id`, hanya tersimpan di header pesanan.

Hasil pemeriksaan ini menjadi kontrak data untuk lapisan berikutnya: pesanan `prior` memiliki isi keranjang historis, pesanan `train` memiliki isi keranjang berlabel, dan pesanan `test` tidak memiliki daftar produk sehingga tidak dipakai sebagai label.

### Layer 2 — Pre-processing dan Apriori Mining

Lapisan ini mengubah riwayat belanja menjadi aturan asosiasi. Pada mode laptop, notebook mengambil 50.000 `order_id` pertama yang `eval_set`-nya `prior`, karena `order_products__prior.csv` berisi sekitar 32,4 juta baris. Mode server kampus memakai seluruh pesanan prior tanpa sampling. File prior dibaca per batch, lalu digabung dengan katalog produk sehingga setiap baris memuat `order_id`, `product_id`, dan `product_name`.

Keranjang itu diubah menjadi matriks one-hot: baris adalah pesanan, kolom adalah nama produk, dan nilai menunjukkan produk tersebut ada atau tidak di dalam pesanan. Produk yang support-nya di bawah ambang dibuang sebelum matriks dibentuk, karena item tersebut tidak mungkin masuk itemset yang lolos ambang Apriori. Pada mode laptop ambangnya `min_support = 0.01`. Pada mode server ambangnya `0.005`. Aturan yang confidence-nya minimal 0,1 dipertahankan, diurutkan menurut confidence dan lift, lalu disimpan ke `outputs/apriori_rules.csv`. Berkas inilah yang menjadi jembatan dari pola belanja bersama ke fitur prediksi.

### Layer 3 — Feature Engineering dan SMOTE

Lapisan ini menyusun satu baris pelatihan untuk setiap pasangan pengguna dan aturan asosiasi. Delapan fitur yang diekstrak adalah:

1. `total_prior_orders`, yaitu jumlah pesanan historis pengguna.
2. `avg_basket_size`, yaitu rata-rata jumlah produk per keranjang.
3. `avg_days_between`, yaitu rata-rata jeda hari antarpesanan.
4. `antecedent_rate`, yaitu seberapa sering pengguna membeli produk antecedent.
5. `rule_confidence` dan `rule_lift`, yaitu kekuatan aturan dari Apriori.
6. `interest_confidence` dan `interest_lift`, yaitu skor ketertarikan terhadap produk consequent, dihitung dari frekuensi antecedent dikali confidence atau lift.

Label `y` bernilai 1 bila pengguna benar-benar membeli produk consequent pada pesanan `train`, dan 0 bila tidak. Pengguna yang keranjangnya sudah dipakai menambang aturan tidak dimasukkan ke matriks ini. Dari pengguna yang tersisa, 12.000 pengguna diambil secara acak. Proporsi kelas pada matriks asli masih timpang, sekitar 9,9% kelas 1. SMOTE kemudian menambah sampel kelas minoritas sampai jumlahnya sama dengan kelas mayoritas. Hasil yang tersimpan di `outputs/layer3_smoted_features.csv` berukuran 519.142 baris dan 9 kolom: delapan fitur ditambah label.

### Layer 4 — Model Training dan Benchmarking

Lapisan ini membandingkan lima algoritma pada tugas yang sama: XGBoost, LightGBM, Random Forest, Support Vector Machine, dan Logistic Regression. Pembagian data latih dan data uji adalah 80:20 dengan `random_state=42`.

Agar skor sidang dapat dipertahankan, SMOTE tidak diterapkan pada data uji. Matriks asli dibagi lebih dulu, lalu SMOTE hanya menyeimbangkan data latih. Kelima model dilatih pada cuplikan latih yang sama sebesar 12.000 baris, karena SVM dengan perhitungan probabilitas tidak praktis pada ratusan ribu baris. Metrik yang dibandingkan adalah akurasi, precision macro, recall macro, F1 macro, dan AUC-ROC. XGBoost disimpan sebagai `models/xgboost_cross_sell_model.pkl` untuk dipakai pada lapisan integrasi. Tabel dan grafik pembandingan disimpan di folder `outputs/`.

### Layer 5 — Integration dan Real-time Inference

Lapisan ini mensimulasikan pemanggilan model di sistem yang sudah berjalan. Skrip `layer5_inference.py` memuat `xgboost_cross_sell_model.pkl`, lalu fungsi `predict_cross_sell` menerima delapan fitur pengguna yang sama dengan saat pelatihan. Keluaran fungsi adalah probabilitas kelas 1 dan keputusan biner. Bila probabilitas mencapai 0,50, sistem menjawab **Ya** dan merekomendasikan produk cross-selling tersebut. Bila di bawah ambang, sistem menjawab **Tidak**. Contoh eksekusi memakai satu baris sungguhan dari data uji Layer 4, sehingga simulasi ini dapat ditunjukkan langsung pada bimbingan.

Alur datanya sebagai berikut.

```text
dataset/*.csv
    -> notebooks/layer2_preprocessing_apriori.ipynb
    -> outputs/apriori_rules.csv
    -> notebooks/layer3_feature_engineering.ipynb
    -> outputs/layer3_smoted_features.csv
    -> notebooks/layer4_model_training.ipynb
    -> models/xgboost_cross_sell_model.pkl
    -> layer5_inference.py
```

Pada Layer 4, evaluasi tidak memakai berkas yang sudah di-SMOTE sebagai data uji. Matriks asli dibagi 80:20 terlebih dahulu. SMOTE hanya diterapkan pada data latih, sehingga data uji tetap berisi pengamatan sungguhan.

## Prasyarat Sistem

- macOS atau Linux dengan akses terminal
- Python **3.11** untuk lingkungan virtual proyek ini

`requirements.txt` mengunci NumPy 1.25.2 dan pandas 2.1.4. Kombinasi itu berjalan pada Python 3.10 dan 3.11. Python 3.12 tidak dipakai karena rilis NumPy tersebut belum mendukungnya. Python 3.9 juga tidak dipakai.

Interpreter yang digunakan untuk membuat `venv` pada mesin pengembangan:

```text
Python 3.11.16
```

## Panduan Instalasi

Masuk ke folder proyek, buat lingkungan virtual, lalu pasang pustaka.

```bash
cd "Cross Selling Retail"
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Di Windows, aktivasi memakai `venv\Scripts\activate`.

Pada Apple Silicon, LightGBM 4.1.0 tidak menyediakan wheel siap pakai, jadi pip menyusunnya dari kode sumber. Pasang CMake dan OpenMP terlebih dahulu, lalu beri tahu CMake 4 agar tetap menerima berkas build LightGBM:

```bash
brew install cmake libomp
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
```

Jika Anaconda sedang aktif di terminal yang sama, jalankan `conda deactivate` sebelum membuat `venv`. Pustaka OpenSSL Anaconda dapat bentrok dengan Python Homebrew.

Periksa instalasi dengan:

```bash
python -c "import pandas, xgboost, lightgbm, sklearn, mlxtend, imblearn; print('lingkungan siap')"
```

## Panduan Eksekusi

Jalankan lapisan berikut secara berurutan. Notebook mencari folder proyek secara otomatis selama `dataset/orders.csv` masih ada di atas direktori kerja, jadi Jupyter boleh dibuka dari folder proyek maupun dari `notebooks/`.

### Mode laptop dan mode server kampus

Ketiga notebook memakai flag yang sama di **sel kode pertama** (Configuration Cell):

```python
# ==========================================
# ENVIRONMENT CONFIGURATION
# Ubah menjadi True jika dijalankan di Server Kampus (RAM/CPU besar)
# Ubah menjadi False jika dijalankan di Laptop Lokal
# ==========================================
RUN_ON_SERVER = False
```

Nilai bawaan `False` adalah mode laptop. Untuk pindah ke server kampus, ubah `RUN_ON_SERVER` menjadi `True` pada ketiga berkas ini, lalu restart kernel dan jalankan semua sel dari atas:

| Notebook | Yang diubah |
|---|---|
| `notebooks/layer2_preprocessing_apriori.ipynb` | `SAMPLE_SIZE` dan `MIN_SUPPORT` |
| `notebooks/layer3_feature_engineering.ipynb` | `CHUNK_SIZE` |
| `notebooks/layer4_model_training.ipynb` | dictionary `XGB_PARAMS` untuk `XGBClassifier` |

Sel pertama mencetak salah satu baris berikut, supaya mode yang aktif terlihat sebelum proses berat dimulai:

```text
--> [INFO] Berjalan dalam mode SERVER (Parameter Maksimal)
--> [INFO] Berjalan dalam mode LAPTOP (Parameter Terbatas)
```

Perbedaan parameternya:

| Parameter | Laptop (`False`) | Server kampus (`True`) |
|---|---|---|
| Layer 2 `SAMPLE_SIZE` | `50000` pesanan prior pertama | `None`, seluruh pesanan `prior` di `orders.csv` (sekitar 3,2 juta dari 3,4 juta baris; pesanan `train` dan `test` tidak punya isi keranjang prior) |
| Layer 2 `MIN_SUPPORT` | `0.01` | `0.005`, supaya lebih banyak aturan tertangkap |
| Layer 3 `CHUNK_SIZE` | `1_000_000` baris per batch | `5_000_000` baris per batch, pembacaan lebih agresif pada RAM besar |
| Layer 4 XGBoost | `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `tree_method="auto"` | `n_estimators=1000`, `max_depth=8`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"` |

Flag di Layer 3 hanya mengubah ukuran batch saat membaca file. Jumlah pengguna, jendela 50.000 pesanan yang dikeluarkan dari matriks latih, dan SMOTE tidak berubah. Flag di Layer 4 hanya mengubah hyperparameter XGBoost. LightGBM, Random Forest, SVM, dan Logistic Regression tetap memakai pengaturan yang sama.

Gunakan nilai flag yang sama pada ketiga notebook dalam satu rangkaian eksekusi. Aturan Apriori, matriks fitur, dan model yang disimpan harus berasal dari mode yang sama. Setelah mengganti flag, jalankan ulang Layer 2, lalu Layer 3, lalu Layer 4. Model lama di `models/xgboost_cross_sell_model.pkl` baru terganti setelah Layer 4 selesai.

### Layer 1 — Data Input

Tidak ada notebook terpisah pada lapisan ini. Enam berkas mentah berada di `dataset/`:

| Berkas | Isi |
|---|---|
| `aisles.csv` | Master lorong |
| `departments.csv` | Master departemen |
| `products.csv` | Master produk |
| `orders.csv` | Header pesanan |
| `order_products__prior.csv` | Isi keranjang riwayat |
| `order_products__train.csv` | Isi keranjang pesanan berlabel |

Relasi utamanya: `products.aisle_id` ke `aisles`, `products.department_id` ke `departments`, serta `order_id` dan `product_id` pada kedua berkas isi keranjang. `user_id` hanya ada di `orders.csv`. Pesanan `eval_set = prior` terhubung ke berkas prior, pesanan `train` terhubung ke berkas train, dan pesanan `test` tidak memiliki isi keranjang.

### Layer 2 — Apriori

```bash
source venv/bin/activate
jupyter notebook notebooks/layer2_preprocessing_apriori.ipynb
```

Pada mode laptop, notebook mengambil 50.000 `order_id` prior pertama dan menjalankan Apriori dengan `min_support=0.01`. Pada mode server, seluruh pesanan prior dipakai dan `min_support=0.005`. `min_confidence` tetap 0,1. Hasilnya ditulis ke `outputs/apriori_rules.csv`.

### Layer 3 — Feature Engineering dan SMOTE

```bash
jupyter notebook notebooks/layer3_feature_engineering.ipynb
```

Satu baris pelatihan adalah satu pasangan pengguna dan aturan asosiasi. Fitur yang dibentuk:

- `total_prior_orders`
- `avg_basket_size`
- `avg_days_between`
- `antecedent_rate`
- `rule_confidence`
- `rule_lift`
- `interest_confidence` = `antecedent_rate × confidence`
- `interest_lift` = `antecedent_rate × lift`
- `y` = 1 jika consequent dibeli pada pesanan train

Pengguna yang prior-nya masuk 50.000 pesanan penambangan pertama dikeluarkan. Dari pengguna train yang tersisa, 12.000 pengguna diambil secara acak (`random_state=42`). File prior dibaca per `CHUNK_SIZE`: 1.000.000 baris di laptop, 5.000.000 baris di server. Matriks seimbang disimpan di `outputs/layer3_smoted_features.csv`.

### Layer 4 — Pelatihan dan Pembandingan Model

```bash
jupyter notebook notebooks/layer4_model_training.ipynb
```

Lima model dilatih pada cuplikan latih yang sama sebesar 12.000 baris, karena SVM dengan `probability=True` tidak praktis pada ratusan ribu baris. Data uji tetap 20% dari matriks asli. Metrik yang dibandingkan: akurasi, precision macro, recall macro, F1 macro, dan AUC-ROC. Hyperparameter XGBoost mengikuti `XGB_PARAMS`: pohon lebih sedikit di laptop, dan 1.000 pohon dengan `tree_method="hist"` di server.

Model yang disimpan untuk integrasi:

```text
models/xgboost_cross_sell_model.pkl
```

Tabel dan grafik pembandingan ditulis ke folder `outputs/`.

### Layer 5 — Inferensi

```bash
source venv/bin/activate
python layer5_inference.py
```

Skrip memuat model, menilai satu baris sungguhan dari data uji Layer 4, lalu mencetak probabilitas kelas 1 dan keputusan **Ya** atau **Tidak**. Ambang keputusannya 0,50.

Fungsi yang dapat dipanggil dari modul lain:

```python
from layer5_inference import load_model, predict_cross_sell

model = load_model()
hasil = predict_cross_sell(
    total_prior_orders=18,
    avg_basket_size=11.11,
    avg_days_between=5.41,
    antecedent_rate=0.056,
    rule_confidence=0.315,
    rule_lift=2.608,
    interest_confidence=0.018,
    interest_lift=0.145,
    model=model,
)
print(hasil["probability"], hasil["decision"])
```

## Struktur Repositori

```text
Cross Selling Retail/
├── dataset/                          # Enam CSV transaksi mentah
├── notebooks/
│   ├── layer2_preprocessing_apriori.ipynb
│   ├── layer3_feature_engineering.ipynb
│   └── layer4_model_training.ipynb
├── models/
│   └── xgboost_cross_sell_model.pkl  # Model integrasi
├── outputs/
│   ├── apriori_rules.csv
│   ├── layer3_smoted_features.csv
│   ├── layer4_model_comparison.csv
│   ├── layer4_model_comparison.png
│   └── layer4_xgboost_confusion_matrix.png
├── venv/                             # Lingkungan virtual Python 3.11
├── requirements.txt
├── layer5_inference.py
├── README.md
└── .gitignore
```

Folder `venv/` dan cache Python tidak perlu disalin ke repositori jarak jauh. Keduanya sudah tercantum di `.gitignore`.
