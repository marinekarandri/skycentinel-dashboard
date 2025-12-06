# SkySentinel: Machine Learning Models for Air Quality Forecasting

**SkySentinel** adalah platform dashboard interaktif untuk memantau kualitas udara historis di DKI Jakarta dan memprediksi konsentrasi polutan PM2.5 di masa depan menggunakan Machine Learning (Random Forest).

Project ini menggabungkan analisis data eksploratif (EDA) mendalam dengan forecasting time-series untuk membantu visualisasi tren polusi udara di berbagai stasiun pemantauan (SPKU) Jakarta.

---

## Fitur Utama

### 1. Monitoring Historis
- **Multi-Polutan**: Analisis tren PM2.5, PM10, SO2, CO, O3, dan NO2.
- **Analisis Waktu**: Pola musiman (bulanan) dan pola mingguan (hari kerja vs akhir pekan).
- **Peringkat Stasiun**: Identifikasi stasiun dengan kualitas udara terbaik dan terburuk.
- **Heatmap Risiko**: Visualisasi intensitas polusi berdasarkan Tahun vs Bulan.

### 2. AI Forecasting (Machine Learning)
- **Model**: Random Forest Regressor dengan Hyperparameter Tuning.
- **Fitur**: Menggunakan *Lag Features* (14 hari terakhir) untuk menangkap pola temporal.
- **Metode Prediksi**: *Recursive Forecasting* untuk memprediksi hingga 7 hari ke depan.
- **Evaluasi**: Dilengkapi metrik MAE, RMSE, dan MAPE untuk transparansi akurasi.
- **Multi-Station Support**: Model dilatih secara spesifik untuk setiap stasiun (Kebon Jeruk, Bundaran HI, Kelapa Gading, dll) agar prediksi lebih akurat sesuai karakteristik lokal.

---

## Teknologi yang Digunakan

- **Bahasa**: Python
- **Dashboard UI**: [Streamlit](https://streamlit.io/)
- **Data Processing**: Pandas, NumPy
- **Visualisasi**: Plotly Express, Plotly Graph Objects
- **Machine Learning**: Scikit-Learn (RandomForest, RandomizedSearchCV, TimeSeriesSplit)
- **Model Serialization**: Joblib

---

## Struktur Folder Proyek

Pastikan struktur folder Anda terlihat seperti ini agar aplikasi berjalan lancar:

```
skysentinel/
│
├── data/
│   └── df\_clean.csv          \# Dataset hasil cleaning
│
├── models/                   \# Folder penyimpanan model .pkl & metadata .json
│   ├── model\_bundaran\_hi.pkl
│   ├── meta\_bundaran\_hi.json
│   ├── model\_kelapa\_gading.pkl
│   └── ... (model stasiun lainnya)
│
├── notebooks/
│   └── SkySentinel\_Training.ipynb  \# Notebook untuk training & tuning model
│
├── main.py                   \# File utama aplikasi Streamlit
├── requirements.txt          \# Daftar library python
└── README.md                 \# Dokumentasi proyek

````
---
## Cara Menjalankan (Installation)

Ikuti langkah-langkah ini untuk menjalankan dashboard di komputer lokal Anda:

### 1. Clone Repository
```bash
git clone [https://github.com/username-anda/skysentinel.git](https://github.com/username-anda/skysentinel.git)
cd skysentinel
````

### 2\. Buat Virtual Environment (Opsional tapi Disarankan)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Training Model (Jika folder `models/` kosong)

Jalankan script training di Jupyter Notebook (`notebooks/SkySentinel_Training.ipynb`) atau jika Anda punya script python terpisah:

```bash
python train_models.py
```

*Script ini akan menghasilkan file `.pkl` dan `.json` di dalam folder `models/`.*

### 5\. Jalankan Streamlit

```bash
streamlit run main.py
```

Buka browser dan akses alamat yang muncul (biasanya `http://localhost:8501`).

-----

## Metodologi Model AI

Model forecasting dibangun dengan tahapan berikut:

1.  **Preprocessing**:
      - Interpolasi data hilang (Missing Values).
      - Resampling data ke frekuensi harian (Daily Average).
2.  **Feature Engineering**:
      - Pembuatan *Lag Features* (t-1 sampai t-14) untuk menangkap autokorelasi data.
3.  **Training & Tuning**:
      - Menggunakan **Random Forest Regressor**.
      - **Hyperparameter Tuning** menggunakan `RandomizedSearchCV` dengan `TimeSeriesSplit` (Cross-Validation anti-bocor data masa depan).
      - Parameter yang di-tuning: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
4.  **Evaluasi**:
      - Model dinilai berdasarkan Mean Absolute Percentage Error (**MAPE**) untuk kemudahan interpretasi.

-----

## Kontribusi

Kontribusi selalu diterima\! Silakan buat **Pull Request** atau buka **Issue** jika Anda menemukan bug atau punya ide fitur baru.

1.  Fork Project ini
2.  Buat Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit Perubahan (`git commit -m 'Add some AmazingFeature'`)
4.  Push ke Branch (`git push origin feature/AmazingFeature`)
5.  Open Pull Request

-----

**Dikembangkan dengan ❤️ untuk Jakarta yang lebih sehat.**
