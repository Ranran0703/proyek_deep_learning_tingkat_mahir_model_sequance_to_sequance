# Submission Akhir: Menyelesaikan Masalah Deret Waktu Multivariate & Multi-Step (Prediksi Harga Bitcoin)

Proyek ini merupakan proyek **Submission Akhir** untuk kelas **Deep Learning untuk Pemodelan Data Deret Waktu (DLTM)** di **Dicoding**. Proyek ini berfokus pada peramalan (*forecasting*) multivariate dan multi-step dari data harga Bitcoin (`close` price) beserta indikator teknikal pendukung untuk horizon waktu 24 jam ke depan menggunakan arsitektur deep learning berbasis Recurrent Neural Networks (LSTM).

## 👥 Profil Pengembang
- **Nama:** Putri Maharani Fetra
- **Username Dicoding:** Putri Maharani Fetra
- **Gmail:** maharanifetra@gmail.com
- **Linkdln:** [Putri Maharani Fetra](https://www.linkedin.com/in/putri-maharani-fetra-5131a7243/)
- **Status Proyek:** Berhasil Memenuhi Semua Kriteria (Utama & Advanced)

---

## 📁 Struktur Direktori Proyek

Berikut adalah susunan file yang terdapat di dalam paket submission `Putri_Maharani_Fetra_Submission_Akhir_DLTM.zip`:

```text
Putri_Maharani_Fetra_Submission_Akhir_DLTM/
│
├── percobaanrani_Submission_akhir5.ipynb  # Notebook Jupyter utama alur eksperimen & visualisasi
├── requirements.txt                       # Daftar library Python dependencies 
├── model_baseline_LSTM (4).keras           # Bobot/Model Arsitektur Baseline LSTM Terlatih
├── model_seq2seq_LSTM (4).keras            # Bobot/Model Arsitektur Kustom Seq2Seq LSTM Terlatih
└── best_model_seq2seq_LSTM (1).keras       # Model terbaik hasil checkpoint (Kustom Seq2Seq LSTM)
```

---

## ⚙️ Alur Eksperimen & Tahapan Pemodelan

Aplikasi dibangun menggunakan rangkaian kode yang terbagi secara terstruktur ke dalam 9 rangkaian sel sukses di dalam notebook `percobaanrani_Submission_akhir5.ipynb`:

*   **Cell 1 (Data Acquisition & Cleaning):** Mengunduh otomatis dataset resmi pergerakan harga Bitcoin dari Dicoding. Melakukan penanganan tanggal bertipe campuran (`format='mixed'`), menyeragamkan nama kolom menjadi huruf kecil, dan menyetel indeks berbasis waktu (Date).
*   **Cell 2 (Exploratory Data Analysis - EDA):** Melakukan visualisasi heatmap untuk melihat korelasi antar-fitur. Fitur valid dengan korelasi optimal yang dipilih untuk pemodelan multivariate ini adalah `['close', 'volume usdt', 'rsi']`.
*   **Cell 3 (Feature Engineering & Train-Test Split):** Melakukan normalisasi data menggunakan `MinMaxScaler`. Data dibagi menjadi data latih (*Training Set*) dan data uji (*Test Set*) dengan rasio 80% : 20% dengan mempertahankan urutan waktu (sekuensial). Pembuatan window dirancang menggunakan window pengamatan sepanjang 24 jam ke belakang untuk memprediksi 24 jam ke depan.
*   **Cell 4 (Custom Components - Kriteria Advanced):**
    *   *Custom Layer:* Implementasi kustom `CustomDense`, `CustomDropout`, dan `CustomMultiHeadAttention` berbasis `tf.keras.layers.Layer`.
    *   *Custom Loss Function:* `HorizonWeightedMAE`, memberikan bobot penalti error yang lebih besar secara linear pada langkah prediksi yang semakin jauh di masa depan.
    *   *Custom Utility/Callback:* Kelas `CustomEarlyStopping` dan `CustomLRScheduler` kustom yang dirancang manual tanpa mewarisi subclass bawaan Keras.
*   **Cell 5 (Architecture Construction):** Membangun dua arsitektur model:
    *   *Model Baseline:* Menggunakan arsitektur Sequential dengan komponen standar LSTM.
    *   *Model Seq2Seq:* Menggunakan arsitektur Functional API berupa struktur Encoder-Decoder yang dipadukan dengan kustom Cross-Attention.
*   **Cell 6 & 7 (Training & Validation Loop):** Menjalankan proses training kedua model menggunakan *custom loop* (siklus pelatihan kustom), pemanfaatan *custom loss*, pelacakan performa per epoch, serta penyimpanan model terbaik ke format `.keras`.
*   **Cell 8 (Evaluation & Forecasting):** Menguji keandalan model pada data uji (*Test Set*) menggunakan metrik evaluasi Mean Absolute Error (MAE) dengan skenario prediksi autoregressive.
*   **Cell 9 (Comparison & Conclusion):** Komparasi singkat dan kesimpulan performa model untuk kenyamanan proses penilaian oleh Reviewer Dicoding.

---

## 📊 Hasil Komparasi Model

Model dievaluasi menggunakan metrik Mean Absolute Error (MAE) dengan skenario prediksi autoregressive (multi-step 24 jam ke depan):

1.  **Model Baseline (Sequential LSTM):** Menghasilkan prediksi linear yang cukup stabil untuk tren jangka pendek, namun cenderung melandai pada horizon waktu yang lebih jauh.
2.  **Model Seq2Seq LSTM (Encoder-Decoder + Attention):** Berhasil mengungguli model baseline. Pendekatan mekanisme kustom Cross-Attention dan struktur sekuensial terbukti jauh lebih adaptif dalam meminimalkan akumulasi error (*error accumulation*) pada prediksi jangka panjang (horizon 24 jam).

---

## 💻 Persyaratan Instalasi (Requirements)

Untuk menjalankan notebook dan memuat model ini secara lokal, pastikan Anda telah memasang Python (disarankan versi 3.10 ke atas) dan semua dependensi library yang tertera pada file `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Daftar Dependensi Utama:
*   `tensorflow` (Versi 3.x kompatibel Keras 3)
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `statsmodels`
*   `scikit-learn`

---

## 🚀 Cara Menjalankan Proyek

Langkah-langkah untuk menjalankan eksperimen dari awal:

1.  Ekstrak file zip proyek `Putri_Maharani_Fetra_Submission_Akhir_DLTM.zip`.
2.  Buka aplikasi Terminal, Command Prompt, atau Git Bash, lalu arahkan (*change directory*) ke folder hasil ekstrak tersebut:
    ```bash
    cd Putri_Maharani_Fetra_Submission_Akhir_DLTM
    ```
3.  Pasang semua dependensi library terlebih dahulu:
    ```bash
    pip install -r requirements.txt
    ```
4.  Jalankan server Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
5.  Pilih dan buka file `percobaanrani_Submission_akhir5.ipynb`.
6.  Jalankan seluruh sel kode secara berurutan mulai dari **Cell 1** sampai **Cell 9** (`Cell` -> `Run All`). Dataset akan diunduh secara otomatis dari URL cloud resmi yang sudah tertanam di dalam kode.
