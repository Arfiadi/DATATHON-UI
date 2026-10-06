# Comprehensive EDA Interpretation & Modeling Strategy
**Goal:** Optimize MSE for Datathon Task 1: City Network Traffic Forecasting (Horizons: +20, +40, +60 mins untuk 1260 jalan).

Dokumen ini merupakan hasil kompilasi dan interpretasi final dari seluruh proses *Exploratory Data Analysis* (EDA) awal maupun *Advanced EDA*. Semua poin di bawah ini dirancang secara eksklusif untuk menghasilkan keputusan *Feature Engineering* yang berdampak langsung pada perbaikan metrik Mean Squared Error (MSE).

---

## 1. Time-Series Dynamics & Data Integrity
**Temuan Utama:**
- **Zero-Speed Anomalies:** Terdapat lonjakan ekstrim data bernilai 0 km/h pada Bulan 2 (~16.7%) dibandingkan Bulan 1 (~1%).
- **Distribusi Target (Skewness):** Distribusi kecepatan historis (*non-zero*) memiliki skor *skewness* **-0.157** (hampir simetris sempurna).
- **Stasioneritas & PACF:** Uji *Augmented Dickey-Fuller* (ADF) menghasilkan *p-value* **1.63e-21** (sangat stasioner). Plot *Partial Autocorrelation* (PACF) menunjukkan bahwa korelasi paling signifikan terkonsentrasi di *lag* 1 hingga 15.

**Interpretasi & Keputusan Pemodelan (MSE-Driven):**
- **Penanganan Anomali:** Data 0 km/h di Bulan 2 akan mengacaukan MSE jika dianggap sebagai kecepatan aktual. **Tindakan:** Buat fitur *boolean flag* `is_sensor_error` atau imputasi nilai 0 tersebut menggunakan nilai rata-rata historis terdekat.
- **Fungsi Objektif:** Distribusi target tidak miring (*not skewed*). **Tindakan:** Hindari *log-transform* (seperti MSLE/RMSLE) yang menambah kompleksitas *inverse-transform*. Lakukan *training* model secara langsung menggunakan **L2 Loss (MSE)**.
- **Lookback Window:** Karena data stasioner, model tidak butuh rentang masa lalu yang terlampau jauh. **Tindakan:** *Window* historis **15-step** (60 menit ke belakang) sudah secara matematis mencukupi. Menambah lag > 15 hanya akan menambah dimensi fitur, memperlambat *training*, dan meningkatkan risiko *overfitting*.

---

## 2. Spatio-Temporal Graph & Network Topology
**Temuan Utama:**
- **Critical Bottlenecks:** Analisis *Betweenness Centrality* mendeteksi 10 segmen jalan paling kritis (mis. Majialou Bridge, Jingkai Road) yang bertindak sebagai "urat nadi" lalu lintas.
- **Lagged Spatial Cross-Correlation:** Rambatan kemacetan dari satu jalan ke jalan yang terhubung (*neighboring edges*) terjadi secara **asinkron** (bertahap), bukan instan.

**Interpretasi & Keputusan Pemodelan (MSE-Driven):**
- **Global Features:** Kemacetan di jalan kritis berdampak luas. **Tindakan:** Ekstrak kecepatan di 10 jalan paling kritis ini dan jadikan *global features* ke dalam prediksi untuk *semua* segmen jalan lainnya.
- **Time-Shifted Spatial Features:** Memasukkan rata-rata kecepatan spasial pada waktu $t$ kurang optimal. **Tindakan:** Ekstrak fitur agregasi tetangga pada waktu $t-1, t-2, \dots, t-4$. Memodelkan "rambatan tertunda" ini adalah kunci utama menurunkan MSE secara drastis pada model *Tree-based* (seperti LightGBM).

---

## 3. External Factors: Temporal Cycles & NLP Incidents
**Temuan Utama:**
- **Siklus Musiman:** Pola kepadatan jam-jaman sangat dipengaruhi oleh hari (pola *Weekday* vs *Weekend* sangat kontras).
- **Dampak Insiden:** Pemetaan laporan NLP (*English*) ke metadata jalan (*Chinese*) membuktikan bahwa insiden menyebabkan *speed drop* drastis hingga **-22%** (contoh kasus di Caihuying South Road).

**Interpretasi & Keputusan Pemodelan (MSE-Driven):**
- **Cyclical Encoding:** Angka jam `23` dan `00` terlihat terpisah jauh oleh model, padahal secara siklus hanya selisih 1 menit. **Tindakan:** Angka Jam (0-23) dan Hari (0-6) wajib diekstrak menggunakan transformasi matematika **Sine/Cosine**. Tanpa ini, akan terjadi lonjakan error (*MSE spike*) pada prediksi yang melintasi tengah malam.
- **Categorical NLP Features:** Kejadian tak terduga (*shock*) adalah penyumbang error terbesar. **Tindakan:** Ekstrak fitur dari teks mentah menggunakan Regex/TF-IDF menjadi fitur kategorik (misal: `incident_severity`: Low/Medium/High) atau durasi sejak kecelakaan (`time_since_incident`).

---

## 4. Kesimpulan Rekomendasi Arsitektur
Untuk mencapai posisi teratas di *leaderboard*, bangun arsitektur **LightGBM / XGBoost** dengan *Feature Engineering* agresif berdasarkan EDA ini, atau gunakan **Spatio-Temporal GNN (STGNN)**. 

Apapun modelnya, pastikan matriks *input* memiliki set fitur berikut:
1. `lag_1` s/d `lag_15` *(Self-Historical Speeds)*
2. `spatial_mean_lag_1`, `spatial_mean_lag_2` *(Time-Shifted Neighbor Speeds)*
3. `hour_sin`, `hour_cos`, `day_sin`, `day_cos` *(Cyclical Time)*
4. `global_critical_road_1_speed`, `global_critical_road_2_speed` *(Bottleneck States)*
5. `is_zero_anomaly_flag` *(Data Integrity)*
6. `incident_severity_encoded` *(NLP Context)*
