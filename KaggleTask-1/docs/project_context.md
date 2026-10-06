# Project Context: Datathon Task 1

## Kompetisi
**Datathon Task 1: City Network Traffic Forecasting**
URL: [Kaggle Competition](https://www.kaggle.com/competitions/datathon-task-1)

## Tujuan Utama
Tantangan utama dari kompetisi ini adalah memprediksi kecepatan lalu lintas (dalam satuan km/jam) di seluruh jaringan jalan raya kota pada masa mendatang. Kita diminta untuk melakukan peramalan (*forecasting*) untuk **1.260 segmen jalan yang saling terhubung** pada tiga horizon waktu yang berbeda:
* **+20 menit** (`h5`) - ekuivalen dengan 5 timestep ke depan (asumsi 1 timestep = 4 menit)
* **+40 menit** (`h10`) - ekuivalen dengan 10 timestep ke depan
* **+60 menit** (`h15`) - ekuivalen dengan 15 timestep ke depan

## Data yang Tersedia (Input Signal)
Dataset tersimpan di dalam direktori `Dataset/` yang mencakup tiga jenis sumber informasi/sinyal:
1. **Speed History (Riwayat Kecepatan):** Data 15 langkah waktu ke belakang (setara dengan 1 jam riwayat kecepatan terakhir) untuk seluruh 1.260 segmen jalan. Disimpan dalam bentuk *numpy array* (`.npy`).
2. **Event Text (Teks Kejadian):** Aliran teks informasi dalam bahasa alami (*natural-language*) yang melaporkan kejadian nyata di lapangan (kecelakaan, penutupan jalan, dsb.). Terdapat dalam format JSON.
3. **Road Network (Jaringan Jalan):** Sebuah *adjacency matrix* berukuran 1.260 x 1.260 (`static/matrix.npy`) yang menggambarkan konektivitas segmen jalan, dilengkapi dengan geometri dan metadata tambahan (`static/Roads1260.json`).

## Metrik Evaluasi
Performa model dinilai menggunakan **Mean Squared Error (MSE)** antara kecepatan prediksi dengan data riil (*ground truth*).
Rumus: `MSE = (1/N) * Σ(ŷ - y)²`

## Format Pengiriman (Submission)
Hasil prediksi harus disimpan dalam file `submission.csv` dengan format ID spesifik per baris: `test_{sample}_h{horizon}_r{road}`.
Contoh:
```csv
id,speed
test_00000_h5_r0,44.0
test_00000_h5_r1,73.0
```

## Setup Lingkungan
- Workspace: `D:\ARFI\Lomba\datathon-ui\Datathon-UI`
- Dataset Path: `D:\ARFI\Lomba\datathon-ui\Datathon-UI\Dataset`
