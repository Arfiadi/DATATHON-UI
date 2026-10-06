# Pondasi Pagi - Datathon Universitas Indonesia 2026

![AuraFresh Concept](https://img.shields.io/badge/AI-Multimodal%20Early%20Fusion-blue) ![Status](https://img.shields.io/badge/Status-Competition-success)

Repository ini merupakan hasil pengerjaan dari tim **Pondasi Pagi** untuk kompetisi **Datathon Universitas Indonesia (UI) 2026** (yang diselenggarakan oleh RISTEK Fasilkom UI). Penilaian kompetisi ini terbagi menjadi 3 komponen utama, yang seluruhnya didokumentasikan di dalam repository ini:

1. 📊 **Kaggle Task 1** - penyelesaian *Data Science Task* tahap pertama.
2. 📈 **Kaggle Task 2** - penyelesaian *Data Science Task* tahap kedua.
3. 📄 **Concept Paper AI** - Makalah konsep ide inovasi AI (*AuraFresh*).

---

## 🏆 Informasi Tim (Pondasi Pagi)
* **Anggota Tim:**
  1. Arfi Adi Nugroho
  2. Joshua Remedial Syeba
  3. Muhammad Zidan Akmal Nurrochman
* **Asal Institusi:** Departemen Informatika dan Teknik Komputer, Politeknik Elektronika Negeri Surabaya (PENS)

---

## 📂 Struktur Repository

Komponen file dan folder diatur berdasarkan tahapan kompetisi:

* **`KaggleTask-1/`** 
  Membahas kasus (problem) pertama dari Kaggle Task. 
* **`KaggleTask-2/`**
  Membahas kasus (problem) kedua dari Kaggle Task yang merupakan kasus berbeda dari Task 1. 
* **`Pondasi Pagi__ConceptPaper2.pdf` & `.docx`** 
  Merupakan dokumen lengkap untuk **Concept Paper AI**. Makalah ini menjelaskan gagasan utama kami mengenai sistem bernama **AuraFresh**.


---

## 💡 Ringkasan Concept Paper: AuraFresh
**AuraFresh** adalah sebuah sistem *Artificial Intelligence* (AI) prediktif berbasis **Early Fusion Multimodal** yang mengintegrasikan:
1. **Aliran Video CCTV (Visual Encoder):** Menggunakan model *ResNet-50* ter-prapelatih untuk mengekstraksi fitur morfologi dan perubahan warna komoditas pangan.
2. **Data Sensor Lingkungan (Sensor Encoder):** Menggunakan jaringan saraf *Bi-LSTM* untuk memproses dinamika temporal lingkungan (suhu, kelembaban, dan etilen).

Sistem ini ditujukan untuk memprediksi sisa umur simpan (*shelf-life*) komoditas pangan dalam hitungan jam guna mengubah manajemen gudang dari sistem *First-In-First-Out* (FIFO) secara pasif menjadi *First-Expired-First-Out* (FEFO) secara otomatis (tanpa perlu investasi hardware baru / Zero-CapEx).

