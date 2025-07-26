# Analisis Trend Wisata Untuk Pengembangan Paket Wisata Terbaik

## 👩‍💻 Kontributor

- Haris Saefuloh / 714220061
- Mochammad Rayfan Aqbillah / 714220044
- Nida Sakina Aulia / 714220040
- Mariana Siregar / 714220068

## 📓 Deskripsi Kasus

Industri pariwisata di Indonesia menunjukkan perkembangan yang pesat dan memberikan kontribusi signifikan terhadap perekonomian nasional. Namun, dinamika tren wisata yang dipengaruhi oleh perubahan sosial, teknologi digital, dan preferensi wisatawan menuntut adaptasi cepat dari pelaku industri, khususnya agen perjalanan. Di era digital, wisatawan semakin bergantung pada ulasan online dan media sosial dalam merencanakan perjalanan, sehingga menghasilkan data besar yang dapat dimanfaatkan untuk menganalisis tren secara real-time. Pasca pandemi COVID-19, muncul preferensi baru terhadap wisata alam, ekowisata, serta wisata berbasis pengalaman seperti budaya dan kuliner. Agar tetap relevan dan kompetitif, agen perjalanan perlu melakukan analisis data yang mendalam guna memahami tren wisata terkini, preferensi pelanggan, dan destinasi yang sedang naik daun.

## 🔎 Sumber Dataset

- [Data Wisata Trenggalek](https://satudata.trenggalekkab.go.id/dataset/415/2024/data-kunjungan-destinasi-wisata)
- [Data Wisata Ancol](https://korporat.ancol.com/annual-report--29)
- [Data Ulasan Tempat Wisata](https://www.google.com/maps)

## 🧹 Langkah Preprocessing

- Menghapus baris yang tidak lengkap/memiliki nilai kosong
- Menghapus baris yang memiliki data berganda
- Mengubah teks menjadi lowercase
- Menghapus karakter non-ASCII
- Menghapus URL
- Menghapus mention (@) dan hastag (#)
- Menghapus simbol
- Menghapus angka pada data di kolom ulasan
- Menghapus Spasi berlebih
- Melakukan filter pada ulasan untuk menghapus ulasan yang tidak berbahasa Indonesia

## 🧠 Algoritma yang Digunakan

LSTM (Long Short-Term Memory)

## 📊 Evaluasi dan Hasil
Berikut adalah hasil evaluasi prediksi teratas untuk tempat wisata terbaik:

| Tempat Wisata               | MAE        | RMSE       | MAPE    |
|-----------------------------|------------|------------|---------|
| Taman Impian Ancol          | 1.120.861  | 1.222.905  | 11.62%  |
| Dufan Ancol                 | 142.578    | 150.419    | 6.58%   |
| SeaWorld Ancol              | 125.216    | 135.810    | 10.53%  |
| Pantai Karanggongso         | 740.62     | 740.62     | 0.19%   |
| Pantai Mutiara Trenggalek   | 116.12     | 116.12     | 0.03%   |


## 🚀 Cara Menjalankan
### 1. Jalankan di Google Colab (Direkomendasikan)

Klik tombol berikut untuk membuka notebook di Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 2. Import File Ini Ke Google Colab
[Notebook](https://github.com/Kelompok-1-Data-Mining/tugas-besar-datamining-kelompok1/tree/main/Report)

### 3. Jalankan Secara Lokal

✅ Persyaratan
Pastikan Anda sudah menginstall:
- Python
- pip

📦 Install Dependensi
```bash
pip install -r src/requirements.txt
```
▶️ Jalankan Secara Lokal
```bash
python src/main.py
```
## 📜 Lisensi

Proyek ini bersifat open-source dan bebas digunakan untuk edukasi dan pengembangan pribadi.
