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

- **Taman Impian Ancol**:  
  MAE = 1.120.861, RMSE = 1.222.905, MAPE = 11.62%

- **Dufan Ancol**:  
  MAE = 142.578, RMSE = 150.419, MAPE = 6.58%

- **SeaWorld Ancol**:  
  MAE = 125.216, RMSE = 135.810, MAPE = 10.53%

- **Pantai Karanggongso**:  
  MAE = 740.62, RMSE = 740.62, MAPE = 0.19%

- **Pantai Mutiara Trenggalek**:  
  MAE = 116.12, RMSE = 116.12, MAPE = 0.03%


## 🚀 Cara Menjalankan



## 📜 Lisensi

Proyek ini bersifat open-source dan bebas digunakan untuk edukasi dan pengembangan pribadi.
