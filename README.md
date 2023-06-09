# Laporan Proyek Machine Learning - Nadya Novalina

## Domain Proyek

Prediksi Harga Saham menggunakan Long Short-Term Memory (LSTM) dmenggunakan data dari IHSG perusahaan Unilever Indonesia.

## Latar Belakang
Perkembangan pasar saham dan investasi semakin menarik minat banyak orang. Salah satu tantangan yang dihadapi oleh para investor adalah memprediksi pergerakan harga saham di masa depan untuk mengambil keputusan investasi yang lebih baik. Dalam proyek ini, fokus diberikan pada prediksi harga saham perusahaan Unilever Indonesia menggunakan metode LSTM (Long Short-Term Memory) berdasarkan data historis dari IHGS (Indonesia Stock Exchange). Tujuan utama proyek ini adalah membantu investor dalam mengoptimalkan keputusan investasi mereka dengan memberikan prediksi harga saham yang lebih akurat.

## Business Understanding
Proyek ini bertujuan untuk mengembangkan model prediksi harga saham menggunakan metode LSTM dengan data historis perusahaan Unilever Indonesia dari IHGS. Dalam hal ini, model akan dilatih menggunakan data historis harga saham Unilever Indonesia untuk memprediksi harga saham di masa depan pada data uji. Dengan menggunakan model prediksi yang akurat, investor dapat memperoleh wawasan yang lebih baik tentang pergerakan harga saham Unilever Indonesia, sehingga dapat mengambil keputusan investasi yang lebih cerdas dan meningkatkan potensi keuntungan mereka.

Bagian laporan ini mencakup:

### Problem Statements
Investor seringkali sulit memprediksi harga saham (contoh: Unilever Indonesia) di masa depan, yang membuat mereka kesulitan dalam mengambil keputusan investasi. Pasar saham seringkali berfluktuasi dengan cepat dan banyak faktor yang memengaruhi harga saham, sehingga sulit untuk membuat prediksi yang akurat. Oleh karena itu, dibutuhkan pendekatan yang dapat membantu investor dalam memprediksi harga saham Unilever Indonesia dengan lebih baik.

### Goals

- Mengembangkan model prediksi harga saham yang akurat menggunakan metode LSTM.
- Membantu investor dalam membuat keputusan investasi yang lebih baik berdasarkan prediksi harga saham yang akurat.
### Solution statements
- Melakukan tuning parameter pada model LSTM, seperti ukuran lapisan LSTM, dropout rate, dan jumlah neuron pada hidden layer.
- Mengimplementasikan model LSTM yang lebih kompleks, dengan menambahkan lapisan LSTM tambahan, peningkatan jumlah neuron pada hidden layer.
- Melakukan preprocessing data dengan menggunakan MinMaxScaler untuk melakukan Feature Scaling pada data harga saham. Dengan menggunakan MinMaxScaler, skala data harga saham dapat disesuaikan sehingga nilainya berada dalam rentang 0 hingga 1. Hal ini membantu menghindari perbedaan skala yang signifikan antara fitur-fitur dan memastikan pengaruh yang seimbang terhadap model LSTM.

## Data Understanding
Dataset yang digunakan berasal dari  [Indonesia Stock Dataset](https://www.kaggle.com/datasets/muamkh/ihsgstockdata). Dataset ini berisi data historis saham yang terdaftar di IHSG dengan rentang waktu per menit, per jam, dan per hari. Sumber dataset diambil dari data publik Yahoo Finance dan website IDX yang terdaftar pada tab metadata. Proyek ini menggunkaan dataset per hari yang diambil dari 16 April 2001 hingga 6 Januari 2023.

### Variabel-variabel pada IHGS-UNVR dataset adalah sebagai berikut:
- timestamp = Tanggal dan waktu transaksi saham
- open = Harga pembukaan
- low = Harga terendah dalam rentang waktu
- high = Harga tertinggi dalam rentang waktu
- close = Harga penutupan
- volume = Total volume yang diperdagangkan dalam rentang waktu tersebu

## Data Preparation
Data historis harga saham Unilever Indonesia dari IHGS diimpor menggunakan library Pandas dan disajikan dalam bentuk DataFrame. Selanjutnya, dilakukan pemahaman terhadap data tersebut dengan menampilkan grafik harga saham Unilever Indonesia dari waktu ke waktu. Grafik ini membantu dalam melihat tren dan pola pergerakan harga saham perusahaan.

Setelah itu, dilakukan Feature Scaling menggunakan MinMaxScaler untuk menormalkan data harga saham sebelum dimasukkan ke dalam model LSTM. Data juga dibagi menjadi data latih (train) dan data uji (test) dengan perbandingan 80:20 menggunakan fungsi split_data.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

**Referensi:**
- [Unilever Indonesia](https://www.unilever.co.id/our-company/)
- [Unilever Indonesia Tbk. - Company Background](https://stockbit.com/symbol/UNVR) 
- [Time-Series Forecasting: Predicting Stock Prices Using An LSTM Model](https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f) 
- [Stock Price Prediction using Machine Learning with Source Code](https://www.projectpro.io/article/stock-price-prediction-using-machine-learning-project/571)
