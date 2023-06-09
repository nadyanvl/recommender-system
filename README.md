# Laporan Proyek Machine Learning - Nadya Novalina

## Domain Proyek

Prediksi Harga Saham menggunakan Long Short-Term Memory (LSTM) dmenggunakan data dari IHSG perusahaan Unilever Indonesia.

## Latar Belakang
Perkembangan pasar saham dan investasi semakin menarik minat banyak orang. Salah satu tantangan yang dihadapi oleh para investor adalah memprediksi pergerakan harga saham di masa depan untuk mengambil keputusan investasi yang lebih baik. Dalam proyek ini, fokus diberikan pada prediksi harga saham perusahaan Unilever Indonesia menggunakan metode LSTM (Long Short-Term Memory) berdasarkan data historis dari IHGS (Indonesia Stock Exchange). Tujuan utama proyek ini adalah membantu investor dalam mengoptimalkan keputusan investasi mereka dengan memberikan prediksi harga saham yang lebih akurat.

## Business Understanding
Proyek ini bertujuan untuk mengembangkan model prediksi harga saham menggunakan metode LSTM dengan data historis perusahaan Unilever Indonesia dari IHGS. Dalam hal ini, model akan dilatih menggunakan data historis harga saham Unilever Indonesia untuk memprediksi harga saham di masa depan pada data uji. Dengan menggunakan model prediksi yang akurat, investor dapat memperoleh wawasan yang lebih baik tentang pergerakan harga saham Unilever Indonesia, sehingga dapat mengambil keputusan investasi yang lebih cerdas dan meningkatkan potensi keuntungan mereka.

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
Berikut grafik visualisasi dataset:

![image](https://github.com/nadyanvl/test/assets/42887151/e8b20b25-2884-4342-b4c8-12cbd2968400)

Setelah itu, dilakukan Feature Scaling menggunakan MinMaxScaler untuk menormalkan data harga saham sebelum dimasukkan ke dalam model LSTM. Data juga dibagi menjadi data latih (train) dan data uji (test) dengan perbandingan 80:20 menggunakan fungsi split_data.

## Modeling
Proyek ini menggunakan model LSTM (Long Short-Term Memory) untuk memprediksi harga saham Unilever Indonesia. Model ini dikembangkan menggunakan TensorFlow dengan Sequential API. LSTM adalah jenis arsitektur jaringan saraf rekurensi (RNN) yang dirancang khusus untuk memproses data dengan urutan waktu, seperti data harga saham. LSTM memiliki keunggulan dalam menangani masalah yang terkait dengan memori jangka panjang, yaitu kemampuannya untuk mengingat dan menggunakan informasi masa lalu dalam memprediksi masa depan.

Model LSTM yang dikembangkan memiliki beberapa layer, yaitu layer LSTM, Dropout, dan Dense. Ukuran lapisan LSTM, dropout rate, dan jumlah neuron pada hidden layer diubah untuk mencari konfigurasi yang optimal. Hal ini bertujuan untuk meningkatkan performa model dalam memprediksi harga saham dengan lebih akurat.
Detail Model sebagai berikut:

![image](https://github.com/nadyanvl/test/assets/42887151/be78a116-9aec-4127-b0c1-b8c3f0fcf688)

Proses pelatihan model dilakukan dengan mengkompilasi model menggunakan optimizer Adam dengan learning rate 0.001 dan loss function Huber. Selama pelatihan, digunakan callbacks EarlyStopping untuk menghentikan pelatihan jika mae pada data validasi sudah cukup kecil.

## Evaluation
Evaluasi model dilakukan menggunakan data uji. Prediksi harga saham dilakukan pada data uji dan hasilnya dibandingkan dengan harga saham aktual, serta dilakukan plotting grafik harga saham aktual dan harga saham yang diprediksi.

![image](https://github.com/nadyanvl/test/assets/42887151/2840e2c2-7622-4011-9d40-e91eb562f44e)

Beberapa metrik evaluasi digunakan untuk mengukur performa model prediksi harga saham seperti Mean Absolute Error (MAE), Root Mean Square Error (RMSE), dan Mean Absolute Percentage Error (MAPE). 

- Mean Absolute Error (MAE): selisih absolut rata-rata antara nilai yang diprediksi oleh model (prediksi dalam sampel satu langkah ke depan) dan data historis yang diamati.
- Root Mean Square Error (RMSE): Akar kuadrat dari MSE. MSE adalah jumlah selisih kuadrat antara nilai yang diprediksi oleh model dan nilai yang diamati, yang kemudian dibagi dengan jumlah titik data historis dikurangi jumlah parameter dalam model. 
- MAPE (Mean Absolute Percentage Error): Selisih persentase absolut rata-rata antara nilai yang diprediksi oleh model dan nilai data yang diamati.

Metrik evaluasi ini memberikan gambaran sejauh mana model dapat memprediksi harga saham Unilever Indonesia dengan akurat. Semakin rendah nilai MAE dan RMSE, semakin baik performa model dalam melakukan prediksi yang akurat. 

**Referensi:**
- [Unilever Indonesia](https://www.unilever.co.id/our-company/)
- [Unilever Indonesia Tbk. - Company Background](https://stockbit.com/symbol/UNVR) 
- [Time-Series Forecasting: Predicting Stock Prices Using An LSTM Model](https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f) 
- [Stock Price Prediction using Machine Learning with Source Code](https://www.projectpro.io/article/stock-price-prediction-using-machine-learning-project/571)
