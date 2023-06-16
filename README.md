# Laporan Proyek Machine Learning (Predictive Analytics) - Nadya Novalina
## Judul: Prediksi Harga Saham Unilever Indonesia menggunakan LSTM
## Domain Proyek
Perkembangan pasar saham dan investasi semakin menarik minat banyak orang. Salah satu tantangan yang dihadapi oleh para investor adalah memprediksi pergerakan harga saham di masa depan untuk mengambil keputusan investasi yang lebih baik. Dalam proyek ini, dilakukan prediksi harga saham perusahaan Unilever Indonesia menggunakan metode LSTM (Long Short-Term Memory) berdasarkan data historis yang diperoleh dari Yahoo Finance open dataset. Tujuan utama proyek ini adalah membantu investor dalam mengoptimalkan keputusan investasi mereka dengan memberikan prediksi harga saham yang lebih akurat.

LSTM digunakan dalam proyek ini karena dapat mengatasi data time series yang kompleks, seperti harga saham, dan mampu menangkap pola dan tren jangka panjang dalam data historis.
Penelitian dalam beberapa tahun terakhir telah menunjukkan bahwa model jaringan LSTM sangat efektif dalam prediksi harga saham. Seperti pada penelitian ini, menggunakan LSTM untuk prediksi saham dan mendapatkan nilai sebesar.

Dengan menggunakan LSTM, proyek ini bertujuan untuk memberikan prediksi harga saham Unilever Indonesia yang lebih akurat, terutama dalam jangka waktu yang lebih panjang. Hal ini akan membantu investor dalam pengambilan keputusan investasi yang lebih baik dan mengoptimalkan hasil investasi mereka

## Business Understanding
Proyek ini bertujuan untuk mengembangkan model prediksi harga saham menggunakan metode LSTM dengan data historis perusahaan Unilever Indonesia dari IHGS. Dalam hal ini, model akan dilatih menggunakan data historis harga saham Unilever Indonesia untuk memprediksi harga saham di masa depan pada data uji. Dengan menggunakan model prediksi yang akurat, investor dapat memperoleh wawasan yang lebih baik tentang pergerakan harga saham Unilever Indonesia, sehingga dapat mengambil keputusan investasi yang lebih cerdas dan meningkatkan potensi keuntungan mereka.

### Problem Statements
Investor seringkali sulit memprediksi harga saham (contoh: Unilever Indonesia) di masa depan, yang membuat mereka kesulitan dalam mengambil keputusan investasi. Pasar saham seringkali berfluktuasi dengan cepat dan banyak faktor yang memengaruhi harga saham, sehingga sulit untuk membuat prediksi yang akurat. Oleh karena itu, dibutuhkan pendekatan yang dapat membantu investor dalam memprediksi harga saham Unilever Indonesia dengan lebih baik.

### Goals
- Mengembangkan model prediksi harga saham yang akurat menggunakan metode LSTM.
- Membantu investor dalam membuat keputusan investasi yang lebih baik berdasarkan prediksi harga saham yang akurat.
- 
### Solution statements
- Melakukan tuning parameter pada model LSTM, seperti ukuran lapisan LSTM, dropout rate, dan jumlah neuron pada hidden layer.
- Mengimplementasikan model LSTM yang lebih kompleks, dengan menambahkan lapisan LSTM tambahan, peningkatan jumlah neuron pada hidden layer.
- Melakukan preprocessing data dengan menggunakan MinMaxScaler untuk melakukan Feature Scaling pada data harga saham. Dengan menggunakan MinMaxScaler, skala data harga saham dapat disesuaikan sehingga nilainya berada dalam rentang 0 hingga 1. Hal ini membantu menghindari perbedaan skala yang signifikan antara fitur-fitur dan memastikan pengaruh yang seimbang terhadap model LSTM.

## Data Understanding
Dataset yang digunakan berasal dari  [Yahoo Finance Dataset](https://finance.yahoo.com/quote/UNVR.JK?p=UNVR.JK&.tsrc=fin-srch). Dataset ini berisi data historis harian saham Unilever Indonesia dari 3 Januari 2005 hingga 9 Juni 2023. Dataset terdiri dari 4563 baris data dan 7 kolom yaitu: Date, Open, Low, High, Close, Adj Close, dan Volume.

### Detail variabel-variabel pada UNVR dataset adalah sebagai berikut:
- Date = Tanggal dan waktu transaksi saham
- Open = Harga pembukaan
- Low = Harga terendah dalam rentang waktu
- High = Harga tertinggi dalam rentang waktu
- Close = Harga penutupan
- Adj Close = Harga penutupan setelah penyesuaian untuk semua pemisahan dan pembagian dividen yang berlaku
- Volume = Total volume yang diperdagangkan dalam rentang waktu tersebu

## Data Preparation
Data historis harga saham Unilever Indonesia dari IHGS diimpor menggunakan library Pandas dan disajikan dalam bentuk DataFrame. Selanjutnya, dilakukan pemahaman terhadap data tersebut dengan menampilkan grafik harga saham Unilever Indonesia dari waktu ke waktu. Grafik ini membantu dalam melihat tren dan pola pergerakan harga saham perusahaan.
Berikut grafik visualisasi dataset:

![image](https://github.com/nadyanvl/StockPricePrediction/blob/main/assets/Grafik%20UNVR.png)

Setelah itu, dilakukan Feature Scaling menggunakan MinMaxScaler untuk menormalkan data harga saham sebelum dimasukkan ke dalam model LSTM. 

Data juga dibagi menjadi data latih (train) dan data uji (test) dengan perbandingan 80:20 menggunakan fungsi split_data. Dengan menggunakan 80% data sebagai data latih, model dapat belajar dari sebagian besar pola dan tren dalam data yang tersedia. Hal ini membantu dalam membangun model yang lebih mampu melakukan generalisasi pada data baru. Dan dengan menyisihkan 20% data sebagai data uji, kita memiliki set data yang independen untuk menguji kinerja model yang telah dilatih. Data uji yang tidak digunakan dalam proses pelatihan memberikan ukuran yang lebih objektif tentang seberapa baik model dapat menggeneralisasi pada data yang belum pernah dilihat sebelumnya dan dapat mengidentifikasi apakah model memiliki masalah overfitting atau tidak.

## Modeling
Proyek ini menggunakan model LSTM (Long Short-Term Memory) untuk memprediksi harga saham Unilever Indonesia. Model ini dikembangkan menggunakan TensorFlow dengan Sequential API. LSTM adalah jenis arsitektur jaringan saraf rekurensi (RNN) yang secara khusus dirancang untuk memproses data berurutan, seperti data harga saham. Keunggulan LSTM terletak pada kemampuannya dalam menangani masalah memori jangka panjang, di mana ia dapat mengingat dan menggunakan informasi masa lalu untuk memprediksi masa depan. Dalam LSTM, terdapat sel memori yang menggantikan neuron tradisional di lapisan tersembunyi jaringan. Dengan adanya sel memori ini, jaringan mampu menghubungkan informasi jarak jauh dalam urutan waktu, memungkinkan pemahaman yang dinamis terhadap struktur data dari waktu ke waktu, serta memberikan kemampuan prediksi yang lebih akurat.

Model LSTM yang dikembangkan memiliki beberapa layer, yaitu layer LSTM, Dropout, dan Dense. Ukuran lapisan LSTM, dropout rate, dan jumlah neuron pada hidden layer diubah untuk mencari konfigurasi yang optimal. Hal ini bertujuan untuk meningkatkan performa model dalam memprediksi harga saham dengan lebih akurat.
Detail Model sebagai berikut:

![image](https://github.com/nadyanvl/StockPricePrediction/blob/main/assets/Model%20UNVR.png)

Proses pelatihan model dilakukan dengan mengkompilasi model menggunakan optimizer Adam dengan learning rate 0.001 dan loss function Huber. Selama pelatihan, digunakan callbacks EarlyStopping untuk menghentikan pelatihan jika mae pada data validasi sudah cukup kecil.

## Evaluation
Evaluasi model dilakukan menggunakan data uji. Prediksi harga saham dilakukan pada data uji dan hasilnya dibandingkan dengan harga saham aktual, serta dilakukan plotting grafik harga saham aktual dan harga saham yang diprediksi.

![image](https://github.com/nadyanvl/StockPricePrediction/blob/main/assets/Prediksi%20UNVR.png)

Beberapa metrik evaluasi digunakan untuk mengukur performa model prediksi harga saham seperti Mean Absolute Error (MAE), Root Mean Square Error (RMSE), dan Mean Absolute Percentage Error (MAPE). 

- Mean Absolute Error (MAE): selisih absolut rata-rata antara nilai yang diprediksi oleh model (prediksi dalam sampel satu langkah ke depan) dan data historis yang diamati.
- Root Mean Square Error (RMSE): Akar kuadrat dari MSE. MSE adalah jumlah selisih kuadrat antara nilai yang diprediksi oleh model dan nilai yang diamati, yang kemudian dibagi dengan jumlah titik data historis dikurangi jumlah parameter dalam model. 
- MAPE (Mean Absolute Percentage Error): Selisih persentase absolut rata-rata antara nilai yang diprediksi oleh model dan nilai data yang diamati.

Metrik evaluasi ini memberikan gambaran sejauh mana model dapat memprediksi harga saham Unilever Indonesia dengan akurat. Semakin rendah nilai MAE dan RMSE, semakin baik performa model dalam melakukan prediksi yang akurat. 

Hasil evaluasi yang diperoleh adalah sebagai berikut:
| Metrics  | Value |
| ------------- | ------------- |
| MAE (Mean Avsolute Error)  | 0.009871991375308926  |
| RSME (Root Mean Squared Error)  | 0.013863547909730151  |
| MAPE (Mean Absolute Percent Error)  | 0.017773489787324806  |

Model yang telah dibangun menghasilkan hasil evaluasi yang baik dalam memprediksi harga saham. MAE, RMSE, dan MAPE yang rendah menunjukkan bahwa model memiliki tingkat kesalahan yang rendah dan mampu mendekati nilai sebenarnya dengan akurasi yang baik.

## Kesimpulan
Berdasarkan hasil evaluasi, model yang telah dibangun menunjukkan performa yang baik dalam memprediksi harga saham. Dalam melakukan evaluasi model, MAE, RMSE, dan MAPE yang rendah menunjukkan bahwa model memiliki tingkat kesalahan yang rendah dan mampu mendekati nilai sebenarnya dengan akurasi yang baik.

## Daftar Pustaka
- [Unilever Indonesia](https://www.unilever.co.id/our-company/)
- [Unilever Indonesia Tbk. - Company Background](https://stockbit.com/symbol/UNVR) 
- [Indonesia Stock Dataset](https://www.kaggle.com/datasets/muamkh/ihsgstockdata)
- [Time-Series Forecasting: Predicting Stock Prices Using An LSTM Model](https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f) 
- [Stock Price Prediction using Machine Learning with Source Code](https://www.projectpro.io/article/stock-price-prediction-using-machine-learning-project/571)
- [Forecasting statistical details](https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=forecasting-statistical-details)
