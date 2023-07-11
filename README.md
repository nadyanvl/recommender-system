# Laporan Proyek Machine Learning (Recommender System) - Nadya Novalina
# Sephora Products Recommender System
# Domain Proyek
Proyek ini bertujuan mengembangkan sistem rekomendasi produk untuk toko online Sephora. Fokus utama proyek ini adalah memberikan rekomendasi produk yang relevan dan personal kepada pelanggan Sephora berdasarkan preferensi dan karakteristik mereka. Dalam proyek ini, digunakan dua pendekatan berbeda, yaitu content-based filtering dan collaborative filtering [[1]](#daftar-pustaka).

Dalam content-based filtering, fitur-fitur produk diekstraksi menggunakan TF-IDF Vectorizer dan dijadikan vektor representasi. Kemudian, kemiripan antara vektor fitur dihitung dengan cosine similarity untuk merekomendasikan produk yang mirip dengan yang diminati pengguna. Collaborative filtering menggunakan data rating pengguna dan SVD untuk mengidentifikasi pola kesamaan antara pengguna. Dengan informasi ini, model merekomendasikan produk yang mendapatkan rating tinggi dari pengguna dengan preferensi serupa.

Beberapa penelitian sebelumnya telah dilakukan dalam bidang sistem rekomendasi produk dan menunjukkan hasil yang menjanjikan. Misalnya, dalam penelitian [[2]](#daftar-pustaka), mereka mengimplementasikan metode content-based filtering pada sistem rekomendasi produk kosmetik dan memberikan rekomendasi berdasarkan kebutuhan khusus pengguna. Penelitian lainnya [[3]](#daftar-pustaka) mencoba pendekatan collaborative filtering pada produk di platform e-commerce dengan menggunakan algoritma SVD dan menghasilkan nilai root mean square error (RMSE) sebesar 1.055586 dan dengan mengoptimalkan hyperparameter, mereka berhasil mendapatkan model dengan nilai RMSE sebesar 1.041784.

Dengan menggabungkan pengetahuan dari penelitian-penelitian sebelumnya, proyek ini bertujuan untuk mengembangkan sistem rekomendasi produk yang dapat memberikan rekomendasi yang lebih relevan dan personal kepada pelanggan. Pendekatan yang digunakan dalam proyek ini akan memanfaatkan teknik content-based filtering dan collaborative filtering. Diharapkan sistem rekomendasi ini dapat meningkatkan pengalaman berbelanja pelanggan dan mendorong peningkatan penjualan produk.

# Business Understanding
Proyek ini bertujuan untuk mengembangkan sistem rekomendasi produk untuk toko online Sephora. Sephora adalah sebuah perusahaan kosmetik dan produk kecantikan yang terkenal dengan berbagai macam produk yang ditawarkan kepada pelanggan. Namun, dengan banyaknya pilihan produk yang tersedia, pelanggan seringkali kesulitan untuk menemukan produk yang sesuai dengan preferensi dan kebutuhan mereka. Oleh karena itu, tujuan proyek ini adalah untuk memberikan pengalaman belanja yang lebih baik kepada pelanggan dengan menyediakan rekomendasi produk yang relevan dan personal.

## Problem Statements
* Pelanggan Sephora kesulitan menemukan produk yang sesuai dengan preferensi dan karakteristik mereka di tengah banyaknya pilihan produk.
* Proses pencarian produk yang memakan waktu lama mengurangi kepuasan pelanggan dan efisiensi proses berbelanja.

## Goals
* Mengembangkan sistem rekomendasi produk yang dapat memberikan rekomendasi yang relevan dan personal kepada pelanggan berdasarkan preferensi dan karakteristik mereka.
* Meningkatkan pengalaman berbelanja, kepuasan pelanggan, dan penjualan melalui pencarian yang lebih cepat, rekomendasi yang akurat, dan pengalaman berbelanja yang lebih baik.

## Solution statements
* Mengimplementasikan metode content-based filtering dengan menggunakan TF-IDF Vectorizer untuk mengekstraksi fitur-fitur seperti bahan-bahan, kategori utama, dan kategori tersier dari setiap produk.
* Menghitung kemiripan antara vektor fitur produk menggunakan metode cosine similarity untuk merekomendasikan produk yang memiliki kesamaan tinggi dengan produk yang diminati oleh pengguna.
* Mengimplementasikan metode collaborative filtering dengan menggunakan teknik Singular Value Decomposition (SVD) untuk menganalisis perilaku rating pengguna terhadap produk.
* Mengidentifikasi pengguna dengan preferensi serupa berdasarkan data rating dan merekomendasikan produk yang mendapatkan rating tinggi dari pengguna-pengguna serupa tersebut.

# Data Understanding
Dataset yang digunakan dalam proyek ini diperoleh dari kumpulan data Kaggle yang berjudul "[Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)". Dataset ini menyediakan informasi komprehensif tentang lebih dari 8.000 produk kecantikan yang tersedia di toko online Sephora. Dataset ini mencakup berbagai atribut untuk setiap produk, seperti nama produk, merek produk, harga, bahan-bahan, penilaian, dan fitur-fitur relevan lainnya.

Dataset yang digunakan dalam proyek ini telah melalui proses [data cleaning](#data-cleaning). Terdapat dua file utama: "product_info.csv" dan "reviews_1500_end.csv".

* File "product_info.csv" memiliki 6.642 baris dan 6 kolom. Setiap baris mewakili sebuah produk kecantikan yang tersedia di toko online Sephora, dan kolom-kolomnya berisi informasi sebagai berikut:

  * `product_id`: ID unik untuk produk tersebut di situs web.
  * `product_name`: Nama lengkap produk.
  * `brand_name`: Nama merek produk.
  * `ingredients`: Daftar bahan yang terkandung dalam produk.
  * `primary_category`: Kategori utama produk.
  * `tertiary_category`: Kategori tersier produk.

* File "reviews_1500_end.csv" terdiri dari 49.977 baris dan 3 kolom. Setiap baris mewakili ulasan pengguna untuk produk tertentu, dan kolom-kolomnya meliputi:

  * `author_id`: ID unik untuk penulis ulasan di situs web.
  * `product_id`: ID unik untuk produk tersebut.
  * `rating`: Penilaian yang diberikan oleh penulis ulasan untuk produk dengan skala 1 hingga 5.

# Data Preparation
## Data Cleaning
Kolom yang tidak digunakan atau tidak relevan dapat dihapus dari kedua file sehingga dapat menyederhanakan dataset dan fokus pada atribut-atribut yang penting. Selain itu, juga dilakukan penghapusan data pada missing value dan data duplikat untuk memastikan kualitas data yang baik.

**Hasil data cleaning:**

![image](https://raw.githubusercontent.com/nadyanvl/recommender-system/main/assets/product_data.png)
Gambar 1. Dataframe Produk

![image](https://raw.githubusercontent.com/nadyanvl/recommender-system/main/assets/review_data.png)

Gambar 2. Dataframe Review

## Exploratory Data Analysis
Pada file "product_info.csv", dilakukan eksplorasi terhadap atribut-atribut seperti nama produk, merek produk, harga, bahan-bahan, kategori utama, dan kategori tersier. Pada tahap ini dapat dilihat Distribusi produk per kategori utama, top-10 brand, top-10 ingredient semua product.
* Kategori produk yang paling banyak terdapat di Sephora adalah skincare dan makeup.
* Brand yang memiliki jumlah produk yang paling banyak adalah Sephora Collection dan Clinique. 
* Bahan (ingredient) yang paling banyak terdapat pada produk secara keseluruhan adalah glycerin dan phenoxyethanol. 
![image](https://raw.githubusercontent.com/nadyanvl/recommender-system/main/assets/EDA_product.png)
Gambar 3. EDA Produk

Pada file "reviews.csv", analisis data eksploratori dilakukan terhadap atribut-atribut seperti ID penulis ulasan, ID produk, dan rating yang diberikan oleh penulis. Pada tahap ini, dapat dilihat distribusi rating, rata-rata rating per product, jumlah rating per product, rata-rata rating.
* Lebih dari 30.000 pengguna memberikan rating 5 untuk produk. Hal ini menunjukkan bahwa banyak pengguna memberikan rating tinggi (5) untuk produk yang ada dalam dataset.
* Rata-rata produk memiliki rating yang bagus. Meskipun ada produk dengan rating yang rendah, secara umum, rata-rata rating produk cenderung tinggi. Ini menunjukkan bahwa sebagian besar produk dalam dataset memiliki rating yang baik.

![image](https://raw.githubusercontent.com/nadyanvl/recommender-system/main/assets/EDA_rating_1.png)
Gambar 4. EDA Review part 1

![image](https://raw.githubusercontent.com/nadyanvl/recommender-system/main/assets/EDA_rating_2.png)
Gambar 5. EDA Review part 2

# Modeling
## Content-Based Filtering
Content-based filtering adalah metode dalam sistem rekomendasi yang menggunakan kesamaan atribut atau konten dari produk yang telah disukai atau dilihat oleh pengguna sebelumnya. Sistem menganalisis atribut produk seperti deskripsi, fitur, kategori, dan metadata lainnya. Profil pengguna dibangun berdasarkan preferensi atau interaksi dengan produk, lalu sistem mencari kesamaan antara atribut produk dalam profil pengguna dan database produk [[1]](#daftar-pustaka)[[2]](#daftar-pustaka).

**Feature Engineering:**

Metode content-based filtering dikembangkan dengan menggunakan algoritma TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengekstraksi fitur-fitur dari produk. Fitur-fitur yang diekstraksi meliputi bahan-bahan, kategori utama, dan kategori tersier. Setelah itu, fitur -fitur tersebut digabungkan menjadi satu vektor representasi menggunakan hstack. Hal ini bertujuan untuk menggabungkan informasi dari berbagai atribut produk menjadi satu representasi fitur yang komprehensif. 

Model TF-IDF adalah sebuah metode analisis yang mengevaluasi seberapa relevan suatu kata dalam sebuah dokumen dalam kumpulan dokumen. TF-IDF merupakan gabungan dari dua metode ekstraksi, yaitu TF (Term Frequency) dan IDF (Inverse Document Frequency). TF menghitung seberapa sering sebuah kata muncul dalam suatu dokumen dan IDF menghitung seberapa umum atau langka sebuah kata dalam seluruh kumpulan dokumen, sehingga mengetahui pentingnya sebuah kata [[4]](#daftar-pustaka).

**Similarity Calculation:**

Selanjutnya, dilakukan perhitungan kemiripan menggunakan metode cosine similarity. Metode ini digunakan untuk mengukur sejauh mana dua vektor fitur tersebut mirip satu sama lain. Hasilnya adalah matriks kemiripan yang menunjukkan tingkat kemiripan antara setiap pasang produk dalam dataset.

Cosine similarity adalah metode untuk mengukur tingkat kesamaan antara dua vektor. Dalam konteks dokumen, cosine similarity digunakan untuk membandingkan kesamaan antara dua dokumen berdasarkan representasi vektor dari dokumen-dokumen tersebut [2](#daftar-pustaka).

Rumus perhitungan cosine similarity:

cosine similarity (x, y) = Σ(xi * yi) / (√(Σ(xi^2)) * √(Σ(yi^2)))

Di mana:
- x dan y adalah dua vektor yang akan dibandingkan.
- xi adalah elemen ke-i pada vektor x.
- yi adalah elemen ke-i pada vektor y.

Perhitungan cosine similarity melibatkan perkalian dan penjumlahan elemen-elemen vektor, yang kemudian dibagi dengan perkalian magnitudo masing-masing vektor. Hasil perhitungan cosine similarity berada dalam rentang -1 hingga 1. Nilai yang mendekati 1 menunjukkan tingkat kesamaan yang tinggi antara dua vektor, sementara nilai yang mendekati -1 menunjukkan perbedaan yang besar. Nilai yang mendekati 0 menunjukkan bahwa dua vektor tidak memiliki kesamaan yang signifikan.

![image](https://raw.githubusercontent.com/nadyanvl/recommender-system/main/assets/cosine%20similarity.png)

Gambar 6. Hasil matrix similarity menggunakan cosine similarity

**Top-10 Recommmendation (Example Usage):**

Kemudian dibuat fungsi untuk menghasilkan rekomendasi produk berdasarkan produk yang diberikan dengan menghitung kemiripan antara produk yang diberikan dengan produk lain dalam dataset, dan mengambil produk dengan kemiripan tertinggi sebagai rekomendasi. Berikut contoh hasil top-10 rekomendasi produk berdasarkan produk dengan ID tertentu.

![image](https://raw.githubusercontent.com/nadyanvl/recommender-system/main/assets/Top-10%20content-based%20filtering.png)
Gambar 7. Top-10 Recommendation: Content-Based Filtering

## Collaborative Filtering
Collaborative filtering adalah metode dalam sistem rekomendasi yang menggunakan informasi dari pengguna lain untuk memberikan rekomendasi kepada pengguna. Metode ini mengidentifikasi pengguna dengan preferensi atau perilaku serupa, dan menggunakan informasi dari pengguna-pengguna tersebut untuk meramalkan preferensi atau rekomendasi bagi pengguna target [[1]](#daftar-pustaka)[[3]](#daftar-pustaka).

**Training:**
Pelatihan model collaborative filtering dilakukan menggunakan algoritma SVD. Dataset dibagi menjadi set pelatihan dan pengujian menggunakan fungsi train_test_split dari pustaka Surprise. Pertama, model SVD dilatih dengan menggunakan cross-validation. Kemudian model SVD tersebut dilatih menggunakan keseluruhan trainset. Metrik evaluasi yang digunakan adalah RMSE (Root Mean Square Error) dan MAE (Mean Absolute Error), yang mengukur keakuratan prediksi model dibandingkan dengan rating sebenarnya.

**Hyperparameter Tuning:**
Untuk mengoptimalkan model SVD, dilakukan hyperparameter tuning menggunakan GridSearchCV dari pustaka Surprise. GridSearchCV digunakan untuk mencari kombinasi hiperparameter terbaik yang meminimalkan skor RMSE dan MAE. Hiperparameter yang dipertimbangkan dalam proses penyetelan meliputi n_factor, epoch, lr_all (tingkat pembelajaran), dan reg_all (tingkat regularisasi).
Hiperparameter terbaik yang ditemukan melalui penyetelan adalah n_factor sebesar 50, epoch sebesar 50, lr_all sebesar 0.01, dan reg_all sebesar 0.1. Model SVD kemudian dilatih dengan seluruh trainset menggunakan hiperparameter optimal ini.

**Top-10 Recommmendation (Example Usage):**

Kemudian dibuat fungsi untuk menghasilkan rekomendasi produk berdasarkan pengguna spesifik dengan memprediksi rating untuk produk-produk yang belum dinilai oleh pengguna tersebut. Prediksi tersebut akan diurutkan secara menurun. Berikut contoh hasil top-10 rekomendasi produk berdasarkan pengguna dengan ID tertentu.

![image](https://raw.githubusercontent.com/nadyanvl/recommender-system/main/assets/Top-10%20collaborative%20filtering.png)
Gambar 8. Top-10 Recommendation: Content-Based Filtering

# Evaluation

# Kesimpulan

# Daftar Pustaka
[[1]](https://iopscience.iop.org/article/10.1088/1742-6596/1897/1/012024/pdf) Abdul Hussien, Farah Tawfiq, et al. “Recommendation Systems for E-Commerce Systems an Overview.” Journal of Physics: Conference Series, vol. 1897, no. 1, 1 May 2021, p. 012024, https://doi.org/10.1088/1742-6596/1897/1/012024.

[[2]](http://www.ijceit.org/published/volume10/issue1/1Vol10No1.pdf) Patty, Joanna, et al. “Recommendations System for Purchase of Cosmetics Using Content- Based Filtering.” International Journal of Computer Engineering and Information Technology, vol. 10, no. 1, 2018, pp. 1–5, www.ijceit.org/published/volume10/issue1/1Vol10No1.pdf. Accessed 10 July 2023.

[[3]](https://doi.org/10.35313/jitel.v2.i2.2022.103-110) Shalannanda, Wervyan, et al. “Singular Value Decomposition Model Application for E-Commerce Recommendation System.” JITEL (Jurnal Ilmiah Telekomunikasi, Elektronika, Dan Listrik Tenaga), vol. 2, no. 2, 30 Sept. 2022, pp. 103–110, https://doi.org/10.35313/jitel.v2.i2.2022.103-110. Accessed 7 Apr. 2023.

[[4]](https://www.ijrte.org/wp-content/uploads/papers/v11i4/D73311111422.pdf) Yeruva, Sagar, et al. “Apparel Recommendation System Using Content-Based Filtering.” International Journal of Recent Technology and Engineering (IJRTE), vol. 11, no. 4, 30 Nov. 2022, pp. 46–51, https://doi.org/10.35940/ijrte.d7331.1111422. Accessed 13 Feb. 2023.
