# Laporan Proyek Machine Learning (Recommender System) - Nadya Novalina
# Sephora Products Recommender System
# Domain Proyek
[Sephora](https://www.sephora.com/) adalah platform e-commerce yang dikenal secara luas sebagai salah satu toko kosmetik dan produk kecantikan terbesar dan terpercaya di dunia. Sephora menawarkan berbagai macam produk kecantikan termasuk skincare, makeup, perawatan rambut, parfum, dan masih banyak lagi dari berbagai merek terkenal dan berkualitas. Pelanggan dapat menemukan produk-produk terbaru, tren terkini, serta berbagai pilihan produk yang sesuai dengan preferensi dan kebutuhan mereka.

Pelanggan sering menghadapi tantangan dalam menemukan produk yang sesuai, seperti banyaknya pilihan, kompleksitas produk, ketersediaan informasi yang terbatas, dan kecepatan pencarian yang lambat. Untuk mengatasi tantangan ini, pengembangan sistem rekomendasi produk dapat membantu memberikan rekomendasi yang relevan dan personal kepada pelanggan, meningkatkan pengalaman berbelanja, dan mendorong peningkatan penjualan.

Proyek ini bertujuan mengembangkan sistem rekomendasi produk untuk Sephora. Fokus utama proyek ini adalah memberikan rekomendasi produk yang relevan dan personal kepada pelanggan Sephora berdasarkan preferensi dan karakteristik mereka. Dalam proyek ini, digunakan dua pendekatan berbeda, yaitu content-based filtering dan collaborative filtering [[1]](#daftar-pustaka).

Dalam content-based filtering, fitur-fitur produk diekstraksi menggunakan TF-IDF Vectorizer dan dijadikan vektor representasi. Kemudian, kemiripan antara vektor fitur dihitung dengan cosine similarity untuk merekomendasikan produk yang mirip dengan yang diminati pengguna. Collaborative filtering menggunakan data rating pengguna dan SVD untuk mengidentifikasi pola kesamaan antara pengguna. Dengan informasi ini, model merekomendasikan produk yang mendapatkan rating tinggi dari pengguna dengan preferensi serupa.

Beberapa penelitian sebelumnya telah dilakukan dalam bidang sistem rekomendasi produk dan menunjukkan hasil yang menjanjikan. Misalnya, dalam penelitian [[2]](#daftar-pustaka), mereka mengimplementasikan metode content-based filtering pada sistem rekomendasi produk kosmetik dan memberikan rekomendasi berdasarkan kebutuhan khusus pengguna. Penelitian lainnya [[3]](#daftar-pustaka) mencoba pendekatan collaborative filtering pada produk di platform e-commerce dengan menggunakan algoritma SVD dan menghasilkan nilai root mean square error (RMSE) sebesar 1.055586 dan dengan mengoptimalkan hyperparameter, mereka berhasil mendapatkan model dengan nilai RMSE sebesar 1.041784.

Dengan menggabungkan pengetahuan dari penelitian-penelitian sebelumnya, proyek ini bertujuan untuk mengembangkan sistem rekomendasi produk yang dapat memberikan rekomendasi yang lebih relevan dan personal kepada pelanggan. Dan diharapkan dapat meningkatkan pengalaman berbelanja pelanggan dan mendorong peningkatan penjualan produk.

# Business Understanding
Proyek ini bertujuan untuk mengembangkan sistem rekomendasi produk untuk toko online Sephora. Sephora adalah sebuah perusahaan kosmetik dan produk kecantikan yang terkenal dengan berbagai macam produk yang ditawarkan kepada pelanggan. Namun, dengan banyaknya pilihan produk yang tersedia, pelanggan seringkali kesulitan untuk menemukan produk yang sesuai dengan preferensi dan kebutuhan mereka. Proses pencarian produk yang memakan waktu lama juga dapat mengurangi kepuasan pelanggan dan efisiensi proses berbelanja. Oleh karena itu, tujuan utama proyek ini adalah untuk memberikan pengalaman belanja yang lebih baik kepada pelanggan dengan menyediakan rekomendasi produk yang relevan dan personal.

## Problem Statements
* Pelanggan Sephora kesulitan menemukan produk yang sesuai dengan preferensi dan karakteristik mereka di tengah banyaknya pilihan produk.
* Proses pencarian produk yang memakan waktu lama mengurangi kepuasan pelanggan dan efisiensi proses berbelanja.

## Goals
* Mengembangkan sistem rekomendasi produk yang dapat memberikan rekomendasi yang relevan dan personal kepada pelanggan berdasarkan preferensi dan karakteristik mereka.
* Meningkatkan pengalaman berbelanja, kepuasan pelanggan, dan penjualan melalui pencarian yang lebih cepat, rekomendasi yang akurat, dan pengalaman berbelanja yang lebih baik.

Dengan mengembangkan sistem rekomendasi produk yang efektif, diharapkan Sephora dapat mengatasi tantangan yang dihadapi oleh pelanggan dalam menemukan produk yang sesuai dan meningkatkan pengalaman belanja mereka. Rekomendasi produk yang relevan dan personal akan membantu pelanggan menemukan produk yang sesuai dengan preferensi dan karakteristik mereka dengan lebih mudah dan cepat. Hal ini akan meningkatkan kepuasan pelanggan, efisiensi proses berbelanja, dan pada akhirnya, mendorong peningkatan penjualan produk di toko online Sephora.

## Solution statements
* Mengimplementasikan metode content-based filtering dengan menggunakan TF-IDF Vectorizer untuk mengekstraksi fitur-fitur seperti bahan-bahan, kategori utama, dan kategori tersier dari setiap produk.
* Menghitung kemiripan antara vektor fitur produk menggunakan metode cosine similarity untuk merekomendasikan produk yang memiliki kesamaan tinggi dengan produk yang diminati oleh pengguna.
* Mengimplementasikan metode collaborative filtering dengan menggunakan teknik Singular Value Decomposition (SVD) untuk menganalisis perilaku rating pengguna terhadap produk.
* Mengidentifikasi pengguna dengan preferensi serupa berdasarkan data rating dan merekomendasikan produk yang mendapatkan rating tinggi dari pengguna-pengguna serupa tersebut.

# Data Understanding
Dataset yang digunakan dalam proyek ini diperoleh dari kumpulan data Kaggle yang berjudul "[Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)". Dataset ini menyediakan informasi komprehensif tentang lebih dari 8.000 produk kecantikan yang tersedia di toko online Sephora. Dataset ini mencakup berbagai atribut untuk setiap produk, seperti nama produk, merek produk, harga, bahan-bahan, penilaian, dan fitur-fitur relevan lainnya.

Dataset yang digunakan dalam proyek ini yaitu: "product_info.csv" dan "reviews_1500_end.csv".

* File "product_info.csv" memiliki 8.494 baris dan 27 kolom. Setiap baris mewakili sebuah produk kecantikan yang tersedia di toko online Sephora, dan kolom-kolomnya berisi informasi sebagai berikut:

  * `product_id`: ID unik untuk produk tersebut di situs web
  * `product_name`: Nama lengkap produk
  * `brand_id`: Identifier unik untuk merek produk dari situs web
  * `brand_name`: Nama merek produk
  * `loves_count` Jumlah orang yang telah menandai produk ini sebagai favorit
  * `rating`: Rating rata-rata produk berdasarkan ulasan pengguna
  * `reviews`: Jumlah ulasan pengguna untuk produk tersebut
  * `size`: Ukuran produk, dapat dalam oz, ml, g, paket, atau unit lain tergantung pada jenis produk
  * `variation_type`: Tipe parameter variasi untuk produk
  * `variation_value`: Nilai spesifik dari parameter variasi untuk produk
  * `variation_desc`: Deskripsi parameter variasi untuk produk
  * `ingredients`: Daftar bahan yang terkandung dalam produk.
  * `price_usd`: Harga produk dalam dolar Amerika Serikat
  * `value_price_usd`: Potensi penghematan biaya produk, ditampilkan di situs web di sebelah harga reguler
  * `sale_price_usd`: Harga jual produk dalam dolar Amerika Serikat
  * `limited_edition`: Menunjukkan apakah produk ini edisi terbatas atau tidak
  * `new`: Menunjukkan apakah produk ini baru atau tidak
  * `online_only`: Menunjukkan apakah produk ini hanya dijual secara online atau tidak 
  * `out_of_stock`: Menunjukkan apakah produk ini saat ini kosong atau tidak
  * `sephora_exclusive`: Menunjukkan apakah produk ini eksklusif untuk Sephora atau tidak
  * `highlights`: Daftar tag atau fitur yang menyoroti atribut-atribut produk
  * `primary_category`: Kategori utama produk
  * `secondary_category`: Kategori kedua
  * `tertiary_category`: Kategori tersier produk
  * `child_count`: Jumlah variasi produk yang tersedia
  * `child_max_price`: Harga tertinggi di antara variasi produk
  * `child_min_price`: Harga terendah di antara variasi produk
 
Gambaran dataframe dari product_info.csv dapat dilihat pada Tabel 1. yang berisi beberapa data terkait rincian produk.

Tabel 1. Dataframe produk
| product_id |       product_name       | brand_id | brand_name | loves_count | rating | reviews |     size      |     variation_type      |   variation_value   | ... | online_only | out_of_stock | sephora_exclusive |                     highlights                    | primary_category | secondary_category |  tertiary_category  | child_count | child_max_price | child_min_price |
|------------|--------------------------|----------|------------|-------------|--------|---------|---------------|-------------------------|---------------------|-----|-------------|--------------|-------------------|---------------------------------------------------|------------------|---------------------|---------------------|-------------|-----------------|-----------------|
|  P473671   | Fragrance Discovery Set  |   6342   |   19-69    |     6320    | 3.6364 |   11.0  |     NaN       |          NaN            |        NaN          | ... |      1      |      0       |        0          | ['Unisex/ Genderless Scent', 'Warm & Spicy Scen... |     Fragrance     |  Value & Gift Sets |  Perfume Gift Sets |      0      |       NaN       |       NaN       |
|  P473668   |  La Habana Eau de Parfum |   6342   |   19-69    |     3827    | 4.1538 |   13.0  | 3.4 oz/ 100 mL| Size + Concentration + Formulation | 3.4 oz/ 100 mL | ... |      1      |      0       |        0          | ['Unisex/ Genderless Scent', 'Layerable Scent'... |     Fragrance     |       Women       |      Perfume       |      2      |      85.0       |      30.0       |
|  P473662   | Rainbow Bar Eau de Parfum|   6342   |   19-69    |     3253    |  4.25  |   16.0  | 3.4 oz/ 100 mL| Size + Concentration + Formulation | 3.4 oz/ 100 mL | ... |      1      |      0       |        0          | ['Unisex/ Genderless Scent', 'Layerable Scent'... |     Fragrance     |       Women       |      Perfume       |      2      |      75.0       |      30.0       |


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

# Modeling and Results
## Content-Based Filtering
Content-based filtering adalah metode dalam sistem rekomendasi yang menggunakan kesamaan atribut atau konten dari produk yang telah disukai atau dilihat oleh pengguna sebelumnya. Sistem menganalisis atribut produk seperti deskripsi, fitur, kategori, dan metadata lainnya. Profil pengguna dibangun berdasarkan preferensi atau interaksi dengan produk, lalu sistem mencari kesamaan antara atribut produk dalam profil pengguna dan database produk [[1]](#daftar-pustaka)[[2]](#daftar-pustaka).

**Feature Engineering:**

Metode content-based filtering dikembangkan dengan menggunakan algoritma TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengekstraksi fitur-fitur dari produk. Fitur-fitur yang diekstraksi meliputi bahan-bahan, kategori utama, dan kategori tersier. Setelah itu, fitur -fitur tersebut digabungkan menjadi satu vektor representasi menggunakan hstack. Hal ini bertujuan untuk menggabungkan informasi dari berbagai atribut produk menjadi satu representasi fitur yang komprehensif. 

Model TF-IDF adalah sebuah metode analisis yang mengevaluasi seberapa relevan suatu kata dalam sebuah dokumen dalam kumpulan dokumen. TF-IDF merupakan gabungan dari dua metode ekstraksi, yaitu TF (Term Frequency) dan IDF (Inverse Document Frequency). TF menghitung seberapa sering sebuah kata muncul dalam suatu dokumen dan IDF menghitung seberapa umum atau langka sebuah kata dalam seluruh kumpulan dokumen, sehingga mengetahui pentingnya sebuah kata [[4]](#daftar-pustaka).

**Similarity Calculation:**

Selanjutnya, dilakukan perhitungan kemiripan menggunakan metode cosine similarity. Metode ini digunakan untuk mengukur sejauh mana dua vektor fitur tersebut mirip satu sama lain. Hasilnya adalah matriks kemiripan yang menunjukkan tingkat kemiripan antara setiap pasang produk dalam dataset.

Cosine similarity adalah metode untuk mengukur tingkat kesamaan antara dua vektor. Dalam konteks dokumen, cosine similarity digunakan untuk membandingkan kesamaan antara dua dokumen berdasarkan representasi vektor dari dokumen-dokumen tersebut [[2]](#daftar-pustaka).

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

**SVD (Singular Value Decomposition):**

SVD (Singular Value Decomposition) adalah teknik untuk memecah sebuah matriks menjadi tiga matriks terpisah: U, Σ, dan V. Matriks Σ berisi nilai-nilai singular yang menunjukkan signifikansi dari vektor singular. SVD digunakan untuk mengurangi dimensi data dan menangkap informasi penting dalam matriks asli. Rumus SVD adalah A = UΣV^T, di mana U dan V adalah matriks singular kiri dan kanan, dan Σ adalah matriks diagonal dengan nilai-nilai singular [[3]](#daftar-pustaka).

Pelatihan model collaborative filtering dilakukan menggunakan algoritma SVD. Dataset dibagi menjadi set pelatihan dan pengujian menggunakan fungsi train_test_split dari pustaka Surprise. Pertama, model SVD dilatih dengan menggunakan cross-validation. Kemudian model SVD tersebut dilatih menggunakan keseluruhan trainset. Metrik evaluasi yang digunakan adalah RMSE (Root Mean Square Error) dan MAE (Mean Absolute Error), yang mengukur keakuratan prediksi model dibandingkan dengan rating sebenarnya.

**Hyperparameter Tuning:**

Untuk mengoptimalkan model SVD, dilakukan hyperparameter tuning menggunakan GridSearchCV dari pustaka Surprise. GridSearchCV digunakan untuk mencari kombinasi hiperparameter terbaik yang meminimalkan skor RMSE dan MAE. Hiperparameter yang dipertimbangkan dalam proses penyetelan meliputi n_factor, epoch, lr_all (tingkat pembelajaran), dan reg_all (tingkat regularisasi).
Hiperparameter terbaik yang ditemukan melalui penyetelan adalah n_factor sebesar 50, epoch sebesar 50, lr_all sebesar 0.01, dan reg_all sebesar 0.1. Model SVD kemudian dilatih dengan seluruh trainset menggunakan hiperparameter optimal ini.

**Top-10 Recommmendation (Example Usage):**

Kemudian dibuat fungsi untuk menghasilkan rekomendasi produk berdasarkan pengguna spesifik dengan memprediksi rating untuk produk-produk yang belum dinilai oleh pengguna tersebut. Prediksi tersebut akan diurutkan secara menurun. Berikut contoh hasil top-10 rekomendasi produk berdasarkan pengguna dengan ID tertentu.

![image](https://raw.githubusercontent.com/nadyanvl/recommender-system/main/assets/Top-10%20collaborative%20filtering.png)
Gambar 8. Top-10 Recommendation: Collaborative Filtering

# Evaluation
## Collaborative Filtering
Evaluasi model collaborative filtering dilakukan untuk mengukur kinerja dan akurasi model dalam memberikan rekomendasi produk kepada pengguna. Terdapat beberapa metrik evaluasi yang umum digunakan:

* Root Mean Square Error (RMSE): Metrik ini mengukur perbedaan antara rating yang diprediksi oleh model dengan rating sebenarnya. RMSE menghitung akar dari rata-rata dari kuadrat selisih antara rating prediksi dan rating sebenarnya. Semakin rendah nilai RMSE, semakin baik model dalam memprediksi rating.

* Mean Absolute Error (MAE): Metrik ini juga mengukur perbedaan antara rating prediksi dan rating sebenarnya, tetapi tanpa memperhatikan arah perbedaan. MAE menghitung rata-rata dari selisih absolut antara rating prediksi dan rating sebenarnya. Nilai MAE yang lebih rendah menunjukkan kinerja model yang lebih baik.

* Precision, Recall, dan F1-Score: Precision mengukur sejauh mana produk yang direkomendasikan benar-benar relevan bagi pengguna. Recall mengukur sejauh mana produk yang relevan berhasil ditemukan oleh model. F1-Score adalah ukuran kombinasi dari precision dan recall. Semakin tinggi nilai precision, recall, dan F1-Score, semakin baik kinerja model dalam memberikan rekomendasi yang relevan.

Hasil evaluasi yang diperoleh adalah sebagai berikut:

Tabel 1. Hasil evaluasi model SVD pada data uji (full trainset)

| Metrics  | Tanpa Hyperparameter Tuning | Dengan Hyperparameter Tuning |
| --------- | --------- | --------- |
| RSME (Root Mean Squared Error)  | 0.7998  | **0.3348** |
| MAE (Mean Avsolute Error)  | 0.6083  | **0.2539** |
| Precision | 0.9194 | **0.9946** |
| Recall | **0.9983** | 0.9978 |
| F1-score | 0.9572 | **0.9962** |

Hasil evaluasi menggunakan model SVD pada data uji (dengan full trainset) menunjukkan bahwa penggunaan hyperparameter tuning memiliki dampak positif terhadap kualitas prediksi. Model dengan hyperparameter tuning menghasilkan nilai RMSE dan MAE yang lebih rendah, menandakan kemampuan yang lebih baik dalam memperkirakan rating yang sebenarnya. Selain itu, terdapat peningkatan yang signifikan dalam presisi, recall, dan F1-score, yang mengindikasikan kemampuan yang lebih baik dalam mengklasifikasikan ulasan dan memberikan rekomendasi produk yang lebih relevan. Oleh karena itu, penerapan hyperparameter tuning pada model SVD dapat meningkatkan akurasi dan kualitas sistem rekomendasi produk, yang berpotensi meningkatkan kepuasan pelanggan dan penjualan.

# Kesimpulan
Dalam proyek pengembangan sistem rekomendasi produk untuk toko online Sephora, telah diimplementasikan dua pendekatan: content-based filtering dan collaborative filtering. Pendekatan content-based filtering menggunakan TF-IDF Vectorizer untuk mengekstraksi fitur-fitur produk dan menghitung kemiripan antara produk menggunakan metode cosine similarity. Pendekatan collaborative filtering menggunakan metode Singular Value Decomposition (SVD) untuk menganalisis perilaku rating pengguna.

Melalui evaluasi model, hasilnya menunjukkan bahwa penggunaan cosine similarity dalam content-based filtering mampu memberikan rekomendasi produk yang relevan dengan preferensi pengguna. Selain itu, melalui hyperparameter tuning pada model SVD, diperoleh model dengan performa yang lebih baik dalam memprediksi rating dan memberikan rekomendasi produk.

Dengan penerapan sistem rekomendasi ini, diharapkan dapat meningkatkan pengalaman berbelanja pelanggan dengan rekomendasi yang akurat dan personal. Hal ini berpotensi meningkatkan kepuasan pelanggan, penjualan, dan retensi pelanggan di toko online Sephora.

# Daftar Pustaka
[[1]](https://iopscience.iop.org/article/10.1088/1742-6596/1897/1/012024/pdf) Abdul Hussien, Farah Tawfiq, et al. “Recommendation Systems for E-Commerce Systems an Overview.” Journal of Physics: Conference Series, vol. 1897, no. 1, 1 May 2021, p. 012024, https://doi.org/10.1088/1742-6596/1897/1/012024.

[[2]](http://www.ijceit.org/published/volume10/issue1/1Vol10No1.pdf) Patty, Joanna, et al. “Recommendations System for Purchase of Cosmetics Using Content- Based Filtering.” International Journal of Computer Engineering and Information Technology, vol. 10, no. 1, 2018, pp. 1–5, www.ijceit.org/published/volume10/issue1/1Vol10No1.pdf. Accessed 10 July 2023.

[[3]](https://doi.org/10.35313/jitel.v2.i2.2022.103-110) Shalannanda, Wervyan, et al. “Singular Value Decomposition Model Application for E-Commerce Recommendation System.” JITEL (Jurnal Ilmiah Telekomunikasi, Elektronika, Dan Listrik Tenaga), vol. 2, no. 2, 30 Sept. 2022, pp. 103–110, https://doi.org/10.35313/jitel.v2.i2.2022.103-110. Accessed 7 Apr. 2023.

[[4]](https://www.ijrte.org/wp-content/uploads/papers/v11i4/D73311111422.pdf) Yeruva, Sagar, et al. “Apparel Recommendation System Using Content-Based Filtering.” International Journal of Recent Technology and Engineering (IJRTE), vol. 11, no. 4, 30 Nov. 2022, pp. 46–51, https://doi.org/10.35940/ijrte.d7331.1111422. Accessed 13 Feb. 2023.
