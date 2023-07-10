# Laporan Proyek Machine Learning (Recommender System) - Nadya Novalina
# Sephora Products Recommender System
# Domain Proyek
Proyek ini bertujuan mengembangkan sistem rekomendasi produk untuk toko online Sephora. Fokus utama proyek ini adalah memberikan rekomendasi produk yang relevan dan personal kepada pelanggan Sephora berdasarkan preferensi dan karakteristik mereka. Dalam proyek ini, digunakan dua pendekatan berbeda, yaitu content-based filtering dan collaborative filtering [[1]](#daftar-pustaka).

Dalam metode content-based filtering, TF-IDF Vectorizer digunakan untuk mengekstraksi fitur-fitur seperti bahan-bahan, kategori utama, dan kategori tersier dari setiap produk. Fitur-fitur ini kemudian dijadikan sebagai vektor representasi. Selanjutnya, kemiripan antara vektor fitur produk dihitung menggunakan metode cosine similarity. Dengan matriks kemiripan ini, model dapat merekomendasikan produk yang memiliki kesamaan tinggi dengan produk yang diminati oleh pengguna.

Di sisi lain, metode collaborative filtering digunakan untuk menganalisis perilaku rating (penilaian) pengguna terhadap produk menggunakan teknik Singular Value Decomposition (SVD) untuk mengidentifikasi pola kesamaan antara pengguna berdasarkan data rating. Dengan informasi ini, model dapat merekomendasikan produk yang mendapatkan rating tinggi dari pengguna dengan preferensi serupa. Pendekatan ini memungkinkan memberikan rekomendasi yang dipengaruhi oleh perilaku dan preferensi kolektif dari sekelompok pengguna.

Beberapa penelitian terkait dalam bidang sistem rekomendasi produk telah menunjukkan hasil yang menjanjikan. Misalnya, penelitian [[2]](#daftar-pustaka) mengimplementasikan metode content-based filtering dalam sistem rekomendasi produk kosmetik dan provides recommendations based on the users' specific requirements. Penelitian [[3]] mengusulkan pendekatan collaborative filtering pada produk di e-commerce menggunakan SVD algorithm with root mean square error (RMSE) of 1.055586 and by optimizing the hyperparameters of the algorithm yield a model with an RMSE value of 1.041784.

eberapa penelitian sebelumnya telah dilakukan dalam bidang sistem rekomendasi produk dan menunjukkan hasil yang menjanjikan. Misalnya, dalam penelitian [[2]](#daftar-pustaka), mereka mengimplementasikan metode content-based filtering pada sistem rekomendasi produk kosmetik dan memberikan rekomendasi berdasarkan kebutuhan khusus pengguna. Penelitian lainnya [[3]](#daftar-pustaka) mencoba pendekatan collaborative filtering pada produk di platform e-commerce dengan menggunakan algoritma SVD dan menghasilkan nilai root mean square error (RMSE) sebesar 1.055586 dan dengan mengoptimalkan hyperparameter algoritma, mereka berhasil mendapatkan model dengan nilai RMSE sebesar 1.041784.

Dengan menggabungkan pengetahuan dari penelitian-penelitian sebelumnya, proyek ini bertujuan untuk mengembangkan sistem rekomendasi produk yang dapat memberikan rekomendasi yang lebih relevan dan personal kepada pelanggan. Pendekatan yang digunakan dalam proyek ini akan memanfaatkan teknik content-based filtering dan collaborative filtering. Diharapkan sistem rekomendasi ini dapat meningkatkan pengalaman berbelanja pelanggan dan mendorong peningkatan penjualan di platform e-commerce.

# Business Understanding


## Problem Statements


## Goals


## Solution statements


# Data Understanding


# Data Preparation


# Modeling

# Evaluation

# Kesimpulan

# Daftar Pustaka
[[1]](https://iopscience.iop.org/article/10.1088/1742-6596/1897/1/012024/pdf) Abdul Hussien, Farah Tawfiq, et al. “Recommendation Systems for E-Commerce Systems an Overview.” Journal of Physics: Conference Series, vol. 1897, no. 1, 1 May 2021, p. 012024, https://doi.org/10.1088/1742-6596/1897/1/012024.
[[2]](http://www.ijceit.org/published/volume10/issue1/1Vol10No1.pdf) Patty, Joanna, et al. “Recommendations System for Purchase of Cosmetics Using Content- Based Filtering.” International Journal of Computer Engineering and Information Technology, vol. 10, no. 1, 2018, pp. 1–5, www.ijceit.org/published/volume10/issue1/1Vol10No1.pdf. Accessed 10 July 2023.
