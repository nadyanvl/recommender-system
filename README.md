# Laporan Proyek Machine Learning (Recommender System) - Nadya Novalina
# Sephora Products Recommender System
# Domain Proyek
Proyek ini bertujuan mengembangkan sistem rekomendasi produk untuk toko online Sephora. Fokus utama proyek ini adalah memberikan rekomendasi produk yang relevan dan personal kepada pelanggan Sephora berdasarkan preferensi dan karakteristik mereka. Dalam proyek ini, digunakan dua pendekatan berbeda, yaitu content-based filtering dan collaborative filtering [[1]](#daftar-pustaka).

Dalam metode content-based filtering, TfidfVectorizer digunakan untuk mengekstraksi fitur-fitur seperti bahan-bahan, kategori utama, dan kategori tersier dari setiap produk. Fitur-fitur ini kemudian dijadikan sebagai vektor representasi. Selanjutnya, kemiripan antara vektor fitur produk dihitung menggunakan metode cosine similarity. Dengan matriks kemiripan ini, model dapat merekomendasikan produk yang memiliki kesamaan tinggi dengan produk yang diminati oleh pengguna.

Di sisi lain, metode collaborative filtering digunakan untuk menganalisis perilaku rating (penilaian) pengguna terhadap produk menggunakan teknik Singular Value Decomposition (SVD) untuk mengidentifikasi pola kesamaan antara pengguna berdasarkan data rating. Dengan informasi ini, model dapat merekomendasikan produk yang mendapatkan rating tinggi dari pengguna dengan preferensi serupa. Pendekatan ini memungkinkan memberikan rekomendasi yang dipengaruhi oleh perilaku dan preferensi kolektif dari sekelompok pengguna.

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
