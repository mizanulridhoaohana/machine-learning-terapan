# machine-learning-terapan
# Proyek Pertama Prediksi Harga Mobil Sport

#### Mizanul Ridho Aohana

Untuk memenuhi kebutuhan tugas proyek Predictive Analytics pada platform Dicoding, maka dibangun sebuah model machine learning untuk memprediksi Harga Mobil Sport berdasarkan kriteria tertentu.

## Domain Proyek

### Latar Belakang

Pesatnya perkembangan teknologi otomotif dan minat masyarakat terhadap kinerja serta gaya hidup mewah menjadi hal yang makin banyak digandrungi oleh masyarakat, mobil sport telah menjadi pilihan unggulan bagi pecinta kendaraan yang mencari kombinasi sempurna antara kekuatan, prestasi, dan gaya.
Dalam upaya untuk memprediksi harga mobil sport, kami memanfaatkan dataset yang berisi informasi tentang harga mobil sport dari berbagai produsen terkemuka. Dataset ini mencakup informasi tentang merek dan model mobil, tahun produksi, ukuran mesin, tenaga kuda, torsi, waktu akselerasi 0-60 MPH, dan harga dalam USD. Data ini menjadi sumber penting untuk menganalisis harga mobil sport beragam dan mengidentifikasi tren dalam pasar. Dengan beragam informasi yang diberikan pada dataset, akan dilakukan studi mendalam untuk mengidentifikasi faktor-faktor yang paling memengaruhi harga mobil sport. Prediksi harga ini akan memberikan wawasan berharga bagi konsumen, pemasar, dan produsen mobil sport, serta para peneliti yang tertarik dalam dinamika pasar otomotif.

<br>

<div><img src="https://www.pinterest.com/pin/357332551696985459/" width="1000"/></div>

[Referensi gambar](https://www.pinterest.com/pin/357332551696985459/)

<br>

Menentukan harga yang sesuai untuk mobil sport adalah tantangan yang sering kali penuh ketidakpastian dalam industri otomotif. Dalam upaya untuk mengatasi kompleksitas ini, perusahaan otomotif dapat mengambil langkah maju dengan memanfaatkan model machine learning untuk melakukan prediksi harga mobil sport. Dengan pendekatan ini, harapannya adalah mampu mengidentifikasi nilai yang paling akurat berdasarkan karakteristik dan fitur-fitur yang dimiliki oleh setiap mobil. Prediksi harga ini akan memberikan panduan yang berharga bagi perusahaan dalam membuat keputusan pembelian dan penjualan yang lebih cerdas, dengan tujuan meningkatkan profitabilitas dan mengoptimalkan posisi mereka di pasar mobil sport yang kompetitif.

Model machine learning yang dibangun akan memungkinkan perusahaan untuk memperkirakan harga jual yang bersaing, sehingga mereka dapat memaksimalkan profitabilitas dan mengurangi risiko kerugian finansial. Dengan demikian, perusahaan dapat memahami dengan lebih baik dinamika pasar mobil sport, mengantisipasi fluktuasi harga, dan merespons perubahan tren konsumen dengan lebih cepat. Seiring waktu, model ini dapat menjadi alat yang sangat berharga dalam mendukung keputusan-strategis perusahaan, memungkinkan mereka untuk beroperasi secara lebih efisien dan efektif di pasar mobil sport yang selalu berubah.

Referensi : 

+ [PREDIKSI HARGA MOBIL MENGGUNAKAN ALGORITMA REGRESSIDENGAN HYPER-PARAMETER TUNING](http://jurnal.unprimdn.ac.id/index.php/JUSIKOM/article/view/2479/1459)
+ [Car Price Prediction Using Machine Learning](https://www.researchgate.net/publication/335799148_Car_Price_Prediction_Using_Machine_Learning)  

## Business Understanding

Proyek ini dibangun untuk perusahaan dengan karakteristik bisnis sebagai berikut :

+ Perusahaan yang aktif melakukan produksi mobil untuk kepentingan komersial.
+ Perusahaan yang membuka jasa konsultasi harga mobil sport ke konsumen.

### Problem Statement

1. Variabel apa saja yang berpengaru terhadap harga mobil sport?
2. Bagaimana cara memproses data agar dapat dilatih dengan baik oleh model?
3. Berapa harga mobil sport di pasaran berdasarkan karakteristik tertentu?

### Goals

1. Mengetahui variabel yang paling berpengaruh pada harga sewa rumah atau apartemen.
2. Melakukan persiapan data untuk dapat dilatih oleh model.
3. Membuat model machine learning yang dapat memprediksi harga mobil sport seakurat mungkin berdasarkan karakteristik tertentu.

### Solution Statement

1. Menganalisis data dengan melakukan univariate analysis dan multivariate analysis. Memahami data juga dapat dilakukan dengan visualisasi. Memahami data dapat membantu untuk mengetahui kolerasi antar fitur dan mendeteksi outlier.
2. Menyiapkan data agar bisa digunakan dalam membangun model.
3. Algoritma yang dipakai dalam proyek ini adalah Logistic Linear Regression dan Decision Tree Algorithm.

## Data Understanding

Dataset yang digunakan dalam proyek ini merupakan data harga mobil sport dengan 8 variabel. Dataset ini dapat diunduh di [Kaggle : Sports Car Prices dataset](https://www.kaggle.com/datasets/rkiattisak/sports-car-prices-dataset).

Berikut informasi pada dataset :

+ Dataset memiliki format CSV (Comma-Seperated Values).
+ Dataset memiliki 1007 sample dengan 8 fitur.
+ Dataset memiliki 6 fitur bertipe string, 1 fitur bertipe decimal dan 1 fitur bertipe Integer.
+ Terdapat beberapa missing value dalam dataset.

### Variabel pada dataset

+ Car Make: Merek mobil sport yang mewakili merek atau perusahaan yang memproduksi mobil tersebut.
+ Car Model: Model mobil sport yang mewakili versi atau varian tertentu dari mobil yang diproduksi oleh pabrikan.
+ Year: Tahun produksi mobil sport, yang menunjukkan tahun model saat mobil tersebut pertama kali diperkenalkan atau tersedia untuk dibeli.
+ Engine Size(L): Ukuran mesin mobil sport dalam liter, yang mewakili volume silinder mesin. Ukuran mesin yang lebih besar biasanya menunjukkan tenaga dan kinerja yang lebih tinggi. 
+ Horsepower: Horsepower mobil sport, yang mewakili keluaran tenaga mesin mobil. Tenaga kuda yang lebih tinggi biasanya menunjukkan akselerasi yang lebih cepat dan kecepatan tertinggi yang lebih tinggi.
+ Torque (lb-ft): Torsi mobil sport dalam pound-feet, yang mewakili gaya putaran yang dihasilkan oleh mesin. Nilai torsi yang lebih tinggi biasanya menunjukkan akselerasi yang lebih kuat dan penanganan yang lebih baik.
+ 0-60 MPH Time (seconds): Waktu yang diperlukan mobil sport untuk berakselerasi dari 0 hingga 60 mil per jam, yang merupakan ukuran umum akselerasi dan performa. Waktu 0-60 MPH yang lebih rendah biasanya menunjukkan akselerasi yang lebih cepat dan kinerja yang lebih baik.
+ Price (in USD): Harga mobil sport dalam dolar AS, yang mewakili biaya pembelian mobil.

## Data preparation

+ Encoding

  Encoding adalah teknik mengubah data kategorik menjadi data numerik dimana setiap kategori menjadi kolom baru dengan nilai unik yang diberikan secara berurutan. Fitur yang akan diubah menjadi numerik pada proyek ini adalah Car Make.
  
+ Fill Missing Values

  Proses ini akan mengisi nilai missing value dari dataset yang sudah ada. Dalam hal ini, label yang termasuk adalah Engine Size (L), Horsepower, Torque (lb-ft), 0-60 MPH Time (seconds). Nilai akan di isi dengan nilai mean dari data tersebut.

+ Convert Data Type

  Pada proses ini, akan dilakukan penyeragaman tipe data pada setiap kolom, sehingga seluruh kolom dapat terstandarisasi dengan baik. Tipe data yang digunakan adalah int64 pada setiap kolom.

+ Train Test Split

  Train test split adalah proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 3696 dibagi menjadi 3511 untuk data latih dan 185 untuk data uji.
  

## Modeling

+ Algoritma
  Penelitian ini melakukan pemodelan dengan 3 algoritma, yaitu K-Nearest Neighbour, Random Forest, dan
  + K-Nearest Neighbour
    K-Nearest Neighbour bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat. Proyek ini menggunakan [sklearn.neighbors.KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_neighbors` = Jumlah k tetangga tedekat.

  + Random Forest
    Algoritma random forest adalah teknik dalam machine learning dengan metode ensemble. Teknik ini beroperasi dengan membangun banyak decision tree pada waktu pelatihan. Proyek ini menggunakan [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
    + `max_depth` = Kedalaman maksimum setiap tree.
    + `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.

  + Adaboost
    AdaBoost juga disebut Adaptive Boosting adalah teknik dalam machine learning dengan metode ensemble.  Algoritma yang paling umum digunakan dengan AdaBoost adalah pohon keputusan (decision trees) satu tingkat yang berarti memiliki pohon Keputusan dengan hanya 1 split. Pohon-pohon ini juga disebut Decision Stumps. Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi dengan cara menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) secara berurutan sehingga membentuk suatu model yang kuat (strong ensemble learner). Proyek ini menggunakan [sklearn.ensemble.AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
    + `learning_rate` = Learning rate memperkuat kontribusi setiap regressor.
    + `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.

+ Hyperparameter Tuning (Grid Search)
  Hyperparameter tuning adalah cara untuk mendapatkan parameter terbaik dari algoritma dalam membangun model. Salah satu teknik dalam hyperparameter tuning yang digunakan dalam proyek ini adalah grid search. Berikut adalah hasil dari Grid Search pada proyek ini :
  | model    | best_params                                                     |
  |----------|-----------------------------------------------------------------|
  | knn      | {'n_neighbors': 7}                                              |
  | boosting | {'learning_rate': 0.1, 'n_estimators': 100, 'random_state': 11} |
  | rf       | {'max_depth': 8, 'n_estimators': 25, 'random_stste': 11}        |

## Evaluation

Metrik evaluasi yang digunakan pada proyek ini adalah akurasi dan mean squared error (MSE). Akurasi menentukan tingkat kemiripan antara hasil prediksi dengan nilai yang sebenarnya (y_test). Mean squared error (MSE) mengukur error dalam model statistik dengan cara menghitung rata-rata error dari kuadrat hasil aktual dikurang hasil prediksi. Berikut formulan MSE :
<div><img src="https://user-images.githubusercontent.com/107544829/188412654-f5dc0ae1-901b-470e-aae5-1f6b5fb68b4d.png" width="300"/></div>

Berikut hasil evaluasi pada proyek ini :

+ Akurasi
  | model    | accuracy |
  |----------|----------|
  | knn      | 0.726775 |
  | boosting | 0.898556 |
  | rf       | 0.932057 |

+ Mean Squared Error (MSE)
  <div><img src="https://user-images.githubusercontent.com/107544829/188413846-7d5454b5-7f83-488e-836f-4f3593eb3d5d.png" width="300"/></div>

Dari hasil evaluasi dapat dilihat bahwa model dengan algoritma Random Forest memiliki akurasi lebih tinggi tinggi dan tingkat error lebih kecil dibandingkan algoritma lainnya dalam proyek ini.