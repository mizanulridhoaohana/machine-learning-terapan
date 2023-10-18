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

## Data Understanding & Removing Outlier

Dataset yang digunakan dalam proyek ini merupakan data harga mobil sport dengan 8 variabel. Dataset ini dapat diunduh di [Kaggle : Sports Car Prices dataset](https://www.kaggle.com/datasets/rkiattisak/sports-car-prices-dataset).

Berikut informasi pada dataset :

+ Dataset memiliki format CSV (Comma-Seperated Values).
+ Dataset memiliki 1007 sample dengan 8 fitur.
+ Dataset memiliki 6 fitur bertipe string, 1 fitur bertipe decimal dan 1 fitur bertipe Integer.
+ Terdapat beberapa missing value dalam dataset.

### Variable - variable pada dataset

+ Car Make 
+ Car Model
+ Year
+ Engine Size(L)
+ Horsepower
+ Torque (lb-ft)
+ 0-60 MPH Time (seconds)
+ Price (in USD)

Dari ke 12 fitur dapat dilihat bahwa fitur Point of Contract dan Posted On tidak mempengaruhi harga sewa rumah sehingga akan dihapus. Hal ini dikarenakan kedua fitur tersebut tidak diperlukan dalam membangun model prediksi harga sewa.

### Univariate Analysis

Univariate Analysis adalah menganalisis setiap fitur secara terpisah.

#### Analisis jumlah nilai unique pada setiap fitur kategorik

Fitur kategorik City, Furnishing Status, dan Tenant Preferred memiliki sebaran sample yang cukup merata.
<div><img src="https://user-images.githubusercontent.com/107544829/188319357-fc12fffa-b709-4584-8363-778bc678b328.png" width="220"/></div> <div><img src="https://user-images.githubusercontent.com/107544829/188319651-02ddb783-da3d-41ed-9b5f-9525aaaf9ed1.png" width="220"/></div> <div><img src="https://user-images.githubusercontent.com/107544829/188319750-1f080942-7826-4eaf-a021-8b9f938a861a.png" width="220"/></div><br />

Berikut adalah fitur dengan sample yang tidak merata :

+ Area Type
  <div><img src="https://user-images.githubusercontent.com/107544829/188318629-f474b626-a16a-4971-ab42-2c183d22b744.png" width="220"/></div>
  Hanya terdapat 2 sample Built Area pada fitur Area Type. Untuk menghindari high dimensional data, maka kedua sample ini akan dihapus.

+ Floor dan Area Locality
  <div><img src="https://user-images.githubusercontent.com/107544829/188319871-603b24b8-26b2-449b-b42e-59501a4803a7.png" width="220"/></div>
   <div><img src="https://user-images.githubusercontent.com/107544829/188319880-3226bd04-920e-4050-b5ab-38dec02fc524.png" width="220"/></div>
  Fitur Floor dan Area Locality memiliki banyak sekali nilai unique. Untuk menghindari high dimensional data, maka kedua fitur ini akan dihapus.

#### Analisis sebaran pada setiap fitur numerik

<div><img src="https://user-images.githubusercontent.com/107544829/188320722-451f25bd-de65-4e09-9d0a-9d8835249492.png" width="450"/></div><br />
Berikut analisis dari grafik di atas :

+ Sebagian besar rumah memiliki 1 sampai 3 BHK dan 1 sampai 3 kamar mandi.
+ Sebagian besar rumah memiliki luas di bawah 2000 sqft.
+ Rentang harga sewa cukup tinggi, yaitu dari 1200 hingga 3500000. Namun, rata-rata harga rumah hanya 35003. Distribusi harga yang kurang bagus seperti ini dapat berimplikasi pada model.

### Multivariate Analysis

Multivariate Analysis menunjukkan hubungan antara dua atau lebih fitur dalam data.

#### Analisis fitur numerik

+ Fitur Size dan BHK (Menghapus BHK Outlier)
  Kedua fitur ini dianalisis karena tidak biasa untuk rumah dengan 1 BHK memiliki luas 100 sqft. Untuk itu ditentukan treshold atau batas 300 sqft/bhk. Data yang berada di bawah batas akan dihapus. Hal ini menyebabkan berkurangnya jumlah sample sebesar 548.

+ Fitur Size dan Rent (Menghapus Price per sqft Outlier)
  Untuk memudahkan dalam mendeteksi outlier, maka dibuat fitur baru 'Price_per_sqft' dari kedua fitur tersebut untuk menganalisis harga sewa per luas sqft.
  <div><img src="https://user-images.githubusercontent.com/107544829/188323140-6174b592-4c7b-4671-9acb-b49a621d2aba.png" width="220"/></div>
  Dari sini dapat terlihat bahwa harga 571 per sqft sangat rendah dan harga 1400000 per sqft sangat tinggi. Untuk itu penghapusan outlier price per sqft outlier dengan mean dan one standard deviation yang telah dikelompokkan berdasarkan kota. Hal ini menyebabkan berkurangnya jumlah sample sebesar 497.

+ Fitur Bathroom dan BHK (Menghapus Bathroom Outlier)
  Kedua fitur ini dianalisis karena tidak biasa untuk rumah dengan 2 BHK memiliki 4 kamar mandi. Untuk itu ditentukan batas bahwa jumlah kamar mandi tidak boleh melebihi jumlah BHK + 2. Hal ini menyebabkan berkurangnya sample sebesar 3.
  
+ Melihat kolerasi antara semua fitur numerik
  <div><img src="https://user-images.githubusercontent.com/107544829/188323797-8186246a-8cdd-4232-8bc7-bce615cf92d0.png" width="350"/></div>
  Fitur BHK, Size, dan Bathroom berkorelasi tidak signifikan dengan fitur target (Rent). Hal ini mungkin   disebabkan oleh kurangnya data dalam penelitian ini.Fitur BHK dan Bathroom berkolerasi signifikan dengan fitur size. Hal ini sudah sesuai harapan dari penghapusan outlier yang sudah dilakukan sebelumnya.

#### Analisis fitur kategorik

Analisis ini dilakukan untuk melihat kolerasi antara fitur kategorik dengan fitur target (Rent).

+ Fitur Area Type
  <div><img src="https://user-images.githubusercontent.com/107544829/188324455-9ae90db3-681a-4f14-bee0-0daaaec86490.png" width="500"/></div>
  Fitur Area Type memiliki pengaruh yang kecil terhadap rata-rata harga sewa.

+ Fitur City
  <div><img src="https://user-images.githubusercontent.com/107544829/188324564-b978b637-122b-403d-a760-eb0f7838bd95.png" width="500"/></div>
  Fitur City memiliki pengaruh cukup besar terhadap rata-rata harga sewa, terutama jika rumah berada di kota Mumbai. Hal ini dibuktikan dengan sebaran rumah yang mencapai harga tertinggi di kota Mumbai. Mumbai merupakan kota paling mahal di India untuk ditinggali, diikuti dengan Delhi.

  Referensi : [Ini Adalah Kota Termahal Untuk Hidup Di India](https://id.yourtripagent.com/these-are-most-expensive-cities-to-live-in-india-4734)

+ Fitur Furnishing Status
  <div><img src="https://user-images.githubusercontent.com/107544829/188324598-a765e404-4140-4518-91eb-fd298ba9d089.png" width="500"/></div>
  Fitur Furnishing Status memiliki pengaruh cukup besar terhadap rata-rata harga sewa. Merupakan hal biasa bila rumah yang memiliki perabotan lengkap akan diberi harga sewa lebih tinggi daripada rumah tanpa perabotan.

+ Fitur Tenant Preferred
  <div><img src="https://user-images.githubusercontent.com/107544829/188324642-0de4fe01-20c8-4560-981a-d6b0822d56ff.png" width="500"/></div>
  Fitur Tenant Preferred memiliki pengaruh yang lumayan terhadap rata-rata harga sewa. Dari grafik dapat terlihat bahwa rumah yang sangat disarankan untuk disewa oleh keluarga memiliki rata-rata harga sewa yang lebih mahal dibanding lainnya.

## Data preparation

+ One Hot Encoding

  One hot encoding adalah teknik mengubah data kategorik menjadi data numerik dimana setiap kategori menjadi kolom baru dengan nilai 0 atau 1. Fitur yang akan diubah menjadi numerik pada proyek ini adalah Area Type, City, Furnishing Status, dan Tenant Preferred.
  
+ Train Test Split

  Train test split aja proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 3696 dibagi menjadi 3511 untuk data latih dan 185 untuk data uji.
  
+ Normalization

  Algoritma machine learning akan memiliki performa lebih baik dan bekerja lebih cepat jika dimodelkan dengan data seragam yang memiliki skala relatif sama. Salah satu teknik normalisasi yang digunakan pada proyek ini adalah Standarisasi dengan sklearn.preprocessing.StandardScaler.

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