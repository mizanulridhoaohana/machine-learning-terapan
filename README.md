# machine-learning-terapan
# Proyek Pertama Prediksi Harga Mobil Sport

#### Mizanul Ridho Aohana

Untuk memenuhi kebutuhan tugas proyek Predictive Analytics pada platform Dicoding, maka dibangun sebuah model machine learning untuk memprediksi Harga Mobil Sport berdasarkan kriteria tertentu.

## Domain Proyek

### Latar Belakang

Pesatnya perkembangan teknologi otomotif dan minat masyarakat terhadap kinerja serta gaya hidup mewah menjadi hal yang makin banyak digandrungi oleh masyarakat, mobil sport telah menjadi pilihan unggulan bagi pecinta kendaraan yang mencari kombinasi sempurna antara kekuatan, prestasi, dan gaya.
Dalam upaya untuk memprediksi harga mobil sport, kami memanfaatkan dataset yang berisi informasi tentang harga mobil sport dari berbagai produsen terkemuka. Dataset ini mencakup informasi tentang merek dan model mobil, tahun produksi, ukuran mesin, tenaga kuda, torsi, waktu akselerasi 0-60 MPH, dan harga dalam USD. Data ini menjadi sumber penting untuk menganalisis harga mobil sport beragam dan mengidentifikasi tren dalam pasar. Dengan beragam informasi yang diberikan pada dataset, akan dilakukan studi mendalam untuk mengidentifikasi faktor-faktor yang paling memengaruhi harga mobil sport. Prediksi harga ini akan memberikan wawasan berharga bagi konsumen, pemasar, dan produsen mobil sport, serta para peneliti yang tertarik dalam dinamika pasar otomotif.

<br>
![image](https://github.com/mizanulridhoaohana/machine-learning-terapan/assets/112617513/2da3b096-c0d2-4bf4-868b-4b181711e36e)

[Referensi gambar]([https://www.pinterest.com/pin/357332551696985459/](https://storage.googleapis.com/kaggle-datasets-images/2988825/5144443/38723a1b40c5912b4224e3378a0eef8b/dataset-cover.jpg?t=2023-03-11-05-55-54))

<br>

Menentukan harga yang sesuai untuk mobil sport adalah tantangan yang sering kali penuh ketidakpastian dalam industri otomotif. Dalam upaya untuk mengatasi kompleksitas ini, perusahaan otomotif dapat mengambil langkah maju dengan memanfaatkan model machine learning untuk melakukan prediksi harga mobil sport. Dengan pendekatan ini, harapannya adalah mampu mengidentifikasi nilai yang paling akurat berdasarkan karakteristik dan fitur-fitur yang dimiliki oleh setiap mobil. Prediksi harga ini akan memberikan panduan yang berharga bagi perusahaan dalam membuat keputusan pembelian dan penjualan yang lebih cerdas, dengan tujuan meningkatkan profitabilitas dan mengoptimalkan posisi mereka di pasar mobil sport yang kompetitif.

Model machine learning yang dibangun akan memungkinkan perusahaan untuk memperkirakan harga jual yang bersaing, sehingga mereka dapat memaksimalkan profitabilitas dan mengurangi risiko kerugian finansial. Dengan demikian, perusahaan dapat memahami dengan lebih baik dinamika pasar mobil sport, mengantisipasi fluktuasi harga, dan merespons perubahan tren konsumen dengan lebih cepat. Seiring waktu, model ini dapat menjadi alat yang sangat berharga dalam mendukung keputusan-strategis perusahaan, memungkinkan mereka untuk beroperasi secara lebih efisien dan efektif di pasar mobil sport yang selalu berubah.

Referensi : 

+ [Prediksi Harga Mobil Menggunakan Algoritma Regression Hyper-Parameter Tuning](http://jurnal.unprimdn.ac.id/index.php/JUSIKOM/article/view/2479/1459)
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

### Exploratory Data Analytics (EDA)

Pada proses EDA, banyak digunakan analisis sebaran dan korelasi yang bisa dilihat langsung pada code yang sudah dilampirkan. Penggunaan visualisasi juga menjadi faktor penting dalam mempermudah memahami karakteristik dataset. Hal ini memberikan insight lanjutan untuk memproses data sebelum dilakukan proses prediksi.

## Data preparation

+ Encoding

  Encoding adalah teknik mengubah data kategorik menjadi data numerik dimana setiap kategori menjadi kolom baru dengan nilai unik yang diberikan secara berurutan. Fitur yang akan diubah menjadi numerik pada proyek ini adalah Car Make. Dengan mengubah data kategori numerik, perhitungan dalam data modeling dapat dipermudah dan disederhanakan.
  
+ Fill Missing Values

  Proses ini akan mengisi nilai missing value dari dataset yang sudah ada. Hal ini dilakukan untuk mengantisipasi overfit atau underfit pada model yang akan dibangun. Dalam hal ini, label yang termasuk adalah Engine Size (L), Horsepower, Torque (lb-ft), 0-60 MPH Time (seconds). Nilai akan di isi dengan nilai mean dari data tersebut.

+ Convert Data Type

  Pada proses ini, akan dilakukan penyeragaman tipe data pada setiap kolom, sehingga seluruh kolom dapat terstandarisasi dengan baik. Tipe data yang digunakan adalah int64 pada setiap kolom.

+ Add Additional Variable

  Proses ini bertujuan untuk menambahkan variabel baru yang dapat memberikan insight baru mengenai mobil yang sudah diproduksi. Variabel yang ditambahkan adalah Age, variabel ini merepresentasikan umur dari kendaraan sampai dengan saat ini.

+ Train Test Split

  Train test split adalah proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 1007 dibagi menjadi 805 untuk data latih dan 202 untuk data uji.
  

## Modeling

+ Algoritma
  Penelitian ini melakukan pemodelan dengan 2 algoritma, yaitu Logistic Linear Regression, dan Decision Tree
  + Logistic Linear Regression

    Regresi logistik linear, sering disebut sebagai regresi logistik, adalah metode statistik yang digunakan untuk tugas klasifikasi biner. Ini adalah jenis analisis regresi yang cocok untuk memodelkan hubungan antara variabel dependen biner (yaitu yang memiliki dua hasil mungkin, biasanya dikodekan sebagai 0 dan 1) dan satu atau lebih variabel independen. Regresi logistik digunakan untuk memprediksi probabilitas bahwa suatu input tertentu termasuk dalam salah satu dari dua kelas. Proyek ini menggunakan [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) dengan memasukkan X_train dan y_train dalam membangun model.

  + Decision Tree

    Decision tree (pohon keputusan) adalah sebuah model prediktif dalam ilmu data dan pembelajaran mesin yang digunakan untuk mengambil keputusan berdasarkan aturan yang didefinisikan dalam bentuk struktur pohon. Model ini digunakan untuk masalah klasifikasi dan regresi, serta dapat digunakan untuk tugas pengambilan keputusan yang melibatkan berbagai variabel dan skenario. Proyek ini menggunakan [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).



+ Hyperparameter Tuning (Grid Search)
  Hyperparameter tuning adalah cara untuk mendapatkan parameter terbaik dari algoritma dalam membangun model. Salah satu teknik dalam hyperparameter tuning yang digunakan dalam proyek ini adalah grid search. 
  
  Dalam kasus ini, parameter tuning dilakukan pada model yang menggunakan algoritma decision tree. Hal ini didasarkan pada hasil sebelumnya yang sudah didapatkan, Decision Tree memiliki hasil r2_score yang lebih baik jika dibandingkan dengan Logistic Linear Regression. Berikut adalah hasil dari Grid Search pada proyek ini :
  | model    | best_params                                                     |
  |----------|-----------------------------------------------------------------|
  | Decision | {'max_depth': None, 'max_features': 'sqrt',                     |
  | Tree     |  'min_samples_leaf': 1, 'min_samples_split':2}                  |


## Evaluation

Metrik evaluasi yang digunakan pada proyek ini [R2 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html). R^2 (R-squared), juga dikenal sebagai koefisien determinasi, adalah metrik evaluasi yang digunakan dalam statistik dan analisis regresi untuk mengukur sejauh mana model regresi cocok dengan data yang diamati. R^2 score mengukur seberapa baik variabilitas dalam data independen (variabel prediktor) menjelaskan variabilitas dalam data dependen (variabel target). R^2 score berkisar antara 0 hingga 1, dan semakin mendekati 1, semakin baik model regresinya sesuai dengan data. Berikut formula R2 Score :
![image](https://github.com/mizanulridhoaohana/machine-learning-terapan/assets/112617513/2ccbd4bb-0da4-4bb6-b2df-3a1194819e86)

Berikut hasil evaluasi pada proyek ini :

+ R2 Score
  | model                       | r2_score |
  |-----------------------------|----------|
  | Logistic Linear Regression  | 0.720902 |
  | Decision Tree               | 0.918839 |
  | Decision Tree (Hyper-tuning)| 0.974387 |


Dari hasil evaluasi yang diperoleh dapat disimpulkan bahwa algoritma terbaik untuk memprediksi permasalahan ini adalah Decision Tree dengan Hyper-Tuning Parameter. Nilai r2_score yang didapatkan adalah 0.974387, nilai ini lebih tinggi jika dibandingkan dengan hasil yang diberikan oleh model Decision Tree tanpa Hyper-Tuning Parameter.
