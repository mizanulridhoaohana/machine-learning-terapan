# machine-learning-terapan
# Proyek Pertama Prediksi Harga Mobil Sport

#### Mizanul Ridho Aohana

Untuk memenuhi kebutuhan tugas proyek Predictive Analytics pada platform Dicoding, maka dibangun sebuah model machine learning untuk memprediksi Harga Mobil Sport berdasarkan kriteria tertentu.

## Domain Proyek

### Latar Belakang

<br>
<div><img src="https://github.com/mizanulridhoaohana/machine-learning-terapan/assets/112617513/2da3b096-c0d2-4bf4-868b-4b181711e36e" width="1000"/></div>
<br>

Pesatnya perkembangan teknologi otomotif dan minat masyarakat terhadap kinerja serta gaya hidup mewah menjadi hal yang makin banyak digandrungi oleh masyarakat, mobil sport telah menjadi pilihan unggulan bagi pecinta kendaraan yang mencari kombinasi sempurna antara kekuatan, prestasi, dan gaya.

Berdasarkan dataset yang tersedia, dapat diidentifikasi beberapa tren dan tantangan dalam industri mobil sport:

Tren:

1. Peningkatan Kinerja: Data menunjukkan bahwa mobil sport memiliki tenaga kuda yang bervariasi, dengan beberapa mobil memiliki tenaga kuda yang sangat tinggi. Ini mengindikasikan tren terus meningkatnya kinerja mobil sport, mungkin sebagai respons terhadap permintaan konsumen untuk mobil yang lebih kuat dan cepat.

2. Pendekatan Harga: Rentang harga yang luas dalam dataset menunjukkan keragaman harga mobil sport. Ini mencerminkan tren pasar yang memungkinkan adanya mobil sport dengan harga yang sesuai dengan berbagai kelas konsumen, dari yang lebih terjangkau hingga yang sangat mewah.

3. Elektrifikasi: Beberapa mobil dalam dataset memiliki mesin listrik. Ini mencerminkan tren industri yang semakin mengarah ke mobil sport yang lebih ramah lingkungan dengan teknologi elektrifikasi untuk meningkatkan efisiensi dan mengurangi emisi.

Tantangan:

1. Kompetisi Sengit: Pasar mobil sport sangat kompetitif dengan banyak produsen yang bersaing. Menentukan harga yang tepat dan menjaga daya saing menjadi tantangan yang signifikan.

2. Regulasi Lingkungan: Regulasi emisi dan ketahanan bahan bakar semakin ketat di banyak negara. Produsen mobil sport perlu menghadapi tantangan ini dengan mencari cara untuk mematuhi peraturan sambil tetap mempertahankan kinerja.

3. Perubahan Teknologi: Dengan adopsi teknologi baru, termasuk elektrifikasi, produsen mobil sport harus beradaptasi dengan perubahan tren dan teknologi, yang dapat memengaruhi desain dan kinerja mobil.

4. Fluktuasi Harga Bahan Baku: Harga bahan mentah, seperti logam dan komponen mobil, dapat fluktuatif. Ini dapat berdampak pada biaya produksi mobil sport dan akhirnya pada harga jual.

Berdasarkan analisis tren dan tantangan di atas, muncul berbagai tantangan menarik seperti kompetisi yang sengit, regulasi lingkungan, perubahan teknologi serta fluktuasi harga bahan baku. Dengan melihat tren yang berkembang saat ini seperti peningkatan kinerja, pendekatan harga dan elektrifikasi, tentunya berbagai perusahaan akan berlomba-lomba untuk melakukan inovasi dan optimasi dalam penjulan mereka agar mendapatkan hasil penjualan yang optimal. Dengan melihat fakta tersebut, maka diperlukan sebuah model atau inovasi baru yang dapat memperhitungkan berbagai faktor terkait untuk memperoleh hasil maksimal dalam industri penjulan mobil. Salah satu langkah yang dapat diambil adalah dengan menggunakan machine learning untuk membuat sebuah model prediksi harga mobil sport.

Menentukan harga yang sesuai untuk mobil sport adalah tantangan yang sering kali penuh ketidakpastian dalam industri otomotif. Dalam upaya untuk mengatasi kompleksitas ini, perusahaan otomotif dapat mengambil langkah maju dengan memanfaatkan model machine learning untuk melakukan prediksi harga mobil sport. Dengan pendekatan ini, harapannya adalah mampu mengidentifikasi nilai yang paling akurat berdasarkan karakteristik dan fitur-fitur yang dimiliki oleh setiap mobil. Prediksi harga ini akan memberikan panduan yang berharga bagi perusahaan dalam membuat keputusan pembelian dan penjualan yang lebih cerdas, dengan tujuan meningkatkan profitabilitas dan mengoptimalkan posisi mereka di pasar mobil sport yang kompetitif.

Model machine learning yang dibangun akan memungkinkan perusahaan untuk memperkirakan harga jual yang bersaing, sehingga mereka dapat memaksimalkan profitabilitas dan mengurangi risiko kerugian finansial. Dengan demikian, perusahaan dapat memahami dengan lebih baik dinamika pasar mobil sport, mengantisipasi fluktuasi harga, dan merespons perubahan tren konsumen dengan lebih cepat. Seiring waktu, model ini dapat menjadi alat yang sangat berharga dalam mendukung keputusan-strategis perusahaan, memungkinkan mereka untuk beroperasi secara lebih efisien dan efektif di pasar mobil sport yang selalu berubah.

Referensi : 

[1] [A. Amalia, M. Radhi, S. H. Sinurat, D. R. H. Sitompul, and E. Indra, “PREDIKSI HARGA MOBIL MENGGUNAKAN ALGORITMA REGRESSI DENGAN HYPER-PARAMETER TUNING”, JUSIKOM PRIMA, vol. 4, no. 2, pp. 28 -32, Feb. 2022.](http://jurnal.unprimdn.ac.id/index.php/JUSIKOM/article/view/2479/1459)

[2] [A. Chandak, P. Ganorkar, S. Sharma, A. Bagmar, and S. Tiwari, “Car price prediction using machine learning,” International Journal of Computer Sciences and Engineering, vol. 7, no. 5, pp. 444–450, 2019. doi:10.26438/ijcse/v7i5.444450 ](https://www.researchgate.net/publication/335799148_Car_Price_Prediction_Using_Machine_Learning)  

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

  Train test split adalah proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 1007 dibagi menjadi komposisi 805 (80%) untuk data latih dan 202 (20%) untuk data uji. Komposisi pembagian ini digunakan karena jumlah dataset yang dimiliki cenderung sedikit, sehingga untuk mengoptimalkan pelatihan model, maka komposisi training yang digunakan adalah 80% dan testing 20%.
  

## Modeling

+ Algoritma
  Penelitian ini melakukan pemodelan dengan 2 algoritma, yaitu Logistic Linear Regression, dan Decision Tree
  + Logistic Linear Regression

    Model Logistic Regression adalah algoritma machine learning yang digunakan untuk mengatasi masalah klasifikasi, terutama dalam konteks binary atau multiclass classification. Cara kerjanya melibatkan transformasi hasil linier dari fitur-fitur masukan menggunakan fungsi sigmoid (logistic function) yang menghasilkan probabilitas kelas. Selama pelatihan, model ini diperbarui untuk meminimalkan kesalahan antara probabilitas prediksi dan label sebenarnya dengan mengoptimalkan fungsi biaya. Kelebihan model ini termasuk interpretabilitas yang baik dan kinerja yang cukup baik pada masalah dengan hubungan linier antara fitur dan variabel target. . Proyek ini menggunakan [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) dengan memasukkan X_train dan y_train dalam membangun model.

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

Dimana,

SSRes adalah jumlah kuadrat dari sisa kesalahan.

SSTot adalah jumlah total kesalahan.

Contoh Interpretasi perhitungan skor R2 :
Asumsikan R2 = 0,68
Dapat dikatakan bahwa 68% variabilitas atribut keluaran dependen dapat dijelaskan oleh model, sedangkan 32% sisanya masih belum dapat dijelaskan.
R2 menunjukkan proporsi titik data yang terletak di dalam garis yang dibuat oleh persamaan regresi. Nilai R2 yang lebih tinggi diinginkan karena menunjukkan hasil yang lebih baik.

Berikut hasil evaluasi pada proyek ini :

+ R2 Score
  | model                       | r2_score |
  |-----------------------------|----------|
  | Logistic Linear Regression  | 0.720902 |
  | Decision Tree               | 0.918839 |
  | Decision Tree (Hyper-tuning)| 0.974387 |


Dari hasil evaluasi yang diperoleh dapat disimpulkan bahwa algoritma terbaik untuk memprediksi permasalahan ini adalah Decision Tree dengan Hyper-Tuning Parameter. Nilai R2 Score yang didapatkan adalah 0.97438. Ini adalah peningkatan yang signifikan jika dibandingkan dengan model Decision Tree dengan pengaturan default. R2 Score yang mendekati 1 menunjukkan bahwa model ini sangat baik dalam menjelaskan variasi dalam data. Hasil ini menunjukkan bahwa pengaturan parameter yang lebih optimal telah meningkatkan kinerja model secara signifikan.
