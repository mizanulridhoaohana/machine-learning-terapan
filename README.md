# machine-learning-terapan
# Proyek Pertama Prediksi Harga Mobil Sport

#### Mizanul Ridho Aohana

Untuk memenuhi kebutuhan tugas proyek _Predictive Analytics_ pada platform Dicoding, maka dibangun sebuah model _machine learning_ untuk memprediksi Harga Mobil Sport berdasarkan kriteria tertentu.

## Domain Proyek

### Latar Belakang

<br>
<div><img src="https://github.com/mizanulridhoaohana/machine-learning-terapan/assets/112617513/2da3b096-c0d2-4bf4-868b-4b181711e36e" width="1000"/></div>
Gambar 1. Ilustrasi Mobil Sport
<br>

Pesatnya perkembangan teknologi otomotif dan minat masyarakat terhadap kinerja serta gaya hidup mewah menjadi hal yang makin banyak digandrungi oleh masyarakat, mobil _sport_ telah menjadi pilihan unggulan bagi pecinta kendaraan yang mencari kombinasi sempurna antara kekuatan, prestasi, dan gaya.

Berdasarkan dataset yang tersedia, dapat diidentifikasi beberapa tren dan tantangan dalam industri mobil _sport_:

Tren:

1. Peningkatan Kinerja: Data menunjukkan bahwa mobil _sport_ memiliki tenaga kuda yang bervariasi, dengan beberapa mobil memiliki tenaga kuda yang sangat tinggi. Ini mengindikasikan tren terus meningkatnya kinerja mobil _sport_, mungkin sebagai respons terhadap permintaan konsumen untuk mobil yang lebih kuat dan cepat.

2. Pendekatan Harga: Rentang harga yang luas dalam dataset menunjukkan keragaman harga mobil _sport_. Ini mencerminkan tren pasar yang memungkinkan adanya mobil _sport_ dengan harga yang sesuai dengan berbagai kelas konsumen, dari yang lebih terjangkau hingga yang sangat mewah.

3. Elektrifikasi: Beberapa mobil dalam dataset memiliki mesin listrik. Ini mencerminkan tren industri yang semakin mengarah ke mobil _sport_ yang lebih ramah lingkungan dengan teknologi elektrifikasi untuk meningkatkan efisiensi dan mengurangi emisi.

Tantangan:

1. Kompetisi Sengit: Pasar mobil sport sangat kompetitif dengan banyak produsen yang bersaing. Menentukan harga yang tepat dan menjaga daya saing menjadi tantangan yang signifikan.

2. Regulasi Lingkungan: Regulasi emisi dan ketahanan bahan bakar semakin ketat di banyak negara. Produsen mobil _sport_ perlu menghadapi tantangan ini dengan mencari cara untuk mematuhi peraturan sambil tetap mempertahankan kinerja.

3. Perubahan Teknologi: Dengan adopsi teknologi baru, termasuk elektrifikasi, produsen mobil _sport_ harus beradaptasi dengan perubahan tren dan teknologi, yang dapat memengaruhi desain dan kinerja mobil.

4. Fluktuasi Harga Bahan Baku: Harga bahan mentah, seperti logam dan komponen mobil, dapat fluktuatif. Ini dapat berdampak pada biaya produksi mobil _sport_ dan akhirnya pada harga jual.

Berdasarkan analisis tren dan tantangan di atas, muncul berbagai tantangan menarik seperti kompetisi yang sengit, regulasi lingkungan, perubahan teknologi serta fluktuasi harga bahan baku. Dengan melihat tren yang berkembang saat ini seperti peningkatan kinerja, pendekatan harga dan elektrifikasi, tentunya berbagai perusahaan akan berlomba-lomba untuk melakukan inovasi dan optimasi dalam penjulan mereka agar mendapatkan hasil penjualan yang optimal. Dengan melihat fakta tersebut, maka diperlukan sebuah model atau inovasi baru yang dapat memperhitungkan berbagai faktor terkait untuk memperoleh hasil maksimal dalam industri penjulan mobil. Salah satu langkah yang dapat diambil adalah dengan menggunakan machine learning untuk membuat sebuah model prediksi harga mobil _sport_.

Menentukan harga yang sesuai untuk mobil _sport_ adalah tantangan yang sering kali penuh ketidakpastian dalam industri otomotif. Dalam upaya untuk mengatasi kompleksitas ini, perusahaan otomotif dapat mengambil langkah maju dengan memanfaatkan model machine learning untuk melakukan prediksi harga mobil sport. Dengan pendekatan ini, harapannya adalah mampu mengidentifikasi nilai yang paling akurat berdasarkan karakteristik dan fitur-fitur yang dimiliki oleh setiap mobil. Prediksi harga ini akan memberikan panduan yang berharga bagi perusahaan dalam membuat keputusan pembelian dan penjualan yang lebih cerdas, dengan tujuan meningkatkan profitabilitas dan mengoptimalkan posisi mereka di pasar mobil _sport_ yang kompetitif.

Model _machine learning_ yang dibangun akan memungkinkan perusahaan untuk memperkirakan harga jual yang bersaing, sehingga mereka dapat memaksimalkan profitabilitas dan mengurangi risiko kerugian finansial. Dengan demikian, perusahaan dapat memahami dengan lebih baik dinamika pasar mobil _sport_, mengantisipasi fluktuasi harga, dan merespons perubahan tren konsumen dengan lebih cepat. Seiring waktu, model ini dapat menjadi alat yang sangat berharga dalam mendukung keputusan-strategis perusahaan, memungkinkan mereka untuk beroperasi secara lebih efisien dan efektif di pasar mobil _sport_ yang selalu berubah.

Referensi : 

[1] [A. Amalia, M. Radhi, S. H. Sinurat, D. R. H. Sitompul, and E. Indra, “PREDIKSI HARGA MOBIL MENGGUNAKAN ALGORITMA REGRESSI DENGAN HYPER-PARAMETER TUNING”, JUSIKOM PRIMA, vol. 4, no. 2, pp. 28 -32, Feb. 2022.](http://jurnal.unprimdn.ac.id/index.php/JUSIKOM/article/view/2479/1459)

[2] [A. Chandak, P. Ganorkar, S. Sharma, A. Bagmar, and S. Tiwari, “Car price prediction using machine learning,” International Journal of Computer Sciences and Engineering, vol. 7, no. 5, pp. 444–450, 2019. doi:10.26438/ijcse/v7i5.444450 ](https://www.researchgate.net/publication/335799148_Car_Price_Prediction_Using_Machine_Learning)  

## Business Understanding

Proyek ini dibangun untuk perusahaan dengan karakteristik bisnis sebagai berikut :

+ Perusahaan yang aktif melakukan produksi mobil untuk kepentingan komersial.
+ Perusahaan yang membuka jasa konsultasi harga mobil _sport_ ke konsumen.

### Problem Statement

1. Informasi apa saja yang dapat diperoleh dari dataset ini, terutama mengenai hubungan antara fitur-fitur tertentu dengan harga mobil _sport_ ?
2. Dengan informasi yang ada pada dataset, bagaimana cara mengoptimalkan produksi mobil _sport_ agar dapat meningkatkan valuasi perusahaan ?
3. Bagaimana penggunaan algoritma _machine learning_ dapat meningkatkan produksi perusahaan ?

### Goals

1. Menentukan variabel yang paling berpengaruh terhadap harga mobil sport, dan mengidentifikasi korelasi antara variabel-variabel tersebut.
2. Mengukur peningkatan profitabilitas dan efisiensi produksi mobil sport sebagai hasil dari peningkatan pemahaman tentang faktor-faktor yang memengaruhi harga.
3. Mengembangkan model _machine learning_ yang mampu memprediksi harga mobil _sport_ dengan tingkat akurasi yang tinggi berdasarkan karakteristik tertentu, sehingga memungkinkan perusahaan untuk menyesuaikan strategi produksi dan penetapan harga.

### Solution Statement

1. Untuk mencapai tujuan ini, langkah awal adalah melakukan analisis data yang mendalam, termasuk analisis _univariate_, _multivariate_, dan visualisasi data. Analisis ini akan membantu dalam mengidentifikasi hubungan antar fitur dan mendeteksi _outlier_ yang mungkin memengaruhi harga mobil _sport_.
2. Selanjutnya, perlu dilakukan persiapan data agar sesuai untuk pelatihan model _machine learning_. Ini termasuk pemrosesan data, penanganan nilai-nilai yang hilang, dan pemilihan fitur yang relevan.
3. Dalam masalah ini akan digunakan algoritma _machine learning_ seperti _Logistic Linear Regression_ dan _Decision Tree_ untuk membangun model yang mampu memprediksi harga mobil _sport_ dengan akurasi tinggi. Model ini akan memberikan wawasan berharga tentang faktor-faktor yang memengaruhi harga dan membantu perusahaan dalam mengoptimalkan produksi dan penetapan harga.

## Data Understanding

Dataset yang digunakan dalam proyek ini merupakan data harga mobil sport dengan 8 variabel. Dataset ini dapat diunduh di [Kaggle : Sports Car Prices dataset](https://www.kaggle.com/datasets/rkiattisak/sports-car-prices-dataset).

Berikut informasi pada dataset :

+ Dataset memiliki format CSV (_Comma-Seperated Values_).
+ Dataset memiliki 1007 sample dengan 8 fitur.
+ Dataset memiliki 6 fitur bertipe string, 1 fitur bertipe _decimal_ dan 1 fitur bertipe _Integer_.
+ Terdapat beberapa missing value dalam dataset.

### Variabel pada dataset

+ _Car Make_: Merek mobil _sport_ yang mewakili merek atau perusahaan yang memproduksi mobil tersebut.
+ _Car Model_: Model mobil _sport_ yang mewakili versi atau varian tertentu dari mobil yang diproduksi oleh pabrikan.
+ _Year_: Tahun produksi mobil _sport_, yang menunjukkan tahun model saat mobil tersebut pertama kali diperkenalkan atau tersedia untuk dibeli.
+ _Engine Size(L)_: Ukuran mesin mobil sport dalam liter, yang mewakili volume silinder mesin. Ukuran mesin yang lebih besar biasanya menunjukkan tenaga dan kinerja yang lebih tinggi. 
+ _Horsepower_: _Horsepower_ mobil _sport_, yang mewakili keluaran tenaga mesin mobil. Tenaga kuda yang lebih tinggi biasanya menunjukkan akselerasi yang lebih cepat dan kecepatan tertinggi yang lebih tinggi.
+ _Torque_ (lb-ft): Torsi mobil _sport_ dalam _pound-feet_, yang mewakili gaya putaran yang dihasilkan oleh mesin. Nilai torsi yang lebih tinggi biasanya menunjukkan akselerasi yang lebih kuat dan penanganan yang lebih baik.
+ 0-60 MPH _Time_ (_seconds_): Waktu yang diperlukan mobil _sport_ untuk berakselerasi dari 0 hingga 60 mil per jam, yang merupakan ukuran umum akselerasi dan performa. Waktu 0-60 MPH yang lebih rendah biasanya menunjukkan akselerasi yang lebih cepat dan kinerja yang lebih baik.
+ _Price_ (_in USD_): Harga mobil _sport_ dalam dolar AS, yang mewakili biaya pembelian mobil.

### Exploratory Data Analytics (EDA)

Pada proses EDA, banyak digunakan analisis sebaran dan korelasi yang bisa dilihat langsung pada code yang sudah dilampirkan. Penggunaan visualisasi juga menjadi faktor penting dalam mempermudah memahami karakteristik dataset. Hal ini memberikan _insight_ lanjutan untuk memproses data sebelum dilakukan proses prediksi.

## Data preparation

+ Encoding

  _Encoding_ adalah teknik mengubah data kategorik menjadi data numerik dimana setiap kategori menjadi kolom baru dengan nilai unik yang diberikan secara berurutan. Fitur yang akan diubah menjadi numerik pada proyek ini adalah _Car Make_. Dengan mengubah data kategori numerik, perhitungan dalam data modeling dapat dipermudah dan disederhanakan.
  
+ Fill Missing Values

  Proses ini akan mengisi nilai _missing value _dari dataset yang sudah ada. Hal ini dilakukan untuk mengantisipasi _overfit_ atau _underfit_ pada model yang akan dibangun. Dalam hal ini, label yang termasuk adalah _Engine Size_ (L), _Horsepower_, _Torque_ (lb-ft), 0-60 MPH _Time_ (_seconds_). Nilai akan di isi dengan nilai _mean_ dari data tersebut.

+ Convert Data Type

  Pada proses ini, akan dilakukan penyeragaman tipe data pada setiap kolom, sehingga seluruh kolom dapat terstandarisasi dengan baik. Tipe data yang digunakan adalah int64 pada setiap kolom.

+ Add Additional Variable

  Proses ini bertujuan untuk menambahkan variabel baru yang dapat memberikan _insight_ baru mengenai mobil yang sudah diproduksi. Variabel yang ditambahkan adalah _Age_, variabel ini merepresentasikan umur dari kendaraan sampai dengan saat ini.

+ Train Test Split

  _Train test split_ adalah proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 1007 dibagi menjadi komposisi 805 (80%) untuk data latih dan 202 (20%) untuk data uji. Komposisi pembagian ini digunakan karena jumlah dataset yang dimiliki cenderung sedikit, sehingga untuk mengoptimalkan pelatihan model, maka komposisi training yang digunakan adalah 80% dan testing 20%.
  

## Modeling

+ Algoritma
  Penelitian ini melakukan pemodelan dengan 2 algoritma, yaitu _Logistic Linear Regression_, dan _Decision Tree_.
  + Logistic Linear Regression

    Model _Logistic Regression_ adalah algoritma _machine learning_ yang digunakan untuk mengatasi masalah klasifikasi, terutama dalam konteks _binary_ atau _multiclass classification_. Cara kerjanya melibatkan inisialisasi bobot dan bias awal, pembuatan model dengan menghitung skor z, perhitungan transformasi skor dengan _sigmoid activation function_, perhitungan fungsi biaya berdasarkan probabilitas prediksi, dan pelatihan model dengan mengoptimalkan fungsi biaya melalui gradien turun. Proses ini diulangi hingga mencapai nilai konvergensi. Berikut adalah rumus untuk Z-Score dan fungsi sigmoid dalam LaTeX:

    **Z-Score Formula:**
    $$Z = \frac{X - \mu}{\sigma}$$
    
    Di mana:
    - $\(Z\)$ adalah Z-Score.
    - $\(X\)$ adalah nilai.
    - $\(\mu\)$ adalah rata-rata (mean).
    - $\(\sigma\)$ adalah deviasi standar (standard deviation).
    
    **Sigmoid Function Formula:**
    $$S(x) = \frac{1}{1 + e^{-x}}$$
    
    Di mana:
    - $\(S(x)\)$ adalah nilai fungsi sigmoid.
    - $\(x\)$ adalah nilai input.
    - $\(e\)$ adalah bilangan Euler (sekitar 2.71828).
    
    Selama pelatihan, model ini diperbarui untuk meminimalkan kesalahan antara probabilitas prediksi dan label sebenarnya dengan mengoptimalkan fungsi biaya. Kelebihan model ini termasuk interpretabilitas yang baik dan kinerja yang cukup baik pada masalah dengan hubungan linier antara fitur dan variabel target. Proyek ini menggunakan [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) dengan memasukkan X_train dan y_train dalam membangun model.

  + Decision Tree

    _Decision tree_ (pohon keputusan) adalah sebuah model prediktif dalam ilmu data dan pembelajaran mesin yang digunakan untuk mengambil keputusan berdasarkan aturan yang didefinisikan dalam bentuk struktur pohon. Model ini digunakan untuk masalah klasifikasi dan regresi, serta dapat digunakan untuk tugas pengambilan keputusan yang melibatkan berbagai variabel dan skenario. Algoritma _Decision Tree_ mengikuti serangkaian langkah untuk membangun model pemutusan keputusan. Langkah-langkah dari algoritma _Decision tree_ adalah sebagai berikut:
    1. Mulai dari simpul akar, misalkan sebagai S, yang berisi dataset lengkap.
    2. Ambil atribut terbaik dalam dataset menggunakan _Attribute Selection Measure_ (ASM). ASM yang bisa digunakan di antaranya _Information Gain_ dan Gini _Index_
    3. Pisahkan himpunan S menjadi himpunan bagian yang berisi kemungkinan nilai untuk atribut terbaik.
    4. Buat simpul _decision tree_ yang berisi atribut terbaik.
    5. Buat simpul _decision tree_ baru secara rekursif menggunakan himpunan bagian dari kumpulan data yang dibuat pada langkah 3. Lanjutkan proses ini sampai tahap terakhir di mana kita tidak dapat mengklasifikasikan simpul lebih lanjut. Simpul ini yang menjadi simpul akhir atau disebut sebagai simpul daun (_leaf node_).

    Dengan demikian, _Decision Tree_ membantu dalam memetakan kondisi atau atribut yang membimbing keputusan berdasarkan data yang ada, dan memungkinkan kita untuk melakukan klasifikasi atau prediksi. Dalam implementasinya digunakan [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) dalam code.

+ Hyperparameter Tuning (Grid Search)
 _ Hyperparameter tuning_ adalah cara untuk mendapatkan parameter terbaik dari algoritma dalam membangun model. Salah satu teknik dalam _hyperparameter tuning_ yang digunakan dalam proyek ini adalah _grid search_. 
  
  Dalam kasus ini, parameter _tuning_ dilakukan pada model yang menggunakan algoritma _decision tree_. Hal ini didasarkan pada hasil sebelumnya yang sudah didapatkan, _Decision Tree_ memiliki hasil r2_score yang lebih baik jika dibandingkan dengan _Logistic Linear Regression_.

  Pemilihan model _Decision Tree_ dengan akurasi 0.91 untuk melakukan _tuning_ parameter daripada model Logistic Regression dengan akurasi 0.72 bisa dijelaskan dengan beberapa faktor. Pertama, kinerja yang lebih baik dari _Decision Tree_ menunjukkan kemampuan model ini dalam mengklasifikasikan data dengan akurasi yang lebih tinggi, yang menjadi tujuan utama dalam _machine learning_. Selain itu, _Decision Tree_ mampu menangani hubungan yang lebih kompleks antara fitur-fitur dan variabel target, yang lebih sulit diakomodasi oleh model _Logistic Regression_ yang bergantung pada hubungan linier. Jika _tuning_ parameter pada _Decision Tree_ berhasil meningkatkan akurasi, ini menandakan bahwa model telah dioptimalkan dengan baik. Konteks masalah, interpretabilitas, dan tujuan akhir dalam analisis juga memainkan peran penting dalam pemilihan model. Akhirnya, pemilihan model selalu bergantung pada kombinasi dari faktor-faktor ini, dan dalam situasi ini, _Decision Tree_ terbukti menjadi pilihan yang lebih baik dalam mencapai kinerja yang diinginkan. Berikut adalah hasil dari _Grid Search_ pada proyek ini :

  Tabel 1. Hyper-Tuning Parameter pada Decision Tree
  | model    | best_params                                                     |
  |----------|-----------------------------------------------------------------|
  | Decision Tree| {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split':2}  |
 


## Evaluation

Metrik evaluasi yang digunakan pada proyek ini [R2 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html). $R^2$ (R-_squared_), juga dikenal sebagai koefisien determinasi, adalah metrik evaluasi yang digunakan dalam statistik dan analisis regresi untuk mengukur sejauh mana model regresi cocok dengan data yang diamati. $R^2 Score$ mengukur seberapa baik variabilitas dalam data independen (variabel prediktor) menjelaskan variabilitas dalam data dependen (variabel target). $R^2 Score$ berkisar antara 0 hingga 1, di mana 1 mengindikasikan bahwa model mampu menjelaskan semua variasi dalam data dengan sempurna, sementara 0 menunjukkan bahwa model tidak menjelaskan variasi apa pun dan hasilnya sama dengan prediksi rata-rata. 

Oleh karena itu, penggunaan $R^2 Score$ dalam  proyek ini adalah untuk mengukur sejauh mana model mampu menjelaskan variasi dalam harga mobil sport berdasarkan atribut-atribut yang ada. Dengan cara membandingkan kinerja model dengan prediksi rata-rata, R2 Score memberikan gambaran tentang tingkat akurasi prediksi, yang penting dalam menentukan prediksi harga mobil. Berikut formula $R^2 Score$ :

<br>
<div><img src="https://github.com/mizanulridhoaohana/machine-learning-terapan/assets/112617513/06c7a544-80f8-4d61-8746-3076900587a5" width="600" align="center"/></div>
<br>


Dalam rumus ini:
- $\( R^2 \)$ adalah $R^2 Score$.
- $SSRES$ adalah jumlah kuadrat dari sisa kesalahan.
- $SSTOT$ adalah jumlah total kesalahan.
- $\( y_i \)$ adalah nilai sebenarnya dari data.
- $\( \hat{y}_i \)$ adalah nilai yang diprediksi oleh model.
- $\( \bar{y} \)$ adalah rata-rata dari nilai sebenarnya.

Contoh Interpretasi perhitungan $R^2 Score$ :
Asumsikan $R^2$ = 0,68
Dapat dikatakan bahwa 68% variabilitas atribut keluaran dependen dapat dijelaskan oleh model, sedangkan 32% sisanya masih belum dapat dijelaskan.
$R^2$ menunjukkan proporsi titik data yang terletak di dalam garis yang dibuat oleh persamaan regresi. Nilai $R^2$ yang lebih tinggi diinginkan karena menunjukkan hasil yang lebih baik.

### Hasil Evaluasi Proyek

Berikut hasil evaluasi pada proyek ini :

+ $R^2$ Score
 
  Tabel 2. Evaluasi $R^2 Score$ pada proyek
  | model                       | r2_score |
  |-----------------------------|----------|
  | Logistic Linear Regression  | 0.720902 |
  | Decision Tree               | 0.918839 |
  | Decision Tree (Hyper-tuning)| 0.974387 |


Dari hasil evaluasi yang diperoleh dapat disimpulkan bahwa algoritma terbaik untuk memprediksi permasalahan ini adalah _Decision Tree_ dengan _Hyper-Tuning Parameter_. Nilai R2 Score yang didapatkan adalah 0.97438 atau bisa dibilang mendekati nilai maksimum 1. R2 Score (_Coefficient of Determination_) adalah ukuran statistik yang mengindikasikan sejauh mana model statistik memprediksi variabilitas data. Nilai R2 Score berkisar antara 0 hingga 1, di mana 1 menunjukkan bahwa model mampu menjelaskan seluruh variasi dalam data dengan sempurna, sementara 0 menunjukkan bahwa model sama buruknya dengan menggunakan nilai rata-rata sebagai prediksi.

Dalam konteks ini, mendekati 1 adalah hal yang baik karena model Decision Tree yang telah di-_tune_ dengan baik mampu menjelaskan sebagian besar variasi dalam harga mobil _sport_ berdasarkan fitur-fitur yang digunakan (_Car Make, Car Model, Year, Engine Size, Horsepower, Torque, 0-60 MPH Time_). Ini menunjukkan bahwa model mampu memberikan prediksi yang sangat baik dan akurat dalam menjelaskan bagaimana berbagai faktor-fitur ini memengaruhi harga mobil _sport_. Dengan kata lain, sekitar 97.44% variasi dalam harga mobil sport dapat dijelaskan oleh model ini.

Dengan hasil ini, kita dapat memiliki tingkat keyakinan yang tinggi dalam kemampuan model untuk melakukan prediksi harga mobil _sport_ berdasarkan atribut-atribut yang diberikan. Hal ini dapat berguna dalam analisis pasar, penetapan harga yang lebih akurat, atau dalam mengidentifikasi faktor-faktor kunci yang memengaruhi harga mobil sport.

### Implikasi Bisnis

Hasil evaluasi dengan $R^2 Score$ yang mendekati 1 memiliki implikasi yang signifikan dalam konteks pengambilan keputusan bisnis. Dalam kasus ini, di mana model _Decision Tree_ yang telah di-_tune_ memiliki $R^2 Score$ sebesar 0.97438, ini berarti bahwa model tersebut mampu menjelaskan sebagian besar variasi dalam harga mobil sport berdasarkan fitur-fitur yang diberikan. 

Implikasi ini dapat berdampak pada sejumlah keputusan bisnis:

1. **Penetapan Harga yang Lebih Akurat**: Model dengan $R^2 Score$ tinggi memungkinkan produsen mobil _sport_ untuk menetapkan harga yang lebih akurat berdasarkan fitur-fitur kendaraan. Dengan pemahaman yang lebih baik tentang bagaimana faktor-faktor seperti merek, model, tahun, ukuran mesin, tenaga kuda, torsi, dan waktu akselerasi memengaruhi harga, produsen dapat menyesuaikan harga dengan lebih baik sesuai dengan karakteristik mobil _sport_ yang ditawarkan.

2. **Optimisasi Portofolio Produk**: Informasi yang dihasilkan dari model dapat membantu produsen dalam mengoptimalkan portofolio produk mereka. Mereka dapat lebih baik dalam mengidentifikasi fitur-fitur yang paling penting bagi konsumen dan memutuskan jenis mobil _sport_ apa yang harus diproduksi dan dipasarkan.

3. **Kustomisasi Harga**: Model yang akurat memungkinkan produsen untuk menyesuaikan harga mobil _sport_ berdasarkan spesifikasi unik setiap model. Ini memungkinkan harga yang lebih kustomisasi sesuai dengan fitur-fitur tambahan, memberikan fleksibilitas yang lebih besar dalam penetapan harga.

4. **Penentuan Rencana Pemasaran**: Dengan pemahaman yang lebih mendalam tentang bagaimana karakteristik mobil _sport_ memengaruhi harga, produsen dapat mengembangkan strategi pemasaran yang lebih efektif, seperti menyoroti fitur-fitur utama yang mempengaruhi harga.

5. **Analisis Kinerja Bisnis**: Hasil ini juga dapat membantu dalam mengevaluasi kinerja bisnis, seperti memahami hubungan antara berbagai atribut dan performa penjualan. Ini memungkinkan produsen untuk membuat langkah-langkah yang lebih baik dalam merespons perubahan pasar.

Dengan kata lain, _R2 Score_ yang tinggi memberikan landasan yang kuat untuk mengambil keputusan bisnis yang lebih tepat dan berorientasi data. Dalam konteks industri mobil _sport_ yang sangat kompetitif, pemahaman yang kuat tentang faktor-faktor yang memengaruhi harga adalah aset berharga dalam merencanakan strategi dan mengoptimalkan operasi bisnis.
