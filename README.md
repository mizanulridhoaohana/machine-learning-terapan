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

    Model Logistic Regression adalah algoritma machine learning yang digunakan untuk mengatasi masalah klasifikasi, terutama dalam konteks binary atau multiclass classification. Cara kerjanya melibatkan inisialisasi bobot dan bias awal, pembuatan model dengan menghitung skor z, perhitungan transformasi skor dengan sigmoid activation function, perhitungan fungsi biaya berdasarkan probabilitas prediksi, dan pelatihan model dengan mengoptimalkan fungsi biaya melalui gradien turun. Proses ini diulangi hingga mencapai nilai konvergensi. Berikut adalah rumus untuk Z-Score dan fungsi sigmoid dalam LaTeX:

    **Z-Score Formula:**
    $$Z = \frac{X - \mu}{\sigma}$$
    
    Di mana:
    - \(Z\) adalah Z-Score.
    - \(X\) adalah nilai.
    - \(\mu\) adalah rata-rata (mean).
    - \(\sigma\) adalah deviasi standar (standard deviation).
    
    **Sigmoid Function Formula:**
    $$S(x) = \frac{1}{1 + e^{-x}}$$
    
    Di mana:
    - \(S(x)\) adalah nilai fungsi sigmoid.
    - \(x\) adalah nilai input.
    - \(e\) adalah bilangan Euler (sekitar 2.71828).
    
    Selama pelatihan, model ini diperbarui untuk meminimalkan kesalahan antara probabilitas prediksi dan label sebenarnya dengan mengoptimalkan fungsi biaya. Kelebihan model ini termasuk interpretabilitas yang baik dan kinerja yang cukup baik pada masalah dengan hubungan linier antara fitur dan variabel target. . Proyek ini menggunakan [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) dengan memasukkan X_train dan y_train dalam membangun model.

  + Decision Tree

    Decision tree (pohon keputusan) adalah sebuah model prediktif dalam ilmu data dan pembelajaran mesin yang digunakan untuk mengambil keputusan berdasarkan aturan yang didefinisikan dalam bentuk struktur pohon. Model ini digunakan untuk masalah klasifikasi dan regresi, serta dapat digunakan untuk tugas pengambilan keputusan yang melibatkan berbagai variabel dan skenario. Algoritma Decision Tree mengikuti serangkaian langkah untuk membangun model pemutusan keputusan. Langkah-langkah dari algoritma Decision tree adalah sebagai berikut:
    1. Mulai dari simpul akar, misalkan sebagai S, yang berisi dataset lengkap.
    2. Ambil atribut terbaik dalam dataset menggunakan Attribute Selection Measure (ASM). ASM yang bisa digunakan di antaranya Information Gain dan Gini Index
    3. Pisahkan himpunan S menjadi himpunan bagian yang berisi kemungkinan nilai untuk atribut terbaik.
    4. Buat simpul decision tree yang berisi atribut terbaik.
    5. Buat simpul decision tree baru secara rekursif menggunakan himpunan bagian dari kumpulan data yang dibuat pada langkah 3. Lanjutkan proses ini sampai tahap terakhir di mana kita tidak dapat mengklasifikasikan simpul lebih lanjut. Simpul ini yang menjadi simpul akhir atau disebut sebagai simpul daun (leaf node).

    Dengan demikian, Decision Tree membantu dalam memetakan kondisi atau atribut yang membimbing keputusan berdasarkan data yang ada, dan memungkinkan kita untuk melakukan klasifikasi atau prediksi. Dalam implementasinya digunakan [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) dalam code.


+ Hyperparameter Tuning (Grid Search)
  Hyperparameter tuning adalah cara untuk mendapatkan parameter terbaik dari algoritma dalam membangun model. Salah satu teknik dalam hyperparameter tuning yang digunakan dalam proyek ini adalah grid search. 
  
  Dalam kasus ini, parameter tuning dilakukan pada model yang menggunakan algoritma decision tree. Hal ini didasarkan pada hasil sebelumnya yang sudah didapatkan, Decision Tree memiliki hasil r2_score yang lebih baik jika dibandingkan dengan Logistic Linear Regression.

  Pemilihan model Decision Tree dengan akurasi 0.91 untuk melakukan tuning parameter daripada model Logistic Regression dengan akurasi 0.72 bisa dijelaskan dengan beberapa faktor. Pertama, kinerja yang lebih baik dari Decision Tree menunjukkan kemampuan model ini dalam mengklasifikasikan data dengan akurasi yang lebih tinggi, yang menjadi tujuan utama dalam machine learning. Selain itu, Decision Tree mampu menangani hubungan yang lebih kompleks antara fitur-fitur dan variabel target, yang lebih sulit diakomodasi oleh model Logistic Regression yang bergantung pada hubungan linier. Jika tuning parameter pada Decision Tree berhasil meningkatkan akurasi, ini menandakan bahwa model telah dioptimalkan dengan baik. Konteks masalah, interpretabilitas, dan tujuan akhir dalam analisis juga memainkan peran penting dalam pemilihan model. Akhirnya, pemilihan model selalu bergantung pada kombinasi dari faktor-faktor ini, dan dalam situasi ini, Decision Tree terbukti menjadi pilihan yang lebih baik dalam mencapai kinerja yang diinginkan. Berikut adalah hasil dari Grid Search pada proyek ini :
  | model    | best_params                                                     |
  |----------|-----------------------------------------------------------------|
  | Decision | {'max_depth': None, 'max_features': 'sqrt',                     |
  | Tree     |  'min_samples_leaf': 1, 'min_samples_split':2}                  |


## Evaluation

Metrik evaluasi yang digunakan pada proyek ini [R2 Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html). R^2 (R-squared), juga dikenal sebagai koefisien determinasi, adalah metrik evaluasi yang digunakan dalam statistik dan analisis regresi untuk mengukur sejauh mana model regresi cocok dengan data yang diamati. R^2 score mengukur seberapa baik variabilitas dalam data independen (variabel prediktor) menjelaskan variabilitas dalam data dependen (variabel target). R^2 score berkisar antara 0 hingga 1, di mana 1 mengindikasikan bahwa model mampu menjelaskan semua variasi dalam data dengan sempurna, sementara 0 menunjukkan bahwa model tidak menjelaskan variasi apa pun dan hasilnya sama dengan prediksi rata-rata. Berikut formula R2 Score :

$$R^2 = \frac{\small {\sum_{n=0}^{n}} {x^2}}{\small \sum_{n=0}^{n}}$$


Dalam rumus ini:
- \( R^2 \) adalah R2 Square.
- \( y_i \) adalah nilai sebenarnya dari data.
- \( \hat{y}_i \) adalah nilai yang diprediksi oleh model.
- \( \bar{y} \) adalah rata-rata dari nilai sebenarnya.

Anda dapat menggunakan rumus LaTeX ini dalam dokumen LaTeX Anda untuk menampilkan formula R2 Square.

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


Dari hasil evaluasi yang diperoleh dapat disimpulkan bahwa algoritma terbaik untuk memprediksi permasalahan ini adalah Decision Tree dengan Hyper-Tuning Parameter. Nilai R2 Score yang didapatkan adalah 0.97438 atau bisa dibilang mendekati nilai maksimum 1. R2 Score (Coefficient of Determination) adalah ukuran statistik yang mengindikasikan sejauh mana model statistik memprediksi variabilitas data. Nilai R2 Score berkisar antara 0 hingga 1, di mana 1 menunjukkan bahwa model mampu menjelaskan seluruh variasi dalam data dengan sempurna, sementara 0 menunjukkan bahwa model sama buruknya dengan menggunakan nilai rata-rata sebagai prediksi.

Dalam konteks ini, mendekati 1 adalah hal yang baik karena model Decision Tree yang telah di-tune dengan baik mampu menjelaskan sebagian besar variasi dalam harga mobil sport berdasarkan fitur-fitur yang digunakan (Car Make, Car Model, Year, Engine Size, Horsepower, Torque, 0-60 MPH Time). Ini menunjukkan bahwa model mampu memberikan prediksi yang sangat baik dan akurat dalam menjelaskan bagaimana berbagai faktor-fitur ini memengaruhi harga mobil sport. Dengan kata lain, sekitar 97.44% variasi dalam harga mobil sport dapat dijelaskan oleh model ini.

Dengan hasil ini, kita dapat memiliki tingkat keyakinan yang tinggi dalam kemampuan model untuk melakukan prediksi harga mobil sport berdasarkan atribut-atribut yang diberikan. Hal ini dapat berguna dalam analisis pasar, penetapan harga yang lebih akurat, atau dalam mengidentifikasi faktor-faktor kunci yang memengaruhi harga mobil sport.

### Implikasi Bisnis

Hasil evaluasi dengan R2 Score yang mendekati 1 memiliki implikasi yang signifikan dalam konteks pengambilan keputusan bisnis. Dalam kasus ini, di mana model Decision Tree yang telah di-tune memiliki R2 Score sebesar 0.97438, ini berarti bahwa model tersebut mampu menjelaskan sebagian besar variasi dalam harga mobil sport berdasarkan fitur-fitur yang diberikan. 

Implikasi ini dapat berdampak pada sejumlah keputusan bisnis:

1. **Penetapan Harga yang Lebih Akurat**: Model dengan R2 Score tinggi memungkinkan produsen mobil sport untuk menetapkan harga yang lebih akurat berdasarkan fitur-fitur kendaraan. Dengan pemahaman yang lebih baik tentang bagaimana faktor-faktor seperti merek, model, tahun, ukuran mesin, tenaga kuda, torsi, dan waktu akselerasi memengaruhi harga, produsen dapat menyesuaikan harga dengan lebih baik sesuai dengan karakteristik mobil sport yang ditawarkan.

2. **Optimisasi Portofolio Produk**: Informasi yang dihasilkan dari model dapat membantu produsen dalam mengoptimalkan portofolio produk mereka. Mereka dapat lebih baik dalam mengidentifikasi fitur-fitur yang paling penting bagi konsumen dan memutuskan jenis mobil sport apa yang harus diproduksi dan dipasarkan.

3. **Kustomisasi Harga**: Model yang akurat memungkinkan produsen untuk menyesuaikan harga mobil sport berdasarkan spesifikasi unik setiap model. Ini memungkinkan harga yang lebih kustomisasi sesuai dengan fitur-fitur tambahan, memberikan fleksibilitas yang lebih besar dalam penetapan harga.

4. **Penentuan Rencana Pemasaran**: Dengan pemahaman yang lebih mendalam tentang bagaimana karakteristik mobil sport memengaruhi harga, produsen dapat mengembangkan strategi pemasaran yang lebih efektif, seperti menyoroti fitur-fitur utama yang mempengaruhi harga.

5. **Analisis Kinerja Bisnis**: Hasil ini juga dapat membantu dalam mengevaluasi kinerja bisnis, seperti memahami hubungan antara berbagai atribut dan performa penjualan. Ini memungkinkan produsen untuk membuat langkah-langkah yang lebih baik dalam merespons perubahan pasar.

Dengan kata lain, R2 Score yang tinggi memberikan landasan yang kuat untuk mengambil keputusan bisnis yang lebih tepat dan berorientasi data. Dalam konteks industri mobil sport yang sangat kompetitif, pemahaman yang kuat tentang faktor-faktor yang memengaruhi harga adalah aset berharga dalam merencanakan strategi dan mengoptimalkan operasi bisnis.
