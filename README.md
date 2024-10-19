# Laporan Proyek Machine Learning - Muhammad Alivian Sidiq

## Domain Proyek

WHO (World Health Organization) melaporkan bahwa sekitar 1,6 juta orang meninggal setiap tahun akibat diabetes [[1]](https://doi.org/10.1016/j.icte.2021.02.004) Diabetes adalah penyakit yang memengaruhi kemampuan tubuh dalam memproduksi hormon insulin, yang pada gilirannya membuat metabolisme karbohidrat menjadi tidak normal dan meningkatkan kadar glukosa dalam darah [[2]](https://doi.org/10.1016/j.procs.2018.05.122).

Penelitian tentang pasien diabetes menunjukkan bahwa diabetes di kalangan orang dewasa (di atas 18 tahun) telah meningkat dari 4,7% menjadi 8,5% pada tahun 1980 hingga 2014 dan berkembang pesat di negara-negara dunia kedua dan ketiga. Hasil statistik pada tahun 2017 menunjukkan bahwa 451 juta orang hidup dengan diabetes di seluruh dunia, yang akan meningkat menjadi 693 juta pada tahun 2045 [[3]](https://doi.org/10.1109/ACCESS.2020.2989857).

Prediksi dini penyakit diabetes diperlukan untuk dapat dikendalikan dan menyelamatkan nyawa manusia [[4]](https://doi.org/10.32520/stmsi.v10i1.1129) . Karena gejalanya yang mirip dengan kondisi sakit biasa, banyak orang yang tidak menyadari bahwa
mereka mengidap penyakit diabetes dan bahkan sudah mengarah pada komplikasi [[5]](https://www.ijert.org/research/diabetes-prediction-using-machine-learning-techniques-IJERTV9IS090496.pdf)

**Mengapa Permasalahan harus diselesaikan ?**
- Penyakit Diabetes menjadi salah satu penyebab utama kematian di dunia.
- Komplikasi diabetes dapat menyebabkan penyakit lain seperti penyakit jantung, stroke, kerusakan ginjal, dan kebutaan dapat mengancam nyawa.
- Pengobatan diabetes membutuhkan biaya yang tinggi, baik untuk perawatan jangka pendek maupun jangka panjang
    
**Referensi Terkait**

[[1]](https://doi.org/10.1016/j.icte.2021.02.004) J. J. Khanam and S. Y. Foo, “A comparison of machine learning algorithms for diabetes prediction,” ICT Express, vol. 7, no. 4, pp. 432–439, Dec. 2021, doi: 10.1016/j.icte.2021.02.004.

[[2]](https://doi.org/10.1016/j.procs.2018.05.122)	D. Sisodia and D. S. Sisodia, “Prediction of Diabetes using Classification Algorithms,” Procedia Comput Sci, vol. 132, pp. 1578–1585, 2018, doi: 10.1016/j.procs.2018.05.122.

[[3]](https://doi.org/10.1109/ACCESS.2020.2989857)	Md. K. Hasan, Md. A. Alam, D. Das, E. Hossain, and M. Hasan, “Diabetes Prediction Using Ensembling of Different Machine Learning Classifiers,” IEEE Access, vol. 8, pp. 76516–76531, 2020, doi: 10.1109/ACCESS.2020.2989857.

[[4]](https://doi.org/10.32520/stmsi.v10i1.1129)	W. Apriliah, I. Kurniawan, M. Baydhowi, and T. Haryati, “Prediksi Kemungkinan Diabetes pada Tahap Awal Menggunakan Algoritma Klasifikasi Random Forest,” SISTEMASI, vol. 10, no. 1, p. 163, Jan. 2021, doi: 10.32520/stmsi.v10i1.1129.

[[5]](https://www.ijert.org/research/diabetes-prediction-using-machine-learning-techniques-IJERTV9IS090496.pdf)    M.Soni, S Varma "Diabetes Prediction using Machine Learning
Techniques," IJERT, Vol. 9 No 09 p. 921-925 Sep. 2020


## Business Understanding

### Problem Statements

- Bagaimana cara mengidentifikasi pasien yang beresiko tinggi mengalami diabetes berdasarkan fitur-fitur seperti BMI,Insulin,Glukosa,Umur,dll ?
- Model Machine Learning apa yang dapat memprediksi penyakit diabetes dengan performa yang tinggi ? 

### Goals

- Membangun model machine learning yang dapat mengidentifikasi pasien yang beresiko tinggi mengalami diabetes
- Meningkatkan performa model prediksi melalui hyperparameter tuning dan beberapa teknik machine learning.

### Solution statements
- Meenggunakan beberapa algoritma untuk klasifikasi yaitu Logistic Regression, Support Vector Machine, Random Forest, Gradient Boosting.
- Melakukan hyperparameter tuning untuk menemukan parameter terbaik setiap algoritma
- Membandingkan performa model dan memilih model terbaik berdasarkan Accuracy, Precision, Recall, dan F1 Score

## Data Understanding
Dataset ini berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Tujuan dari kumpulan data ini adalah untuk memprediksi secara diagnostik apakah pasien menderita diabetes atau tidak, berdasarkan pengukuran diagnostik tertentu yang disertakan dalam kumpulan data. Secara khusus, semua pasien pada data adalah perempuan berusia minimal 21 tahun yang berasal dari suku Indian Pima

Sumber data berasal dari: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)


### Variabel-variabel pada Pima Indians Diabetes Database adalah sebagai berikut:
## Deskripsi Variabel
Berdasarkan informasi dari sumber dataset berikut adalah penjelasan untuk masing-masing kolom :

| No | Column Name |	Meaning |
|----|-------------|----------|
|1|Pregnancies|	Jumlah Kehamilan yang dialami
|2|	Glucose| Konsentrasi glukosa plasma yang diukur dalam tes toleransi glukosa oral selama 2 jam
|3|	BloodPressure|	 Pengukuran tekanan darah diastolik (mm Hg).
|4|	SkinThickness|	Ketebalan lipatan kulit trisep (mm)
|5|	Insulin|	 Serum insulin 2-jam (mu U/ml)
|6|	BMI	| Indeks massa tubuh, yang dihitung sebagai berat badan dalam kg / (tinggi badan dalam meter) ^ 2.
|7|	DiabetesPedigreeFunction|	Fungsi yang menilai kemungkinan diabetes berdasarkan riwayat keluarg
|8|	Age|	Usia Usia individu dalam tahun
|9|	Outcome|	Outcome adalah variabel target, yang mengindikasikan apakah individu mengidap diabetes (1) atau tidak (0)

## Exploratory Data Analysis
**Informasi Dataset**

Setelah memahami deskripsi variabel pada data, langkah selanjutnya adalah mengecek informasi pada dataset dengan fungsi info() berikut.

![{0BA7C2FA-4A90-4EBC-829F-0213EFA64185}](https://github.com/user-attachments/assets/1c741cf0-f029-4e63-b9ee-f45a349ab4c0)

- Terdapat 2 kolom dengan tipe float64, yaitu: BMI dan DiabetesPedigreeFunction 
- Terdapat 6 kolom numerik dengan tipe data int64 yaitu: Pregnancies, Glucose, BloodPressure, SkinTickness, Insulin, Age, Outcome (ini merupakan terget fitur kita) .

**Statistik Deskripsi**

Uraian di atas menunjukkan bahwa setiap kolom telah memiliki tipe data yang sesuai. Selanjutnya, Anda perlu mengecek deskripsi statistik data dengan fitur describe()

![{426F8B34-A4D6-4E44-8EB8-E80A515351FC}](https://github.com/user-attachments/assets/ae4d61a6-fc8b-42bd-bde9-906eea25b464)

Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:

- Count  adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom. 
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama. 
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

**Distribusi Outcome**

Visualisasi ini digunakan untuk menunjukkan distribusi target pada variabel outcome dalam bentuk bar. grafik ini memberikan gambaran yang jelas tentang proporsi outcome dalam dataset.

![{D24F32D6-8497-4832-A297-FFF3309F95A6}](https://github.com/user-attachments/assets/b2f57a28-d469-4fbf-8a6f-109742d66e4e)

- Distribusi tidak seiimbang di antara 2 label.
- No Diabetes(0) 500 data , Diabetes(1) 268 data.

**Correlation Analysis**

Membuat HeatMap untuk menampilkan korelasi matrik antar kolom pada dataset

  ![{53856F2B-B77D-491B-A51D-6C6F94ABBFD9}](https://github.com/user-attachments/assets/2389b1d2-1051-447f-966f-3c239f82226d)

Berdasarkan HeatMap korelasi yang dilakukan menunjukkan bahwa kolom outcome memiliki korelasi paling tinggi dengan kolom glukosa dengan skor korelasi sebesar 0,47. Artinya terdapat hubungan yang cukup kuat antara kadar glukosa dengan outcome diabetes, yang menunjukkan bahwa semakin tinggi kadar glukosa maka semakin besar kemungkinan seseorang menderita diabetes.

Sebaliknya, kolom outcome memiliki korelasi paling rendah dengan kolom skinthickness dan BloodPleassure dengan skor korelasi sebesar 0,07. Hal ini menunjukkan bahwa hubungan antara skin thickness dan Blood pleassure dengan outcome diabetes sangat lemah, sehingga skin thickness dan blood pleasure bukan merupakan indikator yang signifikan dalam memprediksi diabetes.

**Box Plot**

Boxplot digunakan untuk mendeteksi outliers pada fitur yaitu data yang nilainya 1.5 QR di atas Q3 atau 1.5 QR di bawah Q1.

![{2EE97AC3-CAC4-4845-A020-740CEAE7925A}](https://github.com/user-attachments/assets/8fec5e83-3c1a-442f-a457-be96b740e5c9)

Pada grafik diatas ada titik-titik yang menandakan bahwa terdapat fitur yang mengandung outlier itu akan diatasi pada tahap data preparation

## Data Preparation

**Teknik Data Preparation**
- Handling Missing Values: Mengimputasi nilai yang hilang pada dataset
- Removing Outliers: Menghapus data yang memiliki nilai outliers pada kolom tertentu.
- Feature Engginering : Merubah Fitur numerik menjadi Kategorikal
- Encoding Categorical Variables: Mengubah variabel kategorikal menjadi variabel numerik menggunakan teknik one-hot encoding.
- Pisahkan dataset menjadi fitur dan target
- Split Dataset untuk training dan testing dengan presentase 80:20
- Feature Scaling: Melakukan standarisasi pada fitur numerik untuk memastikan semua fitur berada dalam skala yang sama.
- Oversampling untuk menangani data imbalance

**Proses Data Preparation**: 
- Fitur yang memiliki jumlah missing value akan dilakukan imputasi dengan nilai median dari kelas yang sama Artinya, nilai hilang pada suatu fitur akan diisi dengan median dari fitur yang sama, namun hanya untuk data yang termasuk dalam kelas yang sama. Ini berguna jika terdapat perbedaan distribusi nilai antara kelas-kelas yang berbeda.
- Outlier diatasi menggunakan LOF (Local Outlier Factor) LOF adalah algoritma yang mengidentifikasi outlier berdasarkan densitas data di sekitar titik data tersebut. Titik data yang memiliki densitas jauh lebih rendah dibandingkan tetangganya dianggap sebagai outlier. LOF membandingkan densitas lokal suatu titik data dengan densitas rata-rata dari tetangganya. Semakin besar nilai LOF, semakin tinggi kemungkinan titik data tersebut adalah outlier.
- Feature Engginering Merubah kolom BMI, Insulin, Glukosa menjadi kategorikal berdasarkan nilai ambang tertentu sebagai Berikut.
  
  BMI
  ```python
  df.loc[df["BMI"]<18.5, "NewBMI"] = NewBMI[0]
  df.loc[(df["BMI"]>18.5) & df["BMI"]<=24.9, "NewBMI"] = NewBMI[1]
  df.loc[(df["BMI"]>24.9) & df["BMI"]<=29.9, "NewBMI"] = NewBMI[2]
  df.loc[(df["BMI"]>29.9) & df["BMI"]<=34.9, "NewBMI"] = NewBMI[3]
  df.loc[(df["BMI"]>34.9) & df["BMI"]<=39.9, "NewBMI"] = NewBMI[4]
  df.loc[df["BMI"]>39.9, "NewBMI"] = NewBMI[5]
  ```
  Insulin
  ```python
  if row["Insulin"]>=16 and row["Insulin"]<=166:
        return "Normal"
    else:
        return "Abnormal"
  ```
  Glokosa
  ```python
  df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]
  df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]
  df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]
  df.loc[df["Glucose"] > 126 ,"NewGlucose"] = NewGlucose[3]
  ```
- Kolom yang telah dilakukan Feature Engginering dilakukan encoding mengunakan one-hot-encoding. Setiap kategori akan diwakili oleh satu kolom baru. Jika suatu sampel termasuk dalam kategori tertentu, maka nilai pada kolom yang sesuai akan menjadi 1, sedangkan kolom lainnya akan bernilai 0.
- Pisahkan dataset menjadi fitur dan target sebagai berikut :
  ```python
  y=df['Outcome']
  X=df.drop(['Outcome'], axis=1)
  ```
- Split Dataset dengan perbandingan 80:20
- Feature scaling menggunakan MinMaxScaler yang merubah data menjadi rentang 0 sampai 1 dengan rumus sebagai berikut
  
    $$ x' = \frac{X - X_{min}}{X_{max} - X_{min}} $$
- Penanganan Kelas yang tidak seimbang dengan menggunakan SMOTE. Synthetic Minority Over-sampling Technique (SMOTE) adalah teknik oversampling yang menghasilkan data sintetis untuk kelas minoritas dengan cara menginterpolasi data yang ada. SMOTE memilih sampel acak dari kelas minoritas, kemudian mencari k-nearest neighbor terdekat. Selanjutnya, SMOTE akan membuat data sintetis baru di sepanjang garis yang menghubungkan sampel asli dengan k-nearest neighbor-nya.


**Alasan Dilakukan Data Preparation**
- Mengatasi missing values dengan Imputasi menggunakan median dipilih karena lebih robust terhadap outlier dibandingkan rata-rata. Dalam kasus data yang memiliki distribusi tidak normal atau terdapat outlier, median akan memberikan representasi yang lebih baik dari pusat data.
- Removing outliers untuk meningkatkan akurasi model dengan menghilangkan data yang dapat mempengaruhi performa model. LOF dipilih karena sangat berguna untuk mendeteksi outlier yang tidak mengikuti pola distribusi normal.
- Fature Engginering / Mengubah variabel numerik menjadi kategorikal sering dilakukan untuk Menangkap pola non-linear dalam data, Mengurangi kompleksitas model, Meningkatkan interpretabilitas model.
- Setelah variabel diubah menjadi kategorikal, kita perlu mengubahnya menjadi representasi numerik yang dapat dipahami oleh algoritma machine learning. One-Hot Encoding adalah teknik yang paling umum digunakan untuk mengubah variabel kategorikal menjadi vektor biner.
- Pisahkan dataset menjadi fitur dan target sebelum melakukan split training dan testing
- Perbandingan 80:20 adalah perbandingan yang umum digunakan untuk membagi dataset menjadi data latih (80%) dan data uji (20%). Data latih digunakan untuk melatih model, sedangkan data uji digunakan untuk mengevaluasi kinerja model.
- MinMaxScaler akan mentransformasi nilai fitur ke dalam rentang 0 sampai 1. Hal ini berguna untuk Mempercepat proses pelatihan model dan Menghindari dominasi fitur dengan skala yang sangat berbeda.
- Oversampling dilakukan karena jumlah data pada satu kelas jauh lebih banyak dibandingkan kelas lainnya, model cenderung bias ke kelas mayoritas dalam memprediksi.

## Modeling
Pada proyek ini, beberapa model supervised learning diterapkan untuk tugas klasifikasi. Model-model tersebut meliputi:
- Logistic Regression
- Support Vectore Machine
- Random Forest
- Gradient Boosting

**Kekurangan dan Kelebihan Algoritma**
- Logistic Regression
  
  Kelebihan: Algoritma ini mudah diimplementasikan, sederhana, dan cepat untuk digunakan dalam kasus klasifikasi biner. Hasilnya mudah diinterpretasikan karena menggunakan model linier yang memprediksi probabilitas, sehingga ideal untuk analisis di mana interpretasi penting.
  
  Kekurangan: Logistic Regression kurang cocok untuk masalah yang kompleks atau tidak linier, serta kinerjanya menurun jika ada korelasi non-linier antara fitur dan target.

- Support Vector Machine (SVM)
  
  Kelebihan: SVM sangat baik dalam menangani masalah klasifikasi dengan margin yang jelas antara dua kelas. Dengan kernel trick, SVM mampu menangani data yang tidak linier. Selain itu, SVM tahan terhadap overfitting, terutama dalam masalah dengan jumlah fitur yang tinggi.
  
  Kekurangan: SVM kurang efisien pada dataset besar karena kompleksitas waktu komputasi yang tinggi. Pemilihan kernel dan tuning hyperparameter yang tepat juga memerlukan waktu dan eksperimen yang cukup.

- Random Forest
  
  Kelebihan: Random Forest adalah model ensemble yang tangguh terhadap overfitting, karena dibangun dari kombinasi banyak decision trees. Algoritma ini mampu menangani data dengan banyak fitur dan bekerja baik dengan data yang memiliki hubungan non-linier.

  Kekurangan: Interpretasi dari Random Forest lebih sulit dibandingkan model linier, dan model ini cenderung lambat dalam membuat prediksi pada dataset besar. Selain itu, ukuran model bisa menjadi sangat besar karena banyaknya pohon keputusan yang dibuat.

- Gradient Boosting

  Kelebihan: Gradient Boosting sangat akurat karena membangun model secara bertahap dengan mengoreksi kesalahan dari model sebelumnya, dan sangat baik untuk menangani data dengan pola non-linier. Algoritma ini fleksibel dan dapat digunakan untuk klasifikasi maupun regresi.

  Kekurangan: Model ini rentan terhadap overfitting jika tidak dilakukan regularisasi yang tepat, dan proses pelatihannya bisa sangat lambat serta memerlukan banyak waktu komputasi, terutama untuk dataset yang besar.
**HyperParameter Tunning**
GridSearchCV Dipakai untuk menguji berbagai kombinasi hyperparameter pada setiap model. Proses ini memungkinkan pencarian hyperparameter optimal yang memberikan hasil terbaik berdasarkan metrik evaluasi yang dipilih berikut merupakan detail dari setiap algoritma:

1. Logistic Regression

  Parameter yang digunakan:
  ```python
  param_grid = {
      'penalty': ['l1', 'l2', 'elasticnet'],
      'C': [0.01, 0.1, 1, 10, 100],
      'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
  }
  ```
- penalty: Menentukan jenis regularisasi yang digunakan untuk mencegah overfitting.
  - l1 untuk Lasso (menghasilkan sparsity),
  - l2 untuk Ridge (mempertahankan semua fitur tetapi memperkecil koefisien),
  - elasticnet gabungan antara L1 dan L2.
- C: Menentukan kekuatan regularisasi, dengan nilai lebih kecil berarti regularisasi yang lebih kuat. Nilai C yang berbeda diuji untuk menemukan keseimbangan terbaik antara bias dan varians.
- solver: Algoritma optimisasi yang digunakan untuk menemukan koefisien terbaik. Solver yang berbeda bekerja lebih baik untuk kasus atau dataset tertentu.
  - newton-cg, lbfgs, dan sag cocok untuk regularisasi l2,
  - liblinear dan saga bisa digunakan untuk l1 atau elasticnet.

Pemilihan parameter ini mencakup opsi regularisasi yang membantu model menghindari overfitting pada data pelatihan, serta pengujian beberapa solver agar algoritma optimisasi bisa bekerja lebih efisien dengan dataset yang digunakan. Dan parameter terbaik yaitu C=10, max_iter=3000, random_state=42, solver='newton-cg' dengan akurasi 0.8618421052631579

2. Support Vector Machine (SVM)
   
   Parameter yang digunakan:
   ```python
   parameter = {
    "gamma":[0.0001, 0.001, 0.01, 0.1],
    'C': [0.01, 0.05,0.5, 0.01, 1, 10, 15, 20]
    }
   ```
- gamma: Mengontrol seberapa jauh pengaruh dari satu titik training terhadap titik lain. Nilai yang lebih besar membuat model lebih memfokuskan pada data poin yang dekat, sedangkan nilai kecil memperhitungkan poin yang lebih jauh. Ini sangat penting untuk mengontrol kompleksitas model pada SVM non-linier.
- C: Parameter regularisasi yang mengontrol trade-off antara margin yang lebih lebar (yang akan memberikan generalisasi yang lebih baik) dan kesalahan klasifikasi pada training data. Nilai kecil dari C menghasilkan margin yang lebih besar, namun memperbolehkan beberapa data salah klasifikasi.
  
Grid search ini mencoba beberapa nilai gamma untuk menangkap hubungan non-linier yang berbeda dalam data. C digunakan untuk mencari keseimbangan antara regularisasi dan akurasi pada training data. Kombinasi parameter ini membantu menemukan model SVM yang terbaik untuk dataset tersebut. Dan parameter terbaik yaitu C=15, gamma=0.1, probability=True, random_state=42 dengan aukrasi 0.8486842105263158

3. Gradient Boosting Classifier
  Parameter yang digunakan:
  ```python
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3]
}
  ```
- n_estimators: Jumlah pohon yang akan dibangun. Lebih banyak pohon biasanya meningkatkan akurasi, tetapi juga bisa menyebabkan overfitting. Nilai yang diuji dalam grid search memungkinkan pemilihan jumlah yang optimal.
- learning_rate: Mengontrol kontribusi dari setiap pohon terhadap prediksi akhir. Nilai yang lebih kecil sering kali bekerja lebih baik, tetapi membutuhkan lebih banyak pohon (n_estimators). Grid search ini menguji berbagai nilai untuk menemukan keseimbangan antara jumlah pohon dan learning rate.
- max_depth: Menentukan kedalaman maksimum setiap pohon. Pohon yang terlalu dalam bisa menangkap pola yang terlalu spesifik (overfitting), sementara pohon yang terlalu dangkal mungkin tidak cukup kuat untuk menangkap pola yang penting.
- min_samples_split: Jumlah minimum sampel yang dibutuhkan untuk memecah node. Nilai yang lebih tinggi cenderung membuat model lebih sederhana dan mengurangi overfitting.
- min_samples_leaf: Jumlah minimum sampel yang harus dimiliki oleh setiap daun pohon. Mengontrol ukuran daun untuk menghindari pohon yang terlalu rumit.
  
Parameter ini membantu mengatur kompleksitas pohon dalam Gradient Boosting dan mempengaruhi kemampuan model untuk menangkap pola dalam data tanpa overfitting. Dengan menguji berbagai nilai dari parameter-parameter ini, grid search berusaha mencari kombinasi yang memberikan performa terbaik. Dan parameter terbaik yaitu learning_rate=0.2, min_samples_leaf=2, random_state=42 dengab akurasi 0.8289473684210527

4. Random Forest Classifier
Parameter yang digunakan:
  ```python
  param_grid = {
      'n_estimators': [100, 130, 150],
      'criterion': ['gini', 'entropy'],
      'max_depth': [10, 15, 20, None],
      'max_features': [0.5, 0.75, 'sqrt', 'log2'],
      'min_samples_split': [2, 3, 4],
      'min_samples_leaf': [1, 2, 3]
  }
  ```
- n_estimators: Jumlah pohon yang akan digunakan dalam hutan. Nilai yang lebih besar biasanya menghasilkan model yang lebih kuat, tetapi dengan biaya waktu komputasi yang lebih besar.
- criterion: Fungsi yang digunakan untuk mengukur kualitas split. gini menghitung impuritas Gini, sedangkan entropy menghitung gain informasi. Keduanya bisa memberikan hasil yang berbeda pada dataset yang berbeda.
- max_depth: Menentukan kedalaman maksimum setiap pohon. Kedalaman yang terlalu besar bisa menyebabkan overfitting, jadi grid search menguji nilai yang bervariasi untuk menemukan yang terbaik.
- max_features: Menentukan jumlah fitur yang akan dipertimbangkan saat membuat split di setiap node. Nilai sqrt dan log2 adalah pilihan umum untuk meningkatkan efisiensi tanpa mengorbankan akurasi. 0.5 dan 0.75 berarti proporsi dari total fitur yang digunakan.
min_samples_split dan min_samples_leaf: Kontrol pembentukan pohon untuk memastikan setiap split atau daun memiliki sampel yang cukup untuk menghindari overfitting.

Random Forest membutuhkan pengaturan parameter yang mengatur jumlah pohon dan bagaimana setiap pohon dibangun. Grid search ini menguji parameter untuk menemukan keseimbangan yang baik antara bias dan varians, serta untuk memastikan bahwa setiap pohon dalam hutan berkontribusi optimal pada prediksi final. Dan parameter terbaik yaitu max_depth=10, min_samples_split=4, n_estimators=150,random_state=42 dengan akurasi 85.53%

## Evaluation

**Model Terbaik**

Berikut merupakan perbandingan dari performa model emppat algoritma dengan parameter terbaik

![{EE261FEC-1A00-47DF-8D34-8C615BFCB6A7}](https://github.com/user-attachments/assets/ef45dfa9-b9fc-44a8-a8ee-77152cc58908)
![{165316D1-5EDF-4B9D-9AEC-3A2ADBDB84D7}](https://github.com/user-attachments/assets/72cde069-432f-4542-8359-691f73ec1b6d)


Berdasarkan gambar, Logistic Regression terlihat sebagai model terbaik karena memiliki nilai tertinggi pada tiga metrik utama: Accuracy (0.862), Precision (0.851), dan F1 Score (0.856). Meskipun pada Recall (0.865) model ini bukan yang tertinggi, perbedaannya sangat kecil jika dibandingkan dengan Random Forest.

**Penjelasan Mengenai Metrik yang Digunakan**

- Accuracy (0.862): Metrik ini menunjukkan persentase prediksi yang benar dari keseluruhan data. Logistic Regression mencapai akurasi 86,2%, yang berarti bahwa model ini mampu memprediksi dengan benar sekitar 86,2% dari semua data yang diuji.
- Precision (0.851): Precision mengukur akurasi prediksi positif model, yaitu berapa banyak dari prediksi yang positif benar-benar positif. Dengan precision sebesar 85,1%, artinya dari seluruh prediksi positif, 85,1% di antaranya benar-benar merupakan kelas positif.
- Recall (0.865): Recall menunjukkan seberapa baik model mendeteksi semua data positif yang sebenarnya. Logistic Regression memiliki recall sebesar 86,5%, yang menunjukkan bahwa model ini cukup baik dalam mengidentifikasi kelas positif dari keseluruhan data positif yang tersedia.
- F1 Score (0.856): F1 Score adalah rata-rata harmonis dari precision dan recall. Nilai 85,6% berarti model memiliki keseimbangan yang baik antara presisi dan recall, membuatnya ideal untuk situasi di mana keseimbangan antara keduanya diperlukan.

**Hasil proyek berdasarkan metrik evaluasi**

Secara keseluruhan, Logistic Regression menunjukkan performa terbaik di antara model-model lain dengan skor metrik yang konsisten di semua aspek utama. Ini mengindikasikan bahwa model ini tidak hanya akurat, tetapi juga mampu menangani data positif dan negatif dengan cukup seimbang. Jika tujuannya adalah membuat prediksi yang stabil dan andal di semua metrik, Logistic Regression adalah pilihan yang optimal untuk proyek ini.

**Formula Metrik dan Cara Kerjanya**

Accuracy Formula:

$$ Accuracy = \frac{TP+TN}{TP+TN+FP+FN} $$
 
Dimana:

- TP = True Positives (prediksi benar pada kelas positif)
- TN = True Negatives (prediksi benar pada kelas negatif)
- FP = False Positives (prediksi salah, kelas sebenarnya negatif)
- FN = False Negatives (prediksi salah, kelas sebenarnya positif)

Cara Kerja: Accuracy menghitung persentase dari semua prediksi yang benar, baik itu prediksi positif maupun negatif.

Precision Formula:

$$ Precision = \frac{TP)}{TP+FP} $$
 
Cara Kerja: Precision fokus pada kualitas prediksi positif, mengukur seberapa banyak prediksi positif benar dari total prediksi positif yang diberikan model.

Recall Formula:

 $$ Recall = \frac{TP}{TP+FN} $$

 
Cara Kerja: Recall mengukur kemampuan model untuk menangkap semua kasus positif yang sebenarnya dari keseluruhan sampel positif.

F1 Score Formula:

$$ F1 = 2 * \frac{Precision * Recall}{Precision + Recall} $$
 
Cara Kerja: F1 Score memberikan keseimbangan antara precision dan recall, sangat berguna ketika distribusi kelas tidak seimbang atau jika penting untuk mempertimbangkan "false positives" dan "false negatives" secara bersamaan.

Metrik-metrik ini bersama-sama memberikan gambaran lengkap tentang performa model, dan dengan skor yang baik di setiap metrik, Logistic Regression menunjukkan hasil yang kuat pada data yang diuji.
