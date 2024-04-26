#Machine Learning? Apa Itu?
#Machine Learning adalah teknik dimana komputer dapat mengekstraksi atau mempelajari pola dari suatu data, kemudian dengan pola yang telah dipelajari dari data historis, komputer mampu mengenali dan memprediksi trend, hasil atau kejadian di masa mendatang atau dari observasi baru tanpa perlu diprogram secara eksplisit
#Selain mengenali email sebagai spam atau ukan spam ada banyak contoh penggunaan machine learning lainnya, seperti memprediksi harga saham, pengenalan wajah (face recognition), mengenali tulisan tangan, mendeteksi fraud/scam kartu kredit, memprediksi cuaca, dan memprediksi permintaan barang

#Terminologi Machine Learning
#Dalam pembuatan model machine learning tentunya dibutuhkan data. Sekumpulan data yang digunakan dalam machine learning disebut DATASET, yang kemudian dibagi/di-split menjadi training dataset dan test dataset.
#1. TRAINING DATASET digunakan untuk membuat/melatih model machine learning, sedangkan TEST DATASET digunakan untuk menguji performa/akurasi dari model yang telah dilatih/di-training.
#2. Teknik atau pendekatan yang digunakan untuk membangun model disebut ALGORITHM seperti Decision Tree, K-NN, Linear Regression, Random Forest, dsb. dan output atau hasil dari proses melatih algorithm dengan suatu dataset disebut MODEL.
#3. Umumnya dataset disajikan dalam bentuk tabel yang terdiri dari baris dan kolom. Bagian Kolom adalah FEATURE atau VARIABEL data yang dianalisa, sedangkan bagian baris adalah DATA POINT/OBSERVATION/EXAMPLE.
#4. Hal yang menjadi target prediksi atau hal yang akan diprediksi dalam machine learning disebut LABEL/CLASS/TARGET. Dalam statistika/matematika, LABEL/CLASS/TARGET ini dinamakan dengan Dependent Variabel, dan FEATURE adalah Independent Variabel.

#Senja dan Aksara akan membuat suatu model machine learning yang dapat memprediksi apakah customer akan melakukan pembelian setelah mengunjungi beberapa halaman e-commerce. Target adalah 1 jika customer melakukan pembelian dan 0 jika tidak ada pembelian. Berikut, 10 baris pertama dari dataset yang digunakan oleh Senja dan Aksara. 
# Kolom manakah yang dapat digunakan oleh Senja dan Aksara sebagai predictor variable atau feature?
#ProductRelated, BounceRates, ExitRates, Weekend

#Supervised and Unsupervised Learning
#Machine Learning itu terbagi menjadi 2 tipe yaitu supervised dan unsupervised Learning. 
# Jika LABEL/CLASS dari dataset sudah diketahui maka dikategorikan sebagai supervised learning, 
# dan jika Label belum diketahui maka dikategorikan sebagai unsupervised learning
#kasusnya yang email tadi, masuknya ke mana ya
#Mengenali email sebagai spam atau bukan spam tergolong sebagai supervised learning, 
# karena kita mengolah dataset yang berisi data point yang telah diberi LABEL ”spam” dan “not spam”. 
# Sedangkan jika kita ingin mengelompokkan customer ke dalam beberapa segmentasi berdasarkan variabel-variabel 
# seperti pendapatan, umur, hobi, atau jenis pekerjaan, maka tergolong sebagai unsupervised learning

#Jika Senja ingin membuat model Machine Learning 
# untuk mendeteksi transaksi kartu kredit sebagai fraud/scam di suatu e-commerce, 
# Tipe machine learning manakah yang digunakan oleh Aksara untuk membuat model? 
# Dan Jika Aksara ingin membuat segmentasi user dari suatu e-commerce, 
# Tipe machine learning manakah yang tepat digunakan?
#Senja menggunakan Supervised Learning dan Aksara menggunakan Unsupervised Learning.

#Misalkan Aksara ingin membuat model Supervised Machine Learning 
# untuk memprediksi apakah suatu email adalah "SPAM" atau "BUKAN SPAM" Manakah dari pernyataan berikut ini yang benar?
#Aksara tidak dapat menggunakan dataset yang tidak memiliki LABEL “spam” dan “bukan spam"

#Pilih Algorithm yang Mana?
#aku sudah tahu bahwa proyek aku adalah tipe supervised learning, terus gimana cara menentukan algorithm yang cocok?
# dan  Kalau sudah paham bahwa problem aku adalah tipe unsupervised learning, lalu apa algorithm yang tepat untuk kasus seperti ini?
#jadi begini,  penting untuk diingat bahwa tidak ada ML algorithm yang cocok atau fit untuk diaplikasikan di semua problem.
# Oleh karena itu, proses ini terkadang memerlukan trial & error seperti research, bahkan experienced data scientist pun tidak akan tahu apakah algorithm itu akan tepat atau tidak jika tidak mencoba. 
# Biasanya, data scientist akan mencoba beberapa algorithm dan membandingkan performansi dari algorithm - algorithm tersebut. 
# Algorithm dengan performansi yang paling baiklah yang dipilih sebagai model, Jadi lebih banyak mencoba dan praktik ya untuk tahu yang tepat dan relevannya
#Iya, selain itu  untuk supervised learning, jika LABEL dari dataset kalian berupa numerik atau kontinu variabel seperti harga, 
# dan  jumlah penjualan, kita memilih metode REGRESI dan jika bukan numerik atau diskrit maka digunakan metode KLASIFIKASI. 
# Untuk unsupervised learning, seperti segmentasi customer, kita menggunakan metode CLUSTERING

#INTINYA ADALAH
# ML-->UNSUPERVISED LEARNING (Label atau data output tidak diketahui)--> CLUSTERING
# ML-->SUPERVISED LEARNING (Label atau output data diketahui) --> CLASSIFICATION ATAU REGRESSION

#Senja memberikan tantangan kepada Aksara untuk membantu mereka mengembangkan intuisi dalam membedakan yang mana problem klasifikasi dan yang mana problem regresi.
# Aksara diminta untuk mengidentifikasi dari 4 contoh aplikasi machine learning berikut, manakah yang termasuk permasalahan klasifikasi. 
# Bantulah Aksara untuk menyelesaikan tantangan dari Senja.
# JWB: Menggunakan financial data yang memiliki label untuk memprediksi apakah harga saham akan naik atau turun di minggu depan.

#Eksplorasi Data: Memahami Data dengan Statistik - Part 1
#“Selanjutnya saya akan menjelaskan secara singkat tentang tahapan-tahapan dalam pembuatan model machine learning.
#Membuat model machine learning tidak serta-merta langsung modelling, ada tahapan sebelumnya yang penting untuk dilakukan sehingga kita menghasilkan model yang baik. 
#Untuk penjelasan ini, kalian akan mempraktekkan langsung ya. Kita akan memanfaatk#Machine Learning? Apa Itu?
#Machine Learning adalah teknik dimana komputer dapat mengekstraksi atau mempelajari pola dari suatu data, kemudian dengan pola yang telah dipelajari dari data historis, komputer mampu mengenali dan memprediksi trend, hasil atau kejadian di masa mendatang atau dari observasi baru tanpa perlu diprogram secara eksplisit
#Selain mengenali email sebagai spam atau ukan spam ada banyak contoh penggunaan machine learning lainnya, seperti memprediksi harga saham, pengenalan wajah (face recognition), mengenali tulisan tangan, mendeteksi fraud/scam kartu kredit, memprediksi cuaca, dan memprediksi permintaan barang

#Terminologi Machine Learning
#Dalam pembuatan model machine learning tentunya dibutuhkan data. Sekumpulan data yang digunakan dalam machine learning disebut DATASET, yang kemudian dibagi/di-split menjadi training dataset dan test dataset.
#1. TRAINING DATASET digunakan untuk membuat/melatih model machine learning, sedangkan TEST DATASET digunakan untuk menguji performa/akurasi dari model yang telah dilatih/di-training.
#2. Teknik atau pendekatan yang digunakan untuk membangun model disebut ALGORITHM seperti Decision Tree, K-NN, Linear Regression, Random Forest, dsb. dan output atau hasil dari proses melatih algorithm dengan suatu dataset disebut MODEL.
#3. Umumnya dataset disajikan dalam bentuk tabel yang terdiri dari baris dan kolom. Bagian Kolom adalah FEATURE atau VARIABEL data yang dianalisa, sedangkan bagian baris adalah DATA POINT/OBSERVATION/EXAMPLE.
#4. Hal yang menjadi target prediksi atau hal yang akan diprediksi dalam machine learning disebut LABEL/CLASS/TARGET. Dalam statistika/matematika, LABEL/CLASS/TARGET ini dinamakan dengan Dependent Variabel, dan FEATURE adalah Independent Variabel.

#Senja dan Aksara akan membuat suatu model machine learning yang dapat memprediksi apakah customer akan melakukan pembelian setelah mengunjungi beberapa halaman e-commerce. Target adalah 1 jika customer melakukan pembelian dan 0 jika tidak ada pembelian. Berikut, 10 baris pertama dari dataset yang digunakan oleh Senja dan Aksara. 
# Kolom manakah yang dapat digunakan oleh Senja dan Aksara sebagai predictor variable atau feature?
#ProductRelated, BounceRates, ExitRates, Weekend

#Supervised and Unsupervised Learning
#Machine Learning itu terbagi menjadi 2 tipe yaitu supervised dan unsupervised Learning. 
# Jika LABEL/CLASS dari dataset sudah diketahui maka dikategorikan sebagai supervised learning, 
# dan jika Label belum diketahui maka dikategorikan sebagai unsupervised learning
#kasusnya yang email tadi, masuknya ke mana ya
#Mengenali email sebagai spam atau bukan spam tergolong sebagai supervised learning, 
# karena kita mengolah dataset yang berisi data point yang telah diberi LABEL ”spam” dan “not spam”. 
# Sedangkan jika kita ingin mengelompokkan customer ke dalam beberapa segmentasi berdasarkan variabel-variabel 
# seperti pendapatan, umur, hobi, atau jenis pekerjaan, maka tergolong sebagai unsupervised learning

#Jika Senja ingin membuat model Machine Learning 
# untuk mendeteksi transaksi kartu kredit sebagai fraud/scam di suatu e-commerce, 
# Tipe machine learning manakah yang digunakan oleh Aksara untuk membuat model? 
# Dan Jika Aksara ingin membuat segmentasi user dari suatu e-commerce, 
# Tipe machine learning manakah yang tepat digunakan?
#Senja menggunakan Supervised Learning dan Aksara menggunakan Unsupervised Learning.

#Misalkan Aksara ingin membuat model Supervised Machine Learning 
# untuk memprediksi apakah suatu email adalah "SPAM" atau "BUKAN SPAM" Manakah dari pernyataan berikut ini yang benar?
#Aksara tidak dapat menggunakan dataset yang tidak memiliki LABEL “spam” dan “bukan spam"

#Pilih Algorithm yang Mana?
#aku sudah tahu bahwa proyek aku adalah tipe supervised learning, terus gimana cara menentukan algorithm yang cocok?
# dan  Kalau sudah paham bahwa problem aku adalah tipe unsupervised learning, lalu apa algorithm yang tepat untuk kasus seperti ini?
#jadi begini,  penting untuk diingat bahwa tidak ada ML algorithm yang cocok atau fit untuk diaplikasikan di semua problem.
# Oleh karena itu, proses ini terkadang memerlukan trial & error seperti research, bahkan experienced data scientist pun tidak akan tahu apakah algorithm itu akan tepat atau tidak jika tidak mencoba. 
# Biasanya, data scientist akan mencoba beberapa algorithm dan membandingkan performansi dari algorithm - algorithm tersebut. 
# Algorithm dengan performansi yang paling baiklah yang dipilih sebagai model, Jadi lebih banyak mencoba dan praktik ya untuk tahu yang tepat dan relevannya
#Iya, selain itu  untuk supervised learning, jika LABEL dari dataset kalian berupa numerik atau kontinu variabel seperti harga, 
# dan  jumlah penjualan, kita memilih metode REGRESI dan jika bukan numerik atau diskrit maka digunakan metode KLASIFIKASI. 
# Untuk unsupervised learning, seperti segmentasi customer, kita menggunakan metode CLUSTERING

#INTINYA ADALAH
# ML-->UNSUPERVISED LEARNING (Label atau data output tidak diketahui)--> CLUSTERING
# ML-->SUPERVISED LEARNING (Label atau output data diketahui) --> CLASSIFICATION ATAU REGRESSION

#Senja memberikan tantangan kepada Aksara untuk membantu mereka mengembangkan intuisi dalam membedakan yang mana problem klasifikasi dan yang mana problem regresi.
# Aksara diminta untuk mengidentifikasi dari 4 contoh aplikasi machine learning berikut, manakah yang termasuk permasalahan klasifikasi. 
# Bantulah Aksara untuk menyelesaikan tantangan dari Senja.
# JWB: Menggunakan financial data yang memiliki label untuk memprediksi apakah harga saham akan naik atau turun di minggu depan.

#Eksplorasi Data: Memahami Data dengan Statistik - Part 1
#“Selanjutnya saya akan menjelaskan secara singkat tentang tahapan-tahapan dalam pembuatan model machine learning.
#Membuat model machine learning tidak serta-merta langsung modelling, ada tahapan sebelumnya yang penting untuk dilakukan sehingga kita menghasilkan model yang baik. 
#Untuk penjelasan ini, kalian akan mempraktekkan langsung ya. Kita akan memanfaatkan Pandas library. 
#Pandas cukup powerful untuk digunakan dalam menganalisa, memanipulasi dan membersihkan data. Siap, Aksara?”
#Aku mengangguk, siap dengan laptopku.
#“Oke, Pertama- tama,  kita check dimensi data kita terlebih dahulu. Aksara, silahkan load datanya dan gunakan .shape, .head(), .info(), dan .describe() untuk mengeksplorasi dataset secara berurut. Dataset ini adalah data pembeli online yang mengunjungi website dari suatu e-commerce selama setahun, yaitu 'https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv',” perintah Senja.
import pandas as pd
dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')
print('Shape dataset:', dataset.shape)
print('\nLima data teratas:\n',dataset.head())
print('\nInformasi dataset:')
print(dataset.info())
print('\nStatistik deskriptif:\n',dataset.describe())
#“Nah, dengan mengetahui dimensi data yaitu jumlah baris dan kolom, kita bisa mengetahui apakah data kita terlalu banyak atau justru sangat sedikit.
# Jika data terlalu banyak, waktu melatih model akan lebih lama, sedangkan jika data terlalu sedikit, performansi model yang kita hasilkan mungkin tidak cukup bagus, 
# karena tidak mampu mengenali pola dengan baik. Sudah lebih paham sekarang?”
#Aku mengacungkan jempol. Kalau dibareng praktik, memang jadinya lebih jelas.
#Berdasarkan praktek yang telah dilakukan pada bagian sebelumnya, pernyataan manakah yang tidak sesuai berdasarkan hasil eksplorasi?
Nilai rata - rata (mean) dari feature BounceRates adalah 0.2

#Eksplorasi Data: Memahami Data dengan Statistik - Part 2
#Setelah selesai materi tadi, aku kembali diajak memahami eksplorasi data. 
#Banyak sekali materi baru hari ini!
#“Kita lanjut yah, Aksara. 
#Data eksplorasi tidaklah cukup dengan mengetahui dimensi data dan statistical properties saja, 
#tetapi kita juga perlu sedikit menggali tentang hubungan atau korelasi dari setiap feature, 
#karena beberapa algorithm seperti linear regression dan logistic regression akan menghasilkan model dengan performansi yang buruk jika
#kita menggunakan feature/variabel saling dependensi atau berkorelasi kuat (multicollinearity).
#Jadi, jika kita sudah tahu bahwa data kita berkorelasi kuat, kita bisa menggunakan algorithm lain yang tidak sensitif terhadap hubungan korelasi dari feature/variabel seperti decision tree.”
#”Aksara, coba sekarang lanjutkan eksplorasi data untuk melihat korelasi dan distribusi dataset.”
#Aku segera berkutat dengan susunan kodenya:
dataset_corr = dataset.corr()
print('Korelasi dataset:\n', dataset.corr())
print('Distribusi Label (Revenue):\n', dataset['Revenue'].value_counts())
#kenapa mengetahui distribusi LABEL dari dataset itu penting?”
#“Pertanyaan menarik, mengetahui distribusi label sangat penting untuk permasalahan klasifikasi, 
#karena jika distribusi label sangat tidak seimbang (imbalanced class),  
#maka akan sulit bagi model untuk mempelajari pola dari LABEL yang sedikit dan hasilnya bisa misleading. 
#Contohnya, kita memiliki 100 row data, 90 row adalah non fraud dan 10 row adalah fraud. 
#Jika kita menggunakan data ini tanpa melakukan treatment khusus (handling imbalanced class), 
#maka kemungkinan besar model kita akan cenderung mengenali observasi baru sebagai non-fraud, 
#dan hal ini tentunya tidak diinginkan,” jelas Senja panjang lebar.

#Tugas Praktek:
#Sekarang coba inspeksi nilai korelasi dari fitur-fitur berikut pada dataset_corr yang telah diberikan sebelumnya
#1. ExitRates dan BounceRates
#2. Revenue dan PageValues
#3. TrafficType dan Weekend
#Jika dengan benar ditulis dan dijalankan maka output seperti berikut ini yang akan diperoleh:
dataset_corr = dataset.corr()
print('Korelasi dataset:\n', dataset.corr())
print('Distribusi Label (Revenue):\n', dataset['Revenue'].value_counts())
# Tugas praktek
print('\nKorelasi BounceRates-ExitRates:', dataset_corr.loc['BounceRates', 'ExitRates'])
print('\nKorelasi Revenue-PageValues:', dataset_corr.loc['Revenue', 'PageValues'])
print('\nKorelasi TrafficType-Weekend:', dataset_corr.loc['TrafficType', 'Weekend'])

#Berdasarkan hasil eksplorasi, pilihlah pernyataan berikut ini yang tidak sesuai dengan hasil eksplorasi adalah 
Distribusi label dari dataset tidak seimbang karena total data point dengan label 1 adalah 10422 dan total data point dengan label 0 adalah 1908.

#Eksplorasi Data: Memahami Data dengan Visual
#“Aksara, satu lagi, dalam mengeksplorasi data, kita perlu untuk memahami data dengan visual.”
#Aku tertarik, “Maksudnya?”
#“Begini, selain dengan statistik, kita juga bisa melakukan eksplorasi data dalam bentuk visual. 
#Dengan visualisasi kita dapat dengan mudah dan cepat dalam memahami data, 
#bahkan dapat memberikan pemahaman yang lebih baik terkait hubungan setiap variabel/ features.
#Misalnya kita ingin melihat distribusi label dalam bentuk visual, dan jumlah pembelian saat weekend. 
#Kita dapat memanfaatkan matplotlib library untuk membuat chart yang menampilkan perbandingan jumlah yang membeli (1) dan tidak membeli (0), 
#serta perbandingan jumlah pembelian saat weekend,” tambah Senja sembari menampilkan contoh kodenya:
import matplotlib.pyplot as plt
import seaborn as sns
# checking the Distribution of customers on Revenue
plt.rcParams['figure.figsize']=(12,5)
plt.subplot(1,2,1)
sns.countplot(dataset['Revenue'],palette='pastel')
plt.title('Buy or not', fontsize=20)
plt.xlabel('Revenue or not', fontsize=14)
plt.ylabel('count', fontsize=14)
# checking the Distribution of customers on Weekend
plt.subplot(1,2,2)
sns.countplot(dataset['Weekend'],palette='inferno')
plt.title('Purchase on Weekends', fontsize=20)
plt.xlabel('Weekend or not', fontsize=14)
plt.ylabel('count', fontsize=14)
plt.show()

#Tugas Praktek
#Aku kemudian diminta Senja untuk membuat visualisasi berupa histogram yang menggambarkan jumlah customer untuk setiap Region.
#Dalam membuat visualisasi ini aku akan menggunakan dataset['region'] untuk membuat histogram,
#dan berikan judul 'Distribution of Customers' pada title, 
#'Region Codes' sebagai label axis-x dan 
#'Count Users' sebagai label axis-y.
import matplotlib.pyplot as plt
# visualizing the distribution of customers around the Region
plt.hist(dataset['Region'], color = 'lightblue')
plt.title('Distribution of Customers', fontsize = 20)
plt.xlabel('Region Codes', fontsize = 14)
plt.ylabel('Count Users', fontsize = 14)
plt.show()

#Data Pre-processing: Handling Missing Value - Part 1
#Setelah kita melakukan eksplorasi data, kita akan melanjutkan ke tahap data pre-processing. 
# Seperti yang saya jelaskan sebelumnya, raw data kita belum tentu bisa langsung digunakan untuk pemodelan. 
# Jika kita memiliki banyak missing value, maka akan mengurangi performansi model dan juga beberapa algorithm machine learning tidak dapat memproses data dengan missing value. 
# Oleh karena itu, kita perlu mengecek apakah terdapat missing value dalam data atau tidak. 
# Jika tidak, maka kita tidak perlu melakukan apa-apa dan bisa melanjutkan ke tahap berikutnya. 
# Jika ada, maka kita perlu melakukan treatment khusus untuk missing value ini
#Pengecekan missing value dapat dilakukan dengan code berikut
#menggunakan metod .isnull pada dataset dan kemudian men-chaining-nya dengan method sum. 
# Untuk jumlah keseluruhan missing value digunakan chaining method sum sekali lagi. 
#checking missing value for each feature  
print('Checking missing value for each feature:')
print(dataset.isnull().sum())
#Counting total missing value
print('\nCounting total missing value:')
print(dataset.isnull().sum().sum())

#Data Pre-processing: Handling Missing Value - Part 2
#Metode ini dapat diterapkan jika tidak banyak missing value dalam data, 
# sehingga walaupun data point ini dihapus, 
# kita masih memiliki sejumlah data yang cukup untuk melatih model Machine Learning. 
# Tetapi jika kita memiliki banyak missing value dan tersebar di setiap variabel, 
# maka metode menghapus missing value tidak dapat digunakan. 
# Kita akan kehilangan sejumlah data yang tentunya mempengaruhi performansi model. 
# Kita bisa menghapus data point yang memiliki missing value dengan fungsi .dropna( ) dari pandas library. 
# Fungsi dropna( ) akan menghapus data point atau baris yang memiliki missing value.
#Drop rows with missing value
dataset_clean = dataset.dropna()
print('Ukuran dataset_clean:', dataset_clean.shape)

#Data Pre-processing: Handling Missing Value - Part 3
#“Kalau tidak dihapus, ada metode lain yang bisa dipakai?”
#Kita bisa menggunakan metode impute missing value, yaitu mengisi record yang hilang ini dengan suatu nilai. 
#Ada berbagai teknik dalam metode imputing, mulai dari yang paling sederhana yaitu mengisi missing value dengan nilai mean, median, modus, atau nilai konstan, 
#sampai teknik paling advance yaitu dengan menggunakan nilai yang diestimasi oleh suatu predictive model. 
#Untuk kasus ini, kita akan menggunakan imputing sederhana yaitu menggunakan nilai rataan atau mean
print("Before imputation:")
# Checking missing value for each feature
print(dataset.isnull().sum())
# Counting total missing value
print(dataset.isnull().sum().sum())

print("\nAfter imputation:")
# Fill missing value with mean of feature value  
dataset.fillna(dataset.mean(), inplace = True)
# Checking missing value for each feature  
print(dataset.isnull().sum())
# Counting total missing value  
print(dataset.isnull().sum().sum())

#Tugas Praktek
#Praktekkan metode imputing missing value dengan menggunakan nilai median.
import pandas as pd
dataset1 = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')

print("Before imputation:")
# Checking missing value for each feature  
print(dataset1.isnull().sum())
# Counting total missing value  
print(dataset1.isnull().sum().sum())

print("\nAfter imputation:")
# Fill missing value with median of feature value  
dataset1.fillna(dataset1.median(), inplace = True)
# Checking missing value for each feature  
print(dataset1.isnull().sum())
# Counting total missing value  
print(dataset1.isnull().sum().sum())

#Data Preprocessing: Scaling
#Setelah berhasil menangani missing value, sekarang mari kita mempelajari tahapan preprocessing selanjutnya. 
# Aksara, tolong tampilkan kembali 5 dataset teratas dan deskripsi statistik dari dataset. 
# Coba perhatikan, rentang nilai dari setiap feature cukup bervariasi. 
# Misalnya, ProductRelated_Duration vs BounceRates.  
# ProductRelated_Duration memiliki rentang nilai mulai dari 0 - 5000; 
# sedangkan BounceRates rentang nilainya 0 - 1.
#Beberapa machine learning seperti K-NN dan gradient descent mengharuskan semua variabel memiliki rentang nilai yang sama, karena jika tidak sama,
# feature dengan rentang nilai terbesar misalnya ProductRelated_Duration otomatis akan menjadi feature yang paling mendominasi dalam proses training/komputasi, 
# sehingga model yang dihasilkan pun akan sangat bias. 
# Oleh karena itu, sebelum memulai training model, kita terlebih dahulu perlu melakukan data rescaling ke dalam rentang 0 dan 1, 
# sehingga semua feature berada dalam rentang nilai tersebut, yaitu nilai max = 1 dan nilai min = 0.
# Data rescaling ini dengan mudah dapat dilakukan di Python menggunakan .MinMaxScaler( ) dari Scikit-Learn library.
#Kenapa ke range 0 - 1, tidak menggunakan range yang lain?
#Karena rumus dari rescaling adalah
Zi=Xi-min(X)/max(x)-min(x)
#dengan rumus ini, nilai max data akan menjadi 1 dan nilai min menjadi 0; 
# dan nilai lainnya berada di rentang keduanya. 
# Rumus ini tidak memungkinkan adanya rentang nilai selain 0 – 1

#Tugas Praktek
from sklearn.preprocessing import MinMaxScaler  
#Define MinMaxScaler as scaler  
scaler = MinMaxScaler()
#Apply fit_transfrom to scale selected feature  
dataset[scaling_column] = scaler.fit_transform(dataset[scaling_column])
#code diatas merupakan basic code untuk proses scaling dengan asumsi bahwa semua feature adalah numerik. 
# Tetapi, ketika menjalankan code tersebut untuk dataset online_raw, pasti akan terjadi error. 
# Proses scaling hanya bisa dilakukan untuk feature dengan tipe numerik, 
# sedangkan dalam dataset online_raw, terdapat feature dengan tipe string atau karakter dan categorical, 
# seperti Month, VisitorType, Region. Oleh karena itu, kita tidak dapat langsung menggunakan code di atas, 
# tetapi kita perlu terlebih dahulu menyeleksi feature - feature dari dataset yang bertipe numerik.
from sklearn.preprocessing import MinMaxScaler  
#Define MinMaxScaler as scaler  
scaler = MinMaxScaler()  
#list all the feature that need to be scaled  
scaling_column = ['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues']
#Apply fit_transfrom to scale selected feature  
dataset[scaling_column] = scaler.fit_transform(dataset[scaling_column])
#Cheking min and max value of the scaling_column
print(dataset[scaling_column].describe().T[['min','max']])

#Data Pre-processing: Konversi string ke numerik
#kita memiliki dua kolom yang bertipe object yang dinyatakan dalam tipe data str, 
# yaitu kolom 'Month' dan 'VisitorType'. 
# Karena setiap algoritma machine learning bekerja dengan menggunakan nilai numeris, 
# maka kita perlu mengubah kolom dengan tipe pandas object atau str ini ke bertipe numeris. 
# Untuk itu, kita list terlebih dahulu apa saja label unik di kedua kolom ini
#bagaimana ya cara menrubah tipe pandas object ini ke numerik (int, float) ya?
#kita dapat menggunakan LabelEncoder dari sklearn.preprocessing untuk merubah kedua kolom ini seperti ini
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Convert feature/column 'Month'
LE=LabelEncoder()
dataset['Month']=LE.fit_transform(dataset['Month'])
print(LE.classes_)
print(np.sort(dataset['Month'].unique()))
print('')

# Convert feature/column 'VisitorType'
LE=LabelEncoder()
dataset['VisitorType']=LE.fit_transform(dataset['VisitorType'])
print(LE.classes_)
print(np.sort(dataset['VisitorType'].unique()))
#Bisa dilihat, kan, Aksara bahwa LabelEncoder akan mengurutkan label secara otomatis secara alfabetik, 
# posisi/indeks dari setiap label ini digunakan sebagai nilai numeris konversi pandas objek ke numeris 
# (dalam hal ini tipe data int). 
# Dengan demikian kita telah membuat dataset kita menjadi dataset bernilai numeris seluruhnya yang siap digunakan untuk pemodelan dengan algoritma machine learning tertentu

#akhirnya, setelah data eksplorasi dan preprocessing, 
# datasetnya sudah siap untuk digunakan dalam proses modelling. 
# Kalau dipikir-pikir, preprocessing ini panjang juga dan ribet, aku lebih senang langsung modelling aja,
# ” komentarku. ”Sebenarnya tidak panjang, 
# ini karena kamu  masih tahap belajar sehingga kamu  perlu mengerti konsepnya dan tidak asal membuat model, 
# tetapi ketika kamu sudah  mulai implementasi proses ini akan otomatis dilakukan sehingga tidak terasa panjang lagi.
#lanjutkan. Pertama-tama saya akan mengenalkan kamu pada library Scikit - Learn. 
# Scikit-learn adalah library untuk machine learning bagi para pengguna python yang memungkinkan kita melakukan berbagai pekerjaan dalam Data Science, 
# seperti regresi (regression), klasifikasi (classification), pengelompokkan/penggugusan (clustering), 
# data preprocessing, dimensionality reduction, dan model selection (pembandingan, validasi, 
# dan pemilihan parameter maupun model)
#Ada beberapa library machine learning di Python seperti Keras, 
# tetapi Scikit - Learn adalah yang paling basic sehingga jika kita menguasai scikit-learn, 
# kita dapat dengan mudah mempelajari library machine learning yang lain.

#Features & Label
#Dalam dataset user online purchase, label target sudah diketahui, 
#yaitu kolom Revenue yang bernilai 1 untuk user yang membeli dan 0 untuk yang tidak membeli, 
#sehingga pemodelan yang dilakukan ini adalah klasifikasi. 
#Nah, untuk melatih dataset menggunakan Scikit-Learn library, dataset perlu dipisahkan ke dalam Features dan Label/Target. 
#Variabel Feature akan terdiri dari variabel yang dideklarasikan sebagai X dan [Revenue] adalah variabel Target yang dideklarasikan sebagai y. 
#Gunakan fungsi drop() untuk menghapus kolom [Revenue] dari dataset.
# removing the target column Revenue from dataset and assigning to X
x=dataset.drop(['Revenue'],axis=1)
# assigning the target column Revenue to y
y=dataset['Revenue']
# checking the shapes
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

#Training dan Test Dataset
#ebelum kita melatih model dengan suatu algorithm machine , 
# seperti yang saya jelaskan sebelumnya, 
# dataset perlu kita bagi ke dalam training dataset dan test dataset dengan perbandingan 80:20. 80% 
# digunakan untuk training dan 20% untuk proses testing.
#Perbandingan lain yang biasanya digunakan adalah 75:25. 
# Hal penting yang perlu diketahui adalah scikit-learn tidak dapat memproses dataframe dan hanya mengakomodasi format data tipe Array. 
# Tetapi kalian tidak perlu khawatir, fungsi train_test_split( ) dari Scikit-Learn, 
# otomatis mengubah dataset dari dataframe ke dalam format array.
#Fungsi Training adalah melatih model untuk mengenali pola dalam data, 
# sedangkan testing berfungsi untuk memastikan bahwa model yang telah dilatih tersebut mampu dengan baik memprediksi label dari new observation dan belum dipelajari oleh model sebelumnya. 
# Lebih baik kita praktik saja ya, tampaknya kalau praktik kamu lebih paham
#silahkan bagi dataset ke dalam Training dan Testing dengan melanjutkan coding yang  sudah kukerjakan ini. 
# Gunakan test_size = 0.2 dan tambahkan argumen random_state = 0,  
# pada fungsi train_test_split( ).
from sklearn.model_selection import train_test_split
# splitting the X, and y
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# checking the shapes
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

#Training Model: Fit
#Sekarang saatnya kita melatih model atau training. 
# Dengan Scikit-Learn, proses ini menjadi sangat sederhana. 
# Kita cukup memanggil nama algorithm yang akan kita gunakan, 
# biasanya disebut classifier untuk problem klasifikasi, 
# dan regressor untuk problem regresi.
#sebagai contoh, kita akan menggunakan Decision Tree. 
# Kita hanya perlu memanggil fungsi DecisionTreeClassifier() yang kita namakan “model”. 
# Kemudian menggunakan fungsi .fit() 
# dan X_train, y_train untuk melatih classifier tersebut dengan training dataset, seperti ini:
from sklearn.tree import DecisionTreeClassifier
# Call the classifier
model=DecisionTreeClassifier()
# Fit the classifier to the training data
model=model.fit(X_train, y_train)

#Training Model: Predict
#setelah model/classifier terbentuk, 
# selanjutnya kita menggunakan model ini untuk memprediksi LABEL dari testing dataset (X_test), 
# menggunakan fungsi .predict(). 
# Fungsi ini akan mengembalikan hasil prediksi untuk setiap data point dari X_test dalam bentuk array. 
# Proses ini kita kenal dengan TESTING
#melanjutkan proses testing menggunakan fungsi .predict() seperti ini
# Apply the classifier/model to the test data
y_pred=model.predict(X_test)
print(y_pred.shape)

#Evaluasi Model Performance - Part 1
#sekarang kita melanjutkan di tahap terakhir dari modelling yaitu evaluasi hasil model. 
# Untuk evaluasi model performance, setiap algorithm mempunyai metrik yang berbeda-beda. 
# Sekarang saya akan menjelaskan sedikit metrik apa saja yang umumnya digunakan. 
# Metrik paling sederhana untuk mengecek performansi model adalah accuracy.
#Kita bisa munculkan dengan fungsi .score( ). 
# Tetapi, di banyak real problem, accuracy saja tidaklah cukup. 
# Metode lain yang digunakan adalah dengan Confusion Matrix.
# Confusion Matrix merepresentasikan perbandingan prediksi dan real LABEL dari test dataset yang dihasilkan oleh algoritma ML,” 
# tukas Senja sambil membuka  template dari confusion Matrix untukku:
True Positive (TP): Jika user diprediksi (Positif) membeli ([Revenue] = 1]), dan memang benar(True) membeli.
True Negative (TN): Jika user diprediksi tidak (Negatif) membeli dan aktualnya user tersebut memang (True) membeli.
False Positive (FP): Jika user diprediksi Positif membeli, tetapi ternyata tidak membeli (False).
False Negatif (FN): Jika user diprediksi tidak membeli (Negatif), tetapi ternyata sebenarnya membeli.

#Evaluasi Model Performance - Part 2
#Untuk menampilkan confusion matrix cukup menggunakan fungsi confusion_matrix() dari Scikit-Learn
from sklearn.metrics import confusion_matrix, classification_report
# evaluating the model
print('Training Accuracy :', model.score(X_train, y_train))
print('Testing Accuracy :', model.score(X_test, y_test))
# confusion matrix
print('\nConfusion matrix:')
cm=confusion_matrix(y_test, y_pred)
print(cm)\
#Berdasarkan confusion matrix, dapat mengukur metrik - metrik berikut :
Accuracy = (TP + TN ) / (TP+FP+FN+TN)
Precision = (TP) / (TP+FP)
Recall = (TP) / (TP + FN)
F1 Score = 2 * (Recall*Precission) / (Recall + Precission)
#Tidak perlu menghitung nilai ini secara manual. 
#Cukup gunakan  fungsi classification_report() untuk memunculkan hasil perhitungan metrik - metrik tersebut.
# classification report
print('\nClassification report:')
cr=classification_report(y_test,y_pred)
print(cr)

#Pakai Metrik yang Mana?
#Jika dataset memiliki jumlah data False Negatif dan False Positif yang seimbang (Symmetric), 
#maka bisa gunakan Accuracy, tetapi jika tidak seimbang, maka sebaiknya menggunakan F1-Score.
#Dalam suatu problem, jika lebih memilih False Positif lebih baik terjadi daripada False Negatif, 
#misalnya: Dalam kasus Fraud/Scam, kecenderungan model mendeteksi transaksi sebagai fraud walaupun kenyataannya bukan, 
#dianggap lebih baik, daripada transaksi tersebut tidak terdeteksi sebagai fraud tetapi ternyata fraud. 
#Untuk problem ini sebaiknya menggunakan Recall.
#sebaliknya, jika lebih menginginkan terjadinya True Negatif dan sangat tidak menginginkan terjadinya False Positif, 
#sebaiknya menggunakan Precision.
#Contohnya adalah pada kasus klasifikasi email SPAM atau tidak. 
#Banyak orang lebih memilih jika email yang sebenarnya SPAM namun diprediksi tidak SPAM (sehingga tetap ada pada kotak masuk email kita), 
#daripada email yang sebenarnya bukan SPAM tapi diprediksi SPAM (sehingga tidak ada pada kotak masuk email).

#Pendahuluan
#Setelah pemahaman dengan prosedur machine learning modelling. 
#Selanjutnya materi akan membahas mengenai machine learning algorithm.
#Sebagai dasar, akan dipelajari beberapa algorithm machine learning yaitu Logistic Regression, 
#dan Decision Tree untuk classification problem, 
#dan Linear regression untuk regression problem.

#Classification - Logistic Regression
#Logistic Regression merupakan salah satu algoritma klasifikasi dasar yang cukup popular. 
#Secara sederhana, Logistic regression hampir serupa dengan linear regression tetapi linear regression digunakan untuk Label atau 
#Target Variable yang berupa numerik atau continuous value, 
#sedangkan Logistic regression digunakan untuk Label atau Target yang berupa categorical/discrete value.
#Contoh continuous value adalah harga rumah, harga saham, suhu, dsb; 
#dan contoh dari categorical value adalah prediksi SPAM or NOT SPAM (1 dan 0) atau 
#prediksi customer SUBSCRIBE atau UNSUBSCRIBED (1 dan 0).
#Umumnya Logistic Regression dipakai untuk binary classification (1/0; Yes/No; True/False) problem, 
#tetapi beberapa data scientist juga menggunakannya untuk multiclass classification problem.
#Logistic regression adalah salah satu linear classifier, oleh karena itu,
#Logistik regression juga menggunakan rumus atau fungsi yang sama seperti linear regression yaitu:
f(x)=b0+b1X1+...+brXr
#yang disebut Logit, dimana Variabel 𝑏₀, 𝑏₁, …, 𝑏ᵣ adalah koefisien regresi, dan 𝑥₁, …, 𝑥ᵣ adalah explanatory variable/variabel input atau feature.
#Output dari Logistic Regression adalah 1 atau 0;
#sehingga real value dari fungsi logit ini perlu ditransfer ke nilai di antara 1 dan 0
#dengan menggunakan fungsi sigmoid.
f(x)=1/1+e^-(x)
#Jadi, jika output dari fungsi sigmoid bernilai lebih dari 0.5,
# maka data point diklasifikasi ke dalam label/class: 1 atau YES; dan kurang dari 0.5,
# akan diklasifikasikan ke dalam label/class: 0 atau NO.
#Logistic Regression hanya dapat mengolah data dengan tipe numerik.
# Pada saat preparasi data, pastikan untuk mengecek tipe variabel yang ada dalam dataset dan pastikan semuanya adalah numerik, 
# lakukan data transformasi jika diperlukan.

#Pemodelan Permasalahan Klasifikasi dengan Logistic Regression
#Pemodelan Logistic Regression dengan memanfaatkan Scikit-Learn sangatlah mudah.
# Dengan menggunakan dataset yang sama yaitu online_raw,
# dan setelah dataset dibagi ke dalam Training Set dan Test Set,
# cukup menggunakan modul linear_model dari Scikit-learn,
# dan memanggil fungsi
LogisticRegression() # yang diberi nama logreg.
#Kemudian, model yang sudah ditraining ini
#bisa digunakan untuk memprediksi output/label dari test dataset sekaligus
#mengevaluasi model performance dengan fungsi
score(), confusion_matrix() dan classification_report().
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
# Call the classifier
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg = logreg.fit(X_train, y_train)
#Training Model: Predict
y_pred = logreg.predict(X_test)
#Evaluate Model Performance
print('Training Accuracy :', model.score(X_train,y_train))
print('Testing Accuracy :', model.score(X_test,y_test))
# confusion matrix
print('\nConfusion matrix')
cm = confusion_matrix(y_test,y_pred)
print(cm)
# classification report
print('\nClassification report')
cr = classification_report(y_test, y_pred)
print(cr)

#Classification - Decision Tree
#Decision Tree merupakan salah satu metode klasifikasi yang populer dan banyak diimplementasikan serta mudah diinterpretasi. 
#Decision tree adalah model prediksi dengan struktur pohon atau struktur berhierarki.
#Decision Tree dapat digunakan untuk classification problem dan regression problem.
#Secara sederhana, struktur dari decision tree adalah sebagai berikut:
   1.Decision Node yang merupakan feature/input variabel;
   2.Branch yang ditunjukkan oleh garis hitam berpanah, yang adalah rule/aturan keputusan, dan
   3.Leaf yang merupakan output/hasil.
#Decision Node paling atas dalam decision tree dikenal sebagai akar keputusan,
#atau feature utama yang menjadi asal mula percabangan.
#Jadi, decision tree membagi data ke dalam kelompok atau kelas berdasarkan feature/variable input,
#yang dimulai dari node paling atas (akar),
#dan terus bercabang ke bawah sampai dicapai cabang akhir atau leaf.
#Misalnya ingin memprediksi apakah seseorang yang mengajukan aplikasi kredit/pinjaman,
# layak untuk mendapat pinjaman tersebut atau tidak. Dengan menggunakan decision tree,
# dapat membreak-down kriteria-kriteria pengajuan pinjaman ke dalam hierarki
#Seumpama, orang yang mengajukan berumur lebih dari 40 tahun, dan memiliki rumah, maka aplikasi kreditnya dapat diluluskan, 
# sedangkan jika tidak, maka perlu dicek penghasilan orang tersebut.
# Jika kurang dari 5000, maka permohonan kreditnya akan ditolak.
# Dan jika usia kurang dari 40 tahun, maka selanjutnya dicek jenjang pendidikannya,
# apakah universitas atau secondary.
# Nah, percabangan ini masih bisa berlanjut hingga dicapai percabangan akhir/leaf node.
#Seperti yang sudah dilakukan dalam prosedur pemodelan machine learning,
#selanjutnya dapat dengan mudah melakukan pemodelan decision tree dengan menggunakan scikit-learn module, yaitu 
DecisionTreeClassifier.

#Tugas Praktek
#Dengan menggunakan dataset online_raw.csv dan diasumsikan sudah melakukan EDA dan pre-processing,
#Jadikan atau membuat model machine learning dengan menggunakan decision tree :
 ('''
 1.Import DecisionTreeClassifier dan panggil fungsi tersebut dengan nama decision_tree
 2.Split dataset ke dalam training & testing dataset dengan perbandingan 70:30=0.3, dengan random_state = 0
 3.Latih model dengan training feature (X_train) dan training target (y_train) menggunakan .fit()
 4.Evaluasi hasil model decision_tree yang sudah dilatih dengan testing feature (X_test)
 dan print nilai akurasi dari training dan testing dengan fungsi .score()
 ''')
#JAWAB
#import library
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Call the classifier
decision_tree=DecisionTreeClassifier()
# Fit the classifier to the training data
decision_tree = decision_tree.fit(X_train, y_train)

# evaluating the decision_tree performance
print('Training Accuracy :', decision_tree.score(X_train,y_train))
print('Testing Accuracy :', decision_tree.score(X_test,y_test))

#Regression: Linear Regression - Part 1
#Regression merupakan metode statistik dan machine learning yang paling banyak digunakan.
#Seperti yang dijelaskan sebelumnya,
#regresi digunakan untuk memprediksi output label yang berbentuk numerik atau continuous value.
#Dalam proses training, model regresi akan menggunakan variabel input (features) dan variabel output (label)
#untuk mempelajari bagaimana hubungan/pola dari variabel input dan output.
#Model regresi terdiri atas 2 tipe yaitu :
1.Simple regression model → model regresi paling sederhana, hanya terdiri dari satu feature (univariate) dan 1 target.
2.Multiple regression model → sesuai namanya, terdiri dari lebih dari satu feature (multivariate).
#Adapun model regresi yang paling umum digunakan adalah Linear Regression.

#Regression: Linear Regression - Part 2
#Linear regression digunakan untuk menganalisis hubungan linear antara dependent variabel (feature) dan independent variabel (label).
#Hubungan linear disini berarti bahwa jika nilai dari independen variabel mengalami perubahan baik itu naik atau turun,
#maka nilai dari dependen variabel juga mengalami perubahan (naik atau turun).
#Rumus matematis dari Linear Regression adalah:
y=a+bX #untuk simple linear regression, atau
y=a+b1X1+b2X2+...+biXi #untuk multiple liniear regression , y adalah target/label, X adalah feature, dan a,b adalah model parameter (intercept dan slope).
#Perlu diketahui bahwa tidak semua problem dapat diselesaikan dengan linear regression.
# Untuk pemodelan dengan linear regression, terdapat beberapa asumsi yang harus dipenuhi, yaitu :
1.Terdapat hubungan linear antara variabel input (feature) dan variabel output(label).
Untuk melihat hubungan linear feature dan label, dapat menggunakan chart seperti scatter chart.
Untuk mengetahui hubungan dari variabel umumnya dilakukan pada tahap eksplorasi data.
2.Tidak ada multicollinearity antara features.
Multicollinearity artinya terdapat dependency antara feature,
misalnya saja hanya bisa mengetahui nilai feature B jika nilai feature A sudah diketahui.
3.Tidak ada autocorrelation dalam data, contohnya pada time-series data.
#Pemodelan Linear regression menggunakan scikit-learn tidaklah sulit.
#Secara prosedur serupa dengan pemodelan logistic regression.
#Cukup memanggil LinearRegression dengan terlebih dahulu meng-import fungsi tersebut :
from sklearn.linear_model import LinearRegression
#Setelah memahami konsep dasar dari regression,
# kita akan berlatih membuat model machine learning dengan Linear regression.
# Untuk pemodelan ini kita akan menggunakan data ‘Boston Housing Dataset’.
# Setelah pembelajaran kamu sampai di sini,
# tahu tidak   mengapa kita tidak bisa menggunakan data “online purchase
#karena untuk linear regression target/label harus berupa numerik,
# sedangkan target dari online purchase data adalah categorical.
#Tepat sekali, Senja. Kalau begitu kita bisa lanjut ke pemodelan.
# Tujuan dari pemodelan ini adalah memprediksi harga rumah di Boston berdasarkan feature - feature yang ada. 
# Asumsikan saja bahwa kita sudah melakukan data eksplorasi dan data pre-processing.
# Jadi, data yang akan digunakan adalah data yang siap untuk diproses ke tahap pemodelan.

#Tugas Praktek
('''
 1.Pisahkan dataset ke dalam Feature dan Label, gunakan fungsi .drop().
 Pada dataset ini, label/target adalah variabel MEDV
 2.Checking dan print jumlah data setelah Dataset pisahkan ke dalam Feature dan Label, gunakan .shape()
 3.Bagi dataset ke dalam Training dan test dataset, 70% data digunakan untuk training dan 30% untuk testing, gunakan fungsi train_test_split() , 
 dengan random_state = 0
 4.Checking dan print kembali jumlah data dengan fungsi .shape()
 5.Import LinearRegression dari sklearn.linear_model
 6.Deklarasikan  LinearRegression regressor dengan nama reg
 7.Fit regressor ke training dataset dengan .fit(), dan gunakan .predict()
 untuk memprediksi nilai dari testing dataset.
''')
#jawab
#load dataset
import pandas as pd
housing = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/housing_boston.csv')
#Data rescaling
from sklearn import preprocessing
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
housing[['RM','LSTAT','PTRATIO','MEDV']] = data_scaler.fit_transform(housing[['RM','LSTAT','PTRATIO','MEDV']])
# getting dependent and independent variables
X = housing.drop(['MEDV'], axis = 1)
y = housing['MEDV']
# checking the shapes
print('Shape of X:', X.shape)
print('Shape of y:', y.shape)

# splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# checking the shapes
print('Shape of X_train :', X_train.shape)
print('Shape of y_train :', y_train.shape)
print('Shape of X_test :', X_test.shape)
print('Shape of y_test :', y_test.shape)

##import regressor from Scikit-Learn
from sklearn.linear_model import LinearRegression
# Call the regressor
reg = LinearRegression()
# Fit the regressor to the training data
reg = reg.fit(X_train,y_train)
# Apply the regressor/model to the test data
y_pred = reg.predict(X_test)

#Regression Performance Evaluation
#Untuk model regression, kita menghitung selisih antara nilai aktual (y_test) dan nilai prediksi (y_pred) yang disebut error, adapun beberapa metric yang umum digunakan. 
# Coba kamu ke mari, aku jelaskan langkah-langkahnya
#Mean Squared Error (MSE) adalah rata-rata dari squared error
#Root Mean Squared Error (RMSE) adalah akar kuadrat dari MSE
#Mean Absolute Error (MAE) adalah rata-rata dari nilai absolut error
#Semakin kecil nilai MSE, RMSE, dan MAE, semakin baik pula performansi model regresi.
# Untuk menghitung nilai MSE, RMSE dan MAE dapat dilakukan dengan
# menggunakan fungsi mean_squared_error () ,  mean_absolute_error () dari scikit-learn.metrics dan
# untuk RMSE sendiri tidak terdapat fungsi khusus di scikit-learn tapi dapat dengan mudah
# kita hitung dengan terlebih dahulu menghitung MSE kemudian menggunakan numpy module yaitu, sqrt()
# untuk memperoleh nilai akar kuadrat dari MSE.

#Tugas Praktek
('''
1.Import library yang digunakan: mean_squared_error, mean_absolute_error dari  sklearn.metrics
dan numpy sebagai aliasnya yaitu np. Serta, import juga matplotlib.pyplot sebagai aliasnya, plt.
2.Hitung dan print nilai MSE dan RMSE dengan menggunakan argumen y_test dan y_pred,
untuk rmse gunakan np.sqrt()
3.Buat scatter plot yang menggambarkan hasil prediksi (y_pred) dan harga actual (y_test)
''')

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

#Calculating MSE, lower the value better it is. 0 means perfect prediction
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error of testing set:', mse)
#Calculating MAE
mae = mean_absolute_error(y_test, y_pred)
print('Mean absolute error of testing set:', mae)
#Calculating RMSE
rmse = np.sqrt(mse)
print('Root Mean Squared Error of testing set:', rmse)

#Plotting y_test dan y_pred
plt.scatter(y_test, y_pred, c = 'green')
plt.xlabel('Price Actual')
plt.ylabel('Predicted value')
plt.title('True value vs predicted value : Linear Regression')
plt.show()

#Pendahuluan
#eperti yang sudah dijelaskan sebelumnya, Machine Learning terdiri atas 2 tipe yaitu supervised dan unsupervised learning. 
# Kita telah banyak membahas tentang supervised learning yaitu Klasifikasi model dan Regression Model. 
# Sekarang kita akan mempelajari dasar- dasar terkait unsupervised learning
#Unsupervised Learning adalah teknik machine learning dimana tidak terdapat label atau
# output yang digunakan untuk melatih model.
# Jadi, model dengan sendirinya akan bekerja untuk menemukan pola atau informasi dari dataset yang ada. 
# Metode unsupervised learning yang dikenal dengan clustering.
# Sesuai dengan namanya, Clustering memproses data dan mengelompokkannya atau
# mengcluster objek/sample berdasarkan kesamaan antar objek/sampel dalam satu kluster,
# dan objek/sample ini cukup berbeda dengan objek/sample di kluster yang lain.
#Pada awalnya kita tidak mengetahui bagaimana pola dari objek/sample,
# termasuk juga tidak mengetahui bagaimana kesamaan maupun perbedaan antara
# objek yang satu dengan objek yang lain. Setelah dilakukan clustering,
# baru dapat terlihat bawah objek/sample tersebut dapat dikelompokkan ke dalam 3 kluster.
# Untuk menjelaskan tentang metode Clustering,
# kita akan menggunakan metode clustering yang sangat populer,
# yaitu K-Means Algorithm yang akan kita praktikkan nanti

#K-Means Clustering
#"Jadi, Algorithm K-Means itu apa dan bagaimana cara kerjanya?” tanyaku antusias.
# “K-Means merupakan tipe clustering dengan centroid based (titik pusat).
# Artinya kesamaan dari objek/sampel dihitung dari seberapa dekat objek itu dengan centroid atau titik pusat.”
#Untuk menentukan centroid,
# pada awalnya kita perlu mendefinisikan jumlah centroid (K) yang diinginkan,
# semisalnya kita menetapkan jumlah K = 3; maka pada awal iterasi,
# algorithm akan secara random menentukan 3 centroid.
# Setelah itu, objek/sample/data point yang lain akan dikelompokkan sebagai anggota dari salah satu centroid yang terdekat, 
# sehingga terbentuk 3 cluster data.
#Iterasi selanjutnya, titik-titik centroid diupdate atau berpindah ke titik yang lain,
# dan jarak dari data point yang lain ke centroid yang baru dihitung kembali,
# kemudian dikelompokkan kembali berdasarkan jarak terdekat ke centroid yang baru.
# Iterasi akan terus berlanjut hingga diperoleh cluster dengan error terkecil,
# dan posisi centroid tidak lagi berubah
#Secara prosedur, tahap eksplorasi data untuk memahami karakteristik data,
# dan tahap preprocessing tetap dilakukan. Tetapi dalam unsupervised learning,
# kita tidak membagi dataset ke feature dan label; dan juga ke dalam training dan test dataset,
# karena pada dasarnya kita tidak memiliki informasi mengenai label/target data,

#Tugas Praktek
#“Untuk praktik  ini, kita akan menggunakan dataset ‘Mall Customer Segmentation’,” ujar Senja.
#Aku membaca detail latihan yang sudah ia catatkan untukku:
#Dataset ini merupakan data customer suatu mall dan berisi basic informasi customer berupa :
   1.CustomerID, age, gender, annual income, dan spending score.
   2.Adapun tujuan dari clustering adalah untuk memahami customer - customer mana saja yang sering melakukan transaksi sehingga 
   3.informasi ini dapat diberikan kepada marketing team untuk membuat strategi promosi yang sesuai dengan karakteristik customer.
#“Kita akan melakukan segmentasi customer, dengan memanfaatkan fungsi KMeans dari Scikit-Learn.cluster. 
#Silakan berlatih dengan intruksi di catatan tadi ya
1.Import pandas sebagai aliasnya dan KMeans dari sklearn.cluster.
2.Load dataset 'https://storage.googleapis.com/dqlab-dataset/pythonTutorial/mall_customers.csv' dan beri nama dataset
3.Diasumsikan EDA dan preprocessing sudah dilakukan,
selanjutnya kita memilih feature yang akan digunakan untuk membuat model yaitu annual_income dan spending_score. 
Assign dataset dengan feature yang sudah dipilih ke dalam 'X'.
Pada dasarnya terdapat teknik khusus yang dilakukan untuk menyeleksi feature - feature (Feature Selection) mana saja 
yang dapat digunakan untuk machine learning modelling,
karena tidak semua feature itu berguna. Beberapa feature justru bisa menyebabkan performansi model menurun. 
Tetapi untuk problem ini, secara default kita akan menggunakan annual_income dan spending_score.
4.Deklarasikan  KMeans( )  dengan nama cluster_model dan gunakan n_cluster = 5.
n_cluster adalah argumen dari fungsi KMeans( ) yang merupakan jumlah cluster/centroid (K).
random_state = 24.
5.Gunakan fungsi .fit_predict( ) dari cluster_model pada 'X'  untuk proses clustering.

#jawab
#import library
import pandas as pd
from sklearn.cluster import KMeans

#load dataset
dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/mall_customers.csv')

#selecting features
X = dataset[['annual_income','spending_score']]

#Define KMeans as cluster_model
cluster_model = KMeans(n_clusters = 5, random_state = 24)
labels = cluster_model.fit_predict(X)

#Tugas Praktek
#Inspect & Visualizing the Cluster
#“Satu lagi,kalau sudah membuat cluster, tolong  visualisasikan hasil dari clustering yang telah kamu lakukan sebelumnya ya. 
#Langkah-langkahnya sudah saya email,”
1.Pertama - tama, import matplotlib.pyplot dan beri inisial plt.
2.Gunakan fungsi .values untuk mengubah tipe ‘X’ dari dataframe menjadi array
3.Pisahkan X kedalam xs dan ys, di mana xs adalah Kolom index [0] dan ys adalah kolom index [1]
4.Buatlah scatter plot plt.scatter() dari xs dan ys, kemudian tambahkan c = labels untuk secara otomatis memberikan warna yang berbeda pada setiap cluster, dan alpha = 0.5 ke dalam scatter plot argumen.
5.Hitunglah koordinat dari centroid menggunakan .cluster_centers_ dari cluster_model, deklarasikan ke dalam variabel centroids.
6.Pisahkan centroids kedalam centroids_x dan centroids_y, di mana centroids_x adalah kolom index [0] dan centroids_y adalah kolom index [1]
7.Buatlah scatter plot dari centroids_x dan centroids_y , gunakan ‘D’ (diamond) sebagai marker parameter, dengan ukuran 50, s = 50

#hasil
#import library
import matplotlib.pyplot as plt

#convert dataframe to array
X = X.values
#Separate X to xs and ys --> use for chart axis
xs = X[:,0]
ys = X[:,1]
# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = cluster_model.cluster_centers_
# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,marker='D', s=50)
plt.title('KMeans Clustering', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

#Measuring Cluster Criteria
#bagaimana kita tahu bahwa membagi segmentasi ke dalam 5 cluster adalah segmentasi yang paling optimal? 
# Karena jika dilihat pada gambar beberapa data point masih cukup jauh jaraknya dengan centroidnya.
#Clustering yang baik adalah cluster yang data point-nya saling rapat/sangat berdekatan satu sama lain dan cukup berjauhan dengan objek/data point di cluster yang lain. Jadi, objek dalam satu cluster tidak tersebut berjauhan. 
# Nah, untuk mengukur kualitas dari clustering, kita bisa menggunakan inertia,
#Inertia sendiri mengukur seberapa besar penyebaran object/data point data dalam satu cluster, semakin kecil nilai inertia maka semakin baik. Kita tidak perlu bersusah payah menghitung nilai inertia karena secara otomatis, telah dihitung oleh KMeans( ) ketika algorithm di fit ke dataset.
# Untuk mengecek nilai inertia cukup dengan print fungsi .inertia_ dari model yang sudah di fit ke dataset
#Kalau begitu,   bagaimana caranya mengetahui nilai K yang paling baik dengan inertia yang paling kecil? 
# Apakah harus trial Error dengan mencoba berbagai jumlah cluster?
#Benar, kita perlu mencoba beberapa nilai, dan memplot nilai inertia-nya. 
# Semakin banyak cluster maka inertia semakin kecil.
#Meskipun suatu clustering dikatakan baik jika memiliki inertia yang kecil tetapi secara praktikal in real life, terlalu banyak cluster juga tidak diinginkan. 
# Adapun rule untuk memilih jumlah cluster yang optimal adalah dengan memilih jumlah cluster yang terletak pada “elbow” dalam intertia plot, yaitu ketika nilai inertia mulai menurun secara perlahan. 
# Jika dilihat pada gambar maka jumlah cluster yang optimal adalah K = 3

#Tugas Praktek
#Coba kamu membuat inertia plot untuk melihat apakah K = 5 merupakan jumlah cluster yang optimal. 
#Untuk membuat inertia plot, silakan memanfaatkan fungsi looping (for):

1.Pertama - tama, buatlah sebuah list kosong yang dinamakan 'inertia'. List ini akan kita gunakan untuk menyimpan nilai inertia dari setiap nilai K.
2.Gunakan for untuk membuat looping dengan range 1-10. Sebagai index looping gunakan k
3.Di dalam fungsi looping, deklarasikan  KMeans()  dengan nama cluster_model dan gunakan n_cluster = k, dan random_state = 24
4.Gunakan fungsi .fit() dari cluster_model pada 'X'
5.Dari dari cluster_model yang sudah di-fit ke dataset, dapatkan nilai inertia menggunakan inertia_ dan deklarasikan sebagai inertia_value
6.Append inertia_value ke dalam list 'inertia'
7.Setelah iterasi/looping selesai plotlah list 'inertia' tadi sebagai ordinat-nya dan absica-nya adalah range(1, 10).
#maka,
#import library
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Elbow Method - Inertia plot
inertia = []
#looping the inertia calculation for each k
for K in range(1, 10):
    #Assign KMeans as cluster_model
    cluster_model = KMeans(n_clusters = k, random_state = 24)
    #Fit cluster_model to X
    cluster_model.fit(X)
    #Get the inertia value
    inertia_value = cluster_model.inertia_
    #Append the inertia_value to inertia list
    inertia.append(inertia_value)
    
##Inertia plot
plt.plot(range(1, 10), inertia)
plt.title('The Elbow Method - Inertia plot', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('inertia')
plt.show()

#Case Study: Promos for our e-commerce - Part 1
#buatkan machine learning model untuk menyelesaikan permasalahan dari e-commerce divisi kantor.
#Adapun feature - feature dalam dataset ini adalah :
1.'Daily Time Spent on Site' : lama waktu user mengunjungi site (menit)
2.'Age' : usia user (tahun)
3.'Area Income' : rata - rata pendapatan di daerah sekitar user
4.'Daily Internet Usage' : rata - rata waktu yang dihabiskan user di internet dalam sehari (menit)
5.'Ad Topic Line' : topik/konten dari promo banner
5.'City' : kota dimana user mengakses website
7.'Male' : apakah user adalah Pria atau bukan
8.'Country' : negara dimana user mengakses website
9.'Timestamp' : waktu saat user mengklik promo banner atau keluar dari halaman website tanpa mengklik banner
10.'Clicked on Ad' : mengindikasikan user mengklik promo banner atau tidak (0 = tidak; 1 = klik).

#import library
import pandas as pd

# Baca data 'ecommerce_banner_promo.csv'
data = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/ecommerce_banner_promo.csv')

#1. Data eksplorasi dengan head(), info(), describe(), shape
print("\n[1] Data eksplorasi dengan head(), info(), describe(), shape")
print("Lima data teratas:")
print(data.head())
print("Informasi dataset:")
print(data.info())
print("Statistik deskriptif dataset:")
print(data.describe())
print("Ukuran dataset:")
print(data.shape)

#Case Study: Promos for our e-commerce - Part 2
#Sekarang mari melanjutkan dengan ekplorasi data untuk langkah ke-2 dan ke-3:
2.Data eksplorasi dengan dengan mengecek korelasi dari setiap feature menggunakan fungsi corr()
3.Data eksplorasi dengan mengecek distribusi label menggunakan fungsi groupby() dan size()
#2. Data eksplorasi dengan dengan mengecek korelasi dari setiap feature menggunakan fungsi corr()
print("\n[2] Data eksplorasi dengan dengan mengecek korelasi dari setiap feature menggunakan fungsi corr()")
print(data.corr())

#3. Data eksplorasi dengan mengecek distribusi label menggunakan fungsi groupby() dan size()
print("\n[3] Data eksplorasi dengan mengecek distribusi label menggunakan fungsi groupby() dan size()")
print(data.groupby('Clicked on Ad').size())

#Case Study: Promos for our e-commerce - Part 3
#Di proyek ini, aku akan melanjutkan mengeksplorasi data dengan visualisasi dengan tahap - tahap yang perlu dilakukan adalah (langkah ke-4):
4.Data eksplorasi dengan visualisasi:
-Jumlah user dibagi ke dalam rentang usia menggunakan histogram (hist()), 
gunakan bins = data.Age.nunique() sebagai argumen. nunique() adalah fungsi untuk menghitung jumlah data untuk setiap usia (Age).
-Gunakan pairplot() dari seaborn modul untuk menggambarkan hubungan setiap feature. 

#import library
import matplotlib.pyplot as plt
import seaborn as sns

# Seting: matplotlib and seaborn
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

#4. Data eksplorasi dengan visualisasi
#4a. Visualisasi Jumlah user dibagi ke dalam rentang usia (Age) menggunakan histogram (hist()) plot
plt.figure(figsize=(10, 5))
plt.hist(data['Age'], bins = data.Age.nunique())
plt.xlabel('Age')
plt.tight_layout()
plt.show()

#4b. Gunakan pairplot() dari seaborn (sns) modul untuk menggambarkan hubungan setiap feature.
plt.figure()
sns.pairplot(data)
plt.show()

#Case Study: Promos for our e-commerce - Part 4
#Di bagian proyek (langkah ke-5) ini aku akan mengecek apakah terdapat missing value dari data, 
# jika terdapat missing value dapat dilakukan treatment seperti didrop atau diimputasi dan jika tidak maka dapat melanjutkan ke langkah berikutnya.
5.Cek missing value
#5. Cek missing value
print("\n[5] Cek missing value")
print(data.isnull().sum().sum())

#Case Study: Promos for our e-commerce - Part 5
#Pada langkah ke-6 ini aku akan melakukan pemodelan dengan Logistic Regression dengan cara seperti berikut:
6.Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing :
-Deklarasikan data ke dalam X dengan mendrop feature/variabel yang bukan numerik, (type = object) dari data (Logistic Regression hanya dapat memproses numerik variabel). Assign Target/Label feature dan assign sebagai y
-Split X dan y ke dalam training dan testing dataset, gunakan perbandingan 80:20 dan random_state = 42
-Assign classifier sebagai logreg, kemudian fit classifier ke X_train dan predict dengan X_test. Print evaluation score.
#import library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#6.Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing
print("\n[6] Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing")
#6a.Drop Non-Numerical (object type) feature from X, as Logistic Regression can only take numbers, and also drop Target/label, assign Target Variable to y.   
X = data.drop(['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'], axis = 1)
y = data['Clicked on Ad']

#6b. splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

#6c. Modelling
# Call the classifier
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg = logreg.fit(X_train,y_train)
# Prediksi model
y_pred = logreg.predict(X_test)

#6d. Evaluasi Model Performance
print("Evaluasi Model Performance:")
print("Training Accuracy :", logreg.score(X_train,y_train))
print("Testing Accuracy :", logreg.score(X_test,y_test))

#Case Study: Promos for our e-commerce - Part 6
#Di langkah terakhir ini atau langkah ke-7 aku akan melihat performansi model dengan menggunakan confusion matrix dan classification report.
7.Print Confusion matrix dan classification report
# Import library
from sklearn.metrics import confusion_matrix, classification_report

#7. Print Confusion matrix dan classification report
print("\n[7] Print Confusion matrix dan classification report")

#apply confusion_matrix function to y_test and y_pred
print("Confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

#apply classification_report function to y_test and y_pred
print("Classification report:")
cr = classification_report(y_test,y_pred)
print(cr)

#Berdasarkan hasil evaluasi, apakah model yang dibuat cukup baik untuk memprediksi user yang akan mengklik website atau tidak?
#Metrik evaluasi apa yang tepat digunakan untuk mengevaluasi performansi dari model yang telah dilakukan training?
Model sudah sangat baik dalam memprediksi user yang akan mengklik website atau tidak, 
dapat dilihat dari nilai accuracy = 0.90; 
Dataset memiliki jumlah label yang seimbang (balance class), 
sehingga evaluasi performansi dapat menggunakan metrik Accuracy.

#Penutup/Kesimpulan
#Congratulations! Akhirnya selesai satu lagi modul Machine Learning With Python for Beginner. Berdasarkan materi-materi yang telah kupelajari dan praktekkan dalam modul ini, aku telah mendapatkan pengetahuan (knowledge) dan praktek (skill) yang diantaranya
1.Memahami apa itu machine learning dengan jenisnya untuk pemodelan
2.Memahami dan mampu melakukan Eksplorasi Data & Data Pre-processing
3.Memahami dan mampu melakukan proses-proses Pemodelan dengan Scikit-Learn
4.Memahami dan mampu melakukan proses-proses pemodelan dengan menggunakan algoritma pada Supervised Learning
5.Memahami dan mampu melakukan proses-proses pemodelan dengan menggunakan algoritma pada Unsupervised Learning
6.Mengerjakan mini project yang merupakan integrasi keseluruhan materi dan tentunya materi-materi pada modul-modul sebelumnya untuk menyelesaikan persolan bisnis.an Pandas library. 
#Pandas cukup powerful untuk digunakan dalam menganalisa, memanipulasi dan membersihkan data. Siap, Aksara?”
#Aku mengangguk, siap dengan laptopku.
#“Oke, Pertama- tama,  kita check dimensi data kita terlebih dahulu. Aksara, silahkan load datanya dan gunakan .shape, .head(), .info(), dan .describe() untuk mengeksplorasi dataset secara berurut. Dataset ini adalah data pembeli online yang mengunjungi website dari suatu e-commerce selama setahun, yaitu 'https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv',” perintah Senja.
import pandas as pd
dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')
print('Shape dataset:', dataset.shape)
print('\nLima data teratas:\n',dataset.head())
print('\nInformasi dataset:')
print(dataset.info())
print('\nStatistik deskriptif:\n',dataset.describe())
#“Nah, dengan mengetahui dimensi data yaitu jumlah baris dan kolom, kita bisa mengetahui apakah data kita terlalu banyak atau justru sangat sedikit.
# Jika data terlalu banyak, waktu melatih model akan lebih lama, sedangkan jika data terlalu sedikit, performansi model yang kita hasilkan mungkin tidak cukup bagus, 
# karena tidak mampu mengenali pola dengan baik. Sudah lebih paham sekarang?”
#Aku mengacungkan jempol. Kalau dibareng praktik, memang jadinya lebih jelas.
#Berdasarkan praktek yang telah dilakukan pada bagian sebelumnya, pernyataan manakah yang tidak sesuai berdasarkan hasil eksplorasi?
Nilai rata - rata (mean) dari feature BounceRates adalah 0.2

#Eksplorasi Data: Memahami Data dengan Statistik - Part 2
#Setelah selesai materi tadi, aku kembali diajak memahami eksplorasi data. 
#Banyak sekali materi baru hari ini!
#“Kita lanjut yah, Aksara. 
#Data eksplorasi tidaklah cukup dengan mengetahui dimensi data dan statistical properties saja, 
#tetapi kita juga perlu sedikit menggali tentang hubungan atau korelasi dari setiap feature, 
#karena beberapa algorithm seperti linear regression dan logistic regression akan menghasilkan model dengan performansi yang buruk jika
#kita menggunakan feature/variabel saling dependensi atau berkorelasi kuat (multicollinearity).
#Jadi, jika kita sudah tahu bahwa data kita berkorelasi kuat, kita bisa menggunakan algorithm lain yang tidak sensitif terhadap hubungan korelasi dari feature/variabel seperti decision tree.”
#”Aksara, coba sekarang lanjutkan eksplorasi data untuk melihat korelasi dan distribusi dataset.”
#Aku segera berkutat dengan susunan kodenya:
dataset_corr = dataset.corr()
print('Korelasi dataset:\n', dataset.corr())
print('Distribusi Label (Revenue):\n', dataset['Revenue'].value_counts())
#kenapa mengetahui distribusi LABEL dari dataset itu penting?”
#“Pertanyaan menarik, mengetahui distribusi label sangat penting untuk permasalahan klasifikasi, 
#karena jika distribusi label sangat tidak seimbang (imbalanced class),  
#maka akan sulit bagi model untuk mempelajari pola dari LABEL yang sedikit dan hasilnya bisa misleading. 
#Contohnya, kita memiliki 100 row data, 90 row adalah non fraud dan 10 row adalah fraud. 
#Jika kita menggunakan data ini tanpa melakukan treatment khusus (handling imbalanced class), 
#maka kemungkinan besar model kita akan cenderung mengenali observasi baru sebagai non-fraud, 
#dan hal ini tentunya tidak diinginkan,” jelas Senja panjang lebar.

#Tugas Praktek:
#Sekarang coba inspeksi nilai korelasi dari fitur-fitur berikut pada dataset_corr yang telah diberikan sebelumnya
#1. ExitRates dan BounceRates
#2. Revenue dan PageValues
#3. TrafficType dan Weekend
#Jika dengan benar ditulis dan dijalankan maka output seperti berikut ini yang akan diperoleh:
dataset_corr = dataset.corr()
print('Korelasi dataset:\n', dataset.corr())
print('Distribusi Label (Revenue):\n', dataset['Revenue'].value_counts())
# Tugas praktek
print('\nKorelasi BounceRates-ExitRates:', dataset_corr.loc['BounceRates', 'ExitRates'])
print('\nKorelasi Revenue-PageValues:', dataset_corr.loc['Revenue', 'PageValues'])
print('\nKorelasi TrafficType-Weekend:', dataset_corr.loc['TrafficType', 'Weekend'])

#Berdasarkan hasil eksplorasi, pilihlah pernyataan berikut ini yang tidak sesuai dengan hasil eksplorasi adalah 
Distribusi label dari dataset tidak seimbang karena total data point dengan label 1 adalah 10422 dan total data point dengan label 0 adalah 1908.

#Eksplorasi Data: Memahami Data dengan Visual
#“Aksara, satu lagi, dalam mengeksplorasi data, kita perlu untuk memahami data dengan visual.”
#Aku tertarik, “Maksudnya?”
#“Begini, selain dengan statistik, kita juga bisa melakukan eksplorasi data dalam bentuk visual. 
#Dengan visualisasi kita dapat dengan mudah dan cepat dalam memahami data, 
#bahkan dapat memberikan pemahaman yang lebih baik terkait hubungan setiap variabel/ features.
#Misalnya kita ingin melihat distribusi label dalam bentuk visual, dan jumlah pembelian saat weekend. 
#Kita dapat memanfaatkan matplotlib library untuk membuat chart yang menampilkan perbandingan jumlah yang membeli (1) dan tidak membeli (0), 
#serta perbandingan jumlah pembelian saat weekend,” tambah Senja sembari menampilkan contoh kodenya:
import matplotlib.pyplot as plt
import seaborn as sns
# checking the Distribution of customers on Revenue
plt.rcParams['figure.figsize']=(12,5)
plt.subplot(1,2,1)
sns.countplot(dataset['Revenue'],palette='pastel')
plt.title('Buy or not', fontsize=20)
plt.xlabel('Revenue or not', fontsize=14)
plt.ylabel('count', fontsize=14)
# checking the Distribution of customers on Weekend
plt.subplot(1,2,2)
sns.countplot(dataset['Weekend'],palette='inferno')
plt.title('Purchase on Weekends', fontsize=20)
plt.xlabel('Weekend or not', fontsize=14)
plt.ylabel('count', fontsize=14)
plt.show()

#Tugas Praktek
#Aku kemudian diminta Senja untuk membuat visualisasi berupa histogram yang menggambarkan jumlah customer untuk setiap Region.
#Dalam membuat visualisasi ini aku akan menggunakan dataset['region'] untuk membuat histogram,
#dan berikan judul 'Distribution of Customers' pada title, 
#'Region Codes' sebagai label axis-x dan 
#'Count Users' sebagai label axis-y.
import matplotlib.pyplot as plt
# visualizing the distribution of customers around the Region
plt.hist(dataset['Region'], color = 'lightblue')
plt.title('Distribution of Customers', fontsize = 20)
plt.xlabel('Region Codes', fontsize = 14)
plt.ylabel('Count Users', fontsize = 14)
plt.show()

#Data Pre-processing: Handling Missing Value - Part 1
#Setelah kita melakukan eksplorasi data, kita akan melanjutkan ke tahap data pre-processing. 
# Seperti yang saya jelaskan sebelumnya, raw data kita belum tentu bisa langsung digunakan untuk pemodelan. 
# Jika kita memiliki banyak missing value, maka akan mengurangi performansi model dan juga beberapa algorithm machine learning tidak dapat memproses data dengan missing value. 
# Oleh karena itu, kita perlu mengecek apakah terdapat missing value dalam data atau tidak. 
# Jika tidak, maka kita tidak perlu melakukan apa-apa dan bisa melanjutkan ke tahap berikutnya. 
# Jika ada, maka kita perlu melakukan treatment khusus untuk missing value ini
#Pengecekan missing value dapat dilakukan dengan code berikut
#menggunakan metod .isnull pada dataset dan kemudian men-chaining-nya dengan method sum. 
# Untuk jumlah keseluruhan missing value digunakan chaining method sum sekali lagi. 
#checking missing value for each feature  
print('Checking missing value for each feature:')
print(dataset.isnull().sum())
#Counting total missing value
print('\nCounting total missing value:')
print(dataset.isnull().sum().sum())

#Data Pre-processing: Handling Missing Value - Part 2
#Metode ini dapat diterapkan jika tidak banyak missing value dalam data, 
# sehingga walaupun data point ini dihapus, 
# kita masih memiliki sejumlah data yang cukup untuk melatih model Machine Learning. 
# Tetapi jika kita memiliki banyak missing value dan tersebar di setiap variabel, 
# maka metode menghapus missing value tidak dapat digunakan. 
# Kita akan kehilangan sejumlah data yang tentunya mempengaruhi performansi model. 
# Kita bisa menghapus data point yang memiliki missing value dengan fungsi .dropna( ) dari pandas library. 
# Fungsi dropna( ) akan menghapus data point atau baris yang memiliki missing value.
#Drop rows with missing value
dataset_clean = dataset.dropna()
print('Ukuran dataset_clean:', dataset_clean.shape)

#Data Pre-processing: Handling Missing Value - Part 3
#“Kalau tidak dihapus, ada metode lain yang bisa dipakai?”
#Kita bisa menggunakan metode impute missing value, yaitu mengisi record yang hilang ini dengan suatu nilai. 
#Ada berbagai teknik dalam metode imputing, mulai dari yang paling sederhana yaitu mengisi missing value dengan nilai mean, median, modus, atau nilai konstan, 
#sampai teknik paling advance yaitu dengan menggunakan nilai yang diestimasi oleh suatu predictive model. 
#Untuk kasus ini, kita akan menggunakan imputing sederhana yaitu menggunakan nilai rataan atau mean
print("Before imputation:")
# Checking missing value for each feature
print(dataset.isnull().sum())
# Counting total missing value
print(dataset.isnull().sum().sum())

print("\nAfter imputation:")
# Fill missing value with mean of feature value  
dataset.fillna(dataset.mean(), inplace = True)
# Checking missing value for each feature  
print(dataset.isnull().sum())
# Counting total missing value  
print(dataset.isnull().sum().sum())

#Tugas Praktek
#Praktekkan metode imputing missing value dengan menggunakan nilai median.
import pandas as pd
dataset1 = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')

print("Before imputation:")
# Checking missing value for each feature  
print(dataset1.isnull().sum())
# Counting total missing value  
print(dataset1.isnull().sum().sum())

print("\nAfter imputation:")
# Fill missing value with median of feature value  
dataset1.fillna(dataset1.median(), inplace = True)
# Checking missing value for each feature  
print(dataset1.isnull().sum())
# Counting total missing value  
print(dataset1.isnull().sum().sum())

#Data Preprocessing: Scaling
#Setelah berhasil menangani missing value, sekarang mari kita mempelajari tahapan preprocessing selanjutnya. 
# Aksara, tolong tampilkan kembali 5 dataset teratas dan deskripsi statistik dari dataset. 
# Coba perhatikan, rentang nilai dari setiap feature cukup bervariasi. 
# Misalnya, ProductRelated_Duration vs BounceRates.  
# ProductRelated_Duration memiliki rentang nilai mulai dari 0 - 5000; 
# sedangkan BounceRates rentang nilainya 0 - 1.
#Beberapa machine learning seperti K-NN dan gradient descent mengharuskan semua variabel memiliki rentang nilai yang sama, karena jika tidak sama,
# feature dengan rentang nilai terbesar misalnya ProductRelated_Duration otomatis akan menjadi feature yang paling mendominasi dalam proses training/komputasi, 
# sehingga model yang dihasilkan pun akan sangat bias. 
# Oleh karena itu, sebelum memulai training model, kita terlebih dahulu perlu melakukan data rescaling ke dalam rentang 0 dan 1, 
# sehingga semua feature berada dalam rentang nilai tersebut, yaitu nilai max = 1 dan nilai min = 0.
# Data rescaling ini dengan mudah dapat dilakukan di Python menggunakan .MinMaxScaler( ) dari Scikit-Learn library.
#Kenapa ke range 0 - 1, tidak menggunakan range yang lain?
#Karena rumus dari rescaling adalah
Zi=Xi-min(X)/max(x)-min(x)
#dengan rumus ini, nilai max data akan menjadi 1 dan nilai min menjadi 0; 
# dan nilai lainnya berada di rentang keduanya. 
# Rumus ini tidak memungkinkan adanya rentang nilai selain 0 – 1

#Tugas Praktek
from sklearn.preprocessing import MinMaxScaler  
#Define MinMaxScaler as scaler  
scaler = MinMaxScaler()
#Apply fit_transfrom to scale selected feature  
dataset[scaling_column] = scaler.fit_transform(dataset[scaling_column])
#code diatas merupakan basic code untuk proses scaling dengan asumsi bahwa semua feature adalah numerik. 
# Tetapi, ketika menjalankan code tersebut untuk dataset online_raw, pasti akan terjadi error. 
# Proses scaling hanya bisa dilakukan untuk feature dengan tipe numerik, 
# sedangkan dalam dataset online_raw, terdapat feature dengan tipe string atau karakter dan categorical, 
# seperti Month, VisitorType, Region. Oleh karena itu, kita tidak dapat langsung menggunakan code di atas, 
# tetapi kita perlu terlebih dahulu menyeleksi feature - feature dari dataset yang bertipe numerik.
from sklearn.preprocessing import MinMaxScaler  
#Define MinMaxScaler as scaler  
scaler = MinMaxScaler()  
#list all the feature that need to be scaled  
scaling_column = ['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues']
#Apply fit_transfrom to scale selected feature  
dataset[scaling_column] = scaler.fit_transform(dataset[scaling_column])
#Cheking min and max value of the scaling_column
print(dataset[scaling_column].describe().T[['min','max']])

#Data Pre-processing: Konversi string ke numerik
#kita memiliki dua kolom yang bertipe object yang dinyatakan dalam tipe data str, 
# yaitu kolom 'Month' dan 'VisitorType'. 
# Karena setiap algoritma machine learning bekerja dengan menggunakan nilai numeris, 
# maka kita perlu mengubah kolom dengan tipe pandas object atau str ini ke bertipe numeris. 
# Untuk itu, kita list terlebih dahulu apa saja label unik di kedua kolom ini
#bagaimana ya cara menrubah tipe pandas object ini ke numerik (int, float) ya?
#kita dapat menggunakan LabelEncoder dari sklearn.preprocessing untuk merubah kedua kolom ini seperti ini
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Convert feature/column 'Month'
LE=LabelEncoder()
dataset['Month']=LE.fit_transform(dataset['Month'])
print(LE.classes_)
print(np.sort(dataset['Month'].unique()))
print('')

# Convert feature/column 'VisitorType'
LE=LabelEncoder()
dataset['VisitorType']=LE.fit_transform(dataset['VisitorType'])
print(LE.classes_)
print(np.sort(dataset['VisitorType'].unique()))
#Bisa dilihat, kan, Aksara bahwa LabelEncoder akan mengurutkan label secara otomatis secara alfabetik, 
# posisi/indeks dari setiap label ini digunakan sebagai nilai numeris konversi pandas objek ke numeris 
# (dalam hal ini tipe data int). 
# Dengan demikian kita telah membuat dataset kita menjadi dataset bernilai numeris seluruhnya yang siap digunakan untuk pemodelan dengan algoritma machine learning tertentu

#akhirnya, setelah data eksplorasi dan preprocessing, 
# datasetnya sudah siap untuk digunakan dalam proses modelling. 
# Kalau dipikir-pikir, preprocessing ini panjang juga dan ribet, aku lebih senang langsung modelling aja,
# ” komentarku. ”Sebenarnya tidak panjang, 
# ini karena kamu  masih tahap belajar sehingga kamu  perlu mengerti konsepnya dan tidak asal membuat model, 
# tetapi ketika kamu sudah  mulai implementasi proses ini akan otomatis dilakukan sehingga tidak terasa panjang lagi.
#lanjutkan. Pertama-tama saya akan mengenalkan kamu pada library Scikit - Learn. 
# Scikit-learn adalah library untuk machine learning bagi para pengguna python yang memungkinkan kita melakukan berbagai pekerjaan dalam Data Science, 
# seperti regresi (regression), klasifikasi (classification), pengelompokkan/penggugusan (clustering), 
# data preprocessing, dimensionality reduction, dan model selection (pembandingan, validasi, 
# dan pemilihan parameter maupun model)
#Ada beberapa library machine learning di Python seperti Keras, 
# tetapi Scikit - Learn adalah yang paling basic sehingga jika kita menguasai scikit-learn, 
# kita dapat dengan mudah mempelajari library machine learning yang lain.

#Features & Label
#Dalam dataset user online purchase, label target sudah diketahui, 
#yaitu kolom Revenue yang bernilai 1 untuk user yang membeli dan 0 untuk yang tidak membeli, 
#sehingga pemodelan yang dilakukan ini adalah klasifikasi. 
#Nah, untuk melatih dataset menggunakan Scikit-Learn library, dataset perlu dipisahkan ke dalam Features dan Label/Target. 
#Variabel Feature akan terdiri dari variabel yang dideklarasikan sebagai X dan [Revenue] adalah variabel Target yang dideklarasikan sebagai y. 
#Gunakan fungsi drop() untuk menghapus kolom [Revenue] dari dataset.
# removing the target column Revenue from dataset and assigning to X
x=dataset.drop(['Revenue'],axis=1)
# assigning the target column Revenue to y
y=dataset['Revenue']
# checking the shapes
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

#Training dan Test Dataset
#ebelum kita melatih model dengan suatu algorithm machine , 
# seperti yang saya jelaskan sebelumnya, 
# dataset perlu kita bagi ke dalam training dataset dan test dataset dengan perbandingan 80:20. 80% 
# digunakan untuk training dan 20% untuk proses testing.
#Perbandingan lain yang biasanya digunakan adalah 75:25. 
# Hal penting yang perlu diketahui adalah scikit-learn tidak dapat memproses dataframe dan hanya mengakomodasi format data tipe Array. 
# Tetapi kalian tidak perlu khawatir, fungsi train_test_split( ) dari Scikit-Learn, 
# otomatis mengubah dataset dari dataframe ke dalam format array.
#Fungsi Training adalah melatih model untuk mengenali pola dalam data, 
# sedangkan testing berfungsi untuk memastikan bahwa model yang telah dilatih tersebut mampu dengan baik memprediksi label dari new observation dan belum dipelajari oleh model sebelumnya. 
# Lebih baik kita praktik saja ya, tampaknya kalau praktik kamu lebih paham
#silahkan bagi dataset ke dalam Training dan Testing dengan melanjutkan coding yang  sudah kukerjakan ini. 
# Gunakan test_size = 0.2 dan tambahkan argumen random_state = 0,  
# pada fungsi train_test_split( ).
from sklearn.model_selection import train_test_split
# splitting the X, and y
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# checking the shapes
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

#Training Model: Fit
#Sekarang saatnya kita melatih model atau training. 
# Dengan Scikit-Learn, proses ini menjadi sangat sederhana. 
# Kita cukup memanggil nama algorithm yang akan kita gunakan, 
# biasanya disebut classifier untuk problem klasifikasi, 
# dan regressor untuk problem regresi.
#sebagai contoh, kita akan menggunakan Decision Tree. 
# Kita hanya perlu memanggil fungsi DecisionTreeClassifier() yang kita namakan “model”. 
# Kemudian menggunakan fungsi .fit() 
# dan X_train, y_train untuk melatih classifier tersebut dengan training dataset, seperti ini:
from sklearn.tree import DecisionTreeClassifier
# Call the classifier
model=DecisionTreeClassifier()
# Fit the classifier to the training data
model=model.fit(X_train, y_train)

#Training Model: Predict
#setelah model/classifier terbentuk, 
# selanjutnya kita menggunakan model ini untuk memprediksi LABEL dari testing dataset (X_test), 
# menggunakan fungsi .predict(). 
# Fungsi ini akan mengembalikan hasil prediksi untuk setiap data point dari X_test dalam bentuk array. 
# Proses ini kita kenal dengan TESTING
#melanjutkan proses testing menggunakan fungsi .predict() seperti ini
# Apply the classifier/model to the test data
y_pred=model.predict(X_test)
print(y_pred.shape)

#Evaluasi Model Performance - Part 1
#sekarang kita melanjutkan di tahap terakhir dari modelling yaitu evaluasi hasil model. 
# Untuk evaluasi model performance, setiap algorithm mempunyai metrik yang berbeda-beda. 
# Sekarang saya akan menjelaskan sedikit metrik apa saja yang umumnya digunakan. 
# Metrik paling sederhana untuk mengecek performansi model adalah accuracy.
#Kita bisa munculkan dengan fungsi .score( ). 
# Tetapi, di banyak real problem, accuracy saja tidaklah cukup. 
# Metode lain yang digunakan adalah dengan Confusion Matrix.
# Confusion Matrix merepresentasikan perbandingan prediksi dan real LABEL dari test dataset yang dihasilkan oleh algoritma ML,” 
# tukas Senja sambil membuka  template dari confusion Matrix untukku:
True Positive (TP): Jika user diprediksi (Positif) membeli ([Revenue] = 1]), dan memang benar(True) membeli.
True Negative (TN): Jika user diprediksi tidak (Negatif) membeli dan aktualnya user tersebut memang (True) membeli.
False Positive (FP): Jika user diprediksi Positif membeli, tetapi ternyata tidak membeli (False).
False Negatif (FN): Jika user diprediksi tidak membeli (Negatif), tetapi ternyata sebenarnya membeli.

#Evaluasi Model Performance - Part 2
#Untuk menampilkan confusion matrix cukup menggunakan fungsi confusion_matrix() dari Scikit-Learn
from sklearn.metrics import confusion_matrix, classification_report
# evaluating the model
print('Training Accuracy :', model.score(X_train, y_train))
print('Testing Accuracy :', model.score(X_test, y_test))
# confusion matrix
print('\nConfusion matrix:')
cm=confusion_matrix(y_test, y_pred)
print(cm)\
#Berdasarkan confusion matrix, dapat mengukur metrik - metrik berikut :
Accuracy = (TP + TN ) / (TP+FP+FN+TN)
Precision = (TP) / (TP+FP)
Recall = (TP) / (TP + FN)
F1 Score = 2 * (Recall*Precission) / (Recall + Precission)
#Tidak perlu menghitung nilai ini secara manual. 
#Cukup gunakan  fungsi classification_report() untuk memunculkan hasil perhitungan metrik - metrik tersebut.
# classification report
print('\nClassification report:')
cr=classification_report(y_test,y_pred)
print(cr)

#Pakai Metrik yang Mana?
#Jika dataset memiliki jumlah data False Negatif dan False Positif yang seimbang (Symmetric), 
#maka bisa gunakan Accuracy, tetapi jika tidak seimbang, maka sebaiknya menggunakan F1-Score.
#Dalam suatu problem, jika lebih memilih False Positif lebih baik terjadi daripada False Negatif, 
#misalnya: Dalam kasus Fraud/Scam, kecenderungan model mendeteksi transaksi sebagai fraud walaupun kenyataannya bukan, 
#dianggap lebih baik, daripada transaksi tersebut tidak terdeteksi sebagai fraud tetapi ternyata fraud. 
#Untuk problem ini sebaiknya menggunakan Recall.
#sebaliknya, jika lebih menginginkan terjadinya True Negatif dan sangat tidak menginginkan terjadinya False Positif, 
#sebaiknya menggunakan Precision.
#Contohnya adalah pada kasus klasifikasi email SPAM atau tidak. 
#Banyak orang lebih memilih jika email yang sebenarnya SPAM namun diprediksi tidak SPAM (sehingga tetap ada pada kotak masuk email kita), 
#daripada email yang sebenarnya bukan SPAM tapi diprediksi SPAM (sehingga tidak ada pada kotak masuk email).

#Pendahuluan
#Setelah pemahaman dengan prosedur machine learning modelling. 
#Selanjutnya materi akan membahas mengenai machine learning algorithm.
#Sebagai dasar, akan dipelajari beberapa algorithm machine learning yaitu Logistic Regression, 
#dan Decision Tree untuk classification problem, 
#dan Linear regression untuk regression problem.

#Classification - Logistic Regression
#Logistic Regression merupakan salah satu algoritma klasifikasi dasar yang cukup popular. 
#Secara sederhana, Logistic regression hampir serupa dengan linear regression tetapi linear regression digunakan untuk Label atau 
#Target Variable yang berupa numerik atau continuous value, 
#sedangkan Logistic regression digunakan untuk Label atau Target yang berupa categorical/discrete value.
#Contoh continuous value adalah harga rumah, harga saham, suhu, dsb; 
#dan contoh dari categorical value adalah prediksi SPAM or NOT SPAM (1 dan 0) atau 
#prediksi customer SUBSCRIBE atau UNSUBSCRIBED (1 dan 0).
#Umumnya Logistic Regression dipakai untuk binary classification (1/0; Yes/No; True/False) problem, 
#tetapi beberapa data scientist juga menggunakannya untuk multiclass classification problem.
#Logistic regression adalah salah satu linear classifier, oleh karena itu,
#Logistik regression juga menggunakan rumus atau fungsi yang sama seperti linear regression yaitu:
f(x)=b0+b1X1+...+brXr
#yang disebut Logit, dimana Variabel 𝑏₀, 𝑏₁, …, 𝑏ᵣ adalah koefisien regresi, dan 𝑥₁, …, 𝑥ᵣ adalah explanatory variable/variabel input atau feature.
#Output dari Logistic Regression adalah 1 atau 0;
#sehingga real value dari fungsi logit ini perlu ditransfer ke nilai di antara 1 dan 0
#dengan menggunakan fungsi sigmoid.
f(x)=1/1+e^-(x)
#Jadi, jika output dari fungsi sigmoid bernilai lebih dari 0.5,
# maka data point diklasifikasi ke dalam label/class: 1 atau YES; dan kurang dari 0.5,
# akan diklasifikasikan ke dalam label/class: 0 atau NO.
#Logistic Regression hanya dapat mengolah data dengan tipe numerik.
# Pada saat preparasi data, pastikan untuk mengecek tipe variabel yang ada dalam dataset dan pastikan semuanya adalah numerik, 
# lakukan data transformasi jika diperlukan.

#Pemodelan Permasalahan Klasifikasi dengan Logistic Regression
#Pemodelan Logistic Regression dengan memanfaatkan Scikit-Learn sangatlah mudah.
# Dengan menggunakan dataset yang sama yaitu online_raw,
# dan setelah dataset dibagi ke dalam Training Set dan Test Set,
# cukup menggunakan modul linear_model dari Scikit-learn,
# dan memanggil fungsi
LogisticRegression() # yang diberi nama logreg.
#Kemudian, model yang sudah ditraining ini
#bisa digunakan untuk memprediksi output/label dari test dataset sekaligus
#mengevaluasi model performance dengan fungsi
score(), confusion_matrix() dan classification_report().
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
# Call the classifier
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg = logreg.fit(X_train, y_train)
#Training Model: Predict
y_pred = logreg.predict(X_test)
#Evaluate Model Performance
print('Training Accuracy :', model.score(X_train,y_train))
print('Testing Accuracy :', model.score(X_test,y_test))
# confusion matrix
print('\nConfusion matrix')
cm = confusion_matrix(y_test,y_pred)
print(cm)
# classification report
print('\nClassification report')
cr = classification_report(y_test, y_pred)
print(cr)

#Classification - Decision Tree
#Decision Tree merupakan salah satu metode klasifikasi yang populer dan banyak diimplementasikan serta mudah diinterpretasi. 
#Decision tree adalah model prediksi dengan struktur pohon atau struktur berhierarki.
#Decision Tree dapat digunakan untuk classification problem dan regression problem.
#Secara sederhana, struktur dari decision tree adalah sebagai berikut:
   1.Decision Node yang merupakan feature/input variabel;
   2.Branch yang ditunjukkan oleh garis hitam berpanah, yang adalah rule/aturan keputusan, dan
   3.Leaf yang merupakan output/hasil.
#Decision Node paling atas dalam decision tree dikenal sebagai akar keputusan,
#atau feature utama yang menjadi asal mula percabangan.
#Jadi, decision tree membagi data ke dalam kelompok atau kelas berdasarkan feature/variable input,
#yang dimulai dari node paling atas (akar),
#dan terus bercabang ke bawah sampai dicapai cabang akhir atau leaf.
#Misalnya ingin memprediksi apakah seseorang yang mengajukan aplikasi kredit/pinjaman,
# layak untuk mendapat pinjaman tersebut atau tidak. Dengan menggunakan decision tree,
# dapat membreak-down kriteria-kriteria pengajuan pinjaman ke dalam hierarki
#Seumpama, orang yang mengajukan berumur lebih dari 40 tahun, dan memiliki rumah, maka aplikasi kreditnya dapat diluluskan, 
# sedangkan jika tidak, maka perlu dicek penghasilan orang tersebut.
# Jika kurang dari 5000, maka permohonan kreditnya akan ditolak.
# Dan jika usia kurang dari 40 tahun, maka selanjutnya dicek jenjang pendidikannya,
# apakah universitas atau secondary.
# Nah, percabangan ini masih bisa berlanjut hingga dicapai percabangan akhir/leaf node.
#Seperti yang sudah dilakukan dalam prosedur pemodelan machine learning,
#selanjutnya dapat dengan mudah melakukan pemodelan decision tree dengan menggunakan scikit-learn module, yaitu 
DecisionTreeClassifier.

#Tugas Praktek
#Dengan menggunakan dataset online_raw.csv dan diasumsikan sudah melakukan EDA dan pre-processing,
#Jadikan atau membuat model machine learning dengan menggunakan decision tree :
 ('''
 1.Import DecisionTreeClassifier dan panggil fungsi tersebut dengan nama decision_tree
 2.Split dataset ke dalam training & testing dataset dengan perbandingan 70:30=0.3, dengan random_state = 0
 3.Latih model dengan training feature (X_train) dan training target (y_train) menggunakan .fit()
 4.Evaluasi hasil model decision_tree yang sudah dilatih dengan testing feature (X_test)
 dan print nilai akurasi dari training dan testing dengan fungsi .score()
 ''')
#JAWAB
#import library
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Call the classifier
decision_tree=DecisionTreeClassifier()
# Fit the classifier to the training data
decision_tree = decision_tree.fit(X_train, y_train)

# evaluating the decision_tree performance
print('Training Accuracy :', decision_tree.score(X_train,y_train))
print('Testing Accuracy :', decision_tree.score(X_test,y_test))

#Regression: Linear Regression - Part 1
#Regression merupakan metode statistik dan machine learning yang paling banyak digunakan.
#Seperti yang dijelaskan sebelumnya,
#regresi digunakan untuk memprediksi output label yang berbentuk numerik atau continuous value.
#Dalam proses training, model regresi akan menggunakan variabel input (features) dan variabel output (label)
#untuk mempelajari bagaimana hubungan/pola dari variabel input dan output.
#Model regresi terdiri atas 2 tipe yaitu :
1.Simple regression model → model regresi paling sederhana, hanya terdiri dari satu feature (univariate) dan 1 target.
2.Multiple regression model → sesuai namanya, terdiri dari lebih dari satu feature (multivariate).
#Adapun model regresi yang paling umum digunakan adalah Linear Regression.

#Regression: Linear Regression - Part 2
#Linear regression digunakan untuk menganalisis hubungan linear antara dependent variabel (feature) dan independent variabel (label).
#Hubungan linear disini berarti bahwa jika nilai dari independen variabel mengalami perubahan baik itu naik atau turun,
#maka nilai dari dependen variabel juga mengalami perubahan (naik atau turun).
#Rumus matematis dari Linear Regression adalah:
y=a+bX #untuk simple linear regression, atau
y=a+b1X1+b2X2+...+biXi #untuk multiple liniear regression , y adalah target/label, X adalah feature, dan a,b adalah model parameter (intercept dan slope).
#Perlu diketahui bahwa tidak semua problem dapat diselesaikan dengan linear regression.
# Untuk pemodelan dengan linear regression, terdapat beberapa asumsi yang harus dipenuhi, yaitu :
1.Terdapat hubungan linear antara variabel input (feature) dan variabel output(label).
Untuk melihat hubungan linear feature dan label, dapat menggunakan chart seperti scatter chart.
Untuk mengetahui hubungan dari variabel umumnya dilakukan pada tahap eksplorasi data.
2.Tidak ada multicollinearity antara features.
Multicollinearity artinya terdapat dependency antara feature,
misalnya saja hanya bisa mengetahui nilai feature B jika nilai feature A sudah diketahui.
3.Tidak ada autocorrelation dalam data, contohnya pada time-series data.
#Pemodelan Linear regression menggunakan scikit-learn tidaklah sulit.
#Secara prosedur serupa dengan pemodelan logistic regression.
#Cukup memanggil LinearRegression dengan terlebih dahulu meng-import fungsi tersebut :
from sklearn.linear_model import LinearRegression
#Setelah memahami konsep dasar dari regression,
# kita akan berlatih membuat model machine learning dengan Linear regression.
# Untuk pemodelan ini kita akan menggunakan data ‘Boston Housing Dataset’.
# Setelah pembelajaran kamu sampai di sini,
# tahu tidak   mengapa kita tidak bisa menggunakan data “online purchase
#karena untuk linear regression target/label harus berupa numerik,
# sedangkan target dari online purchase data adalah categorical.
#Tepat sekali, Senja. Kalau begitu kita bisa lanjut ke pemodelan.
# Tujuan dari pemodelan ini adalah memprediksi harga rumah di Boston berdasarkan feature - feature yang ada. 
# Asumsikan saja bahwa kita sudah melakukan data eksplorasi dan data pre-processing.
# Jadi, data yang akan digunakan adalah data yang siap untuk diproses ke tahap pemodelan.

#Tugas Praktek
('''
 1.Pisahkan dataset ke dalam Feature dan Label, gunakan fungsi .drop().
 Pada dataset ini, label/target adalah variabel MEDV
 2.Checking dan print jumlah data setelah Dataset pisahkan ke dalam Feature dan Label, gunakan .shape()
 3.Bagi dataset ke dalam Training dan test dataset, 70% data digunakan untuk training dan 30% untuk testing, gunakan fungsi train_test_split() , 
 dengan random_state = 0
 4.Checking dan print kembali jumlah data dengan fungsi .shape()
 5.Import LinearRegression dari sklearn.linear_model
 6.Deklarasikan  LinearRegression regressor dengan nama reg
 7.Fit regressor ke training dataset dengan .fit(), dan gunakan .predict()
 untuk memprediksi nilai dari testing dataset.
''')
#jawab
#load dataset
import pandas as pd
housing = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/housing_boston.csv')
#Data rescaling
from sklearn import preprocessing
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
housing[['RM','LSTAT','PTRATIO','MEDV']] = data_scaler.fit_transform(housing[['RM','LSTAT','PTRATIO','MEDV']])
# getting dependent and independent variables
X = housing.drop(['MEDV'], axis = 1)
y = housing['MEDV']
# checking the shapes
print('Shape of X:', X.shape)
print('Shape of y:', y.shape)

# splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# checking the shapes
print('Shape of X_train :', X_train.shape)
print('Shape of y_train :', y_train.shape)
print('Shape of X_test :', X_test.shape)
print('Shape of y_test :', y_test.shape)

##import regressor from Scikit-Learn
from sklearn.linear_model import LinearRegression
# Call the regressor
reg = LinearRegression()
# Fit the regressor to the training data
reg = reg.fit(X_train,y_train)
# Apply the regressor/model to the test data
y_pred = reg.predict(X_test)

#Regression Performance Evaluation
#Untuk model regression, kita menghitung selisih antara nilai aktual (y_test) dan nilai prediksi (y_pred) yang disebut error, adapun beberapa metric yang umum digunakan. 
# Coba kamu ke mari, aku jelaskan langkah-langkahnya
#Mean Squared Error (MSE) adalah rata-rata dari squared error
#Root Mean Squared Error (RMSE) adalah akar kuadrat dari MSE
#Mean Absolute Error (MAE) adalah rata-rata dari nilai absolut error
#Semakin kecil nilai MSE, RMSE, dan MAE, semakin baik pula performansi model regresi.
# Untuk menghitung nilai MSE, RMSE dan MAE dapat dilakukan dengan
# menggunakan fungsi mean_squared_error () ,  mean_absolute_error () dari scikit-learn.metrics dan
# untuk RMSE sendiri tidak terdapat fungsi khusus di scikit-learn tapi dapat dengan mudah
# kita hitung dengan terlebih dahulu menghitung MSE kemudian menggunakan numpy module yaitu, sqrt()
# untuk memperoleh nilai akar kuadrat dari MSE.

#Tugas Praktek
('''
1.Import library yang digunakan: mean_squared_error, mean_absolute_error dari  sklearn.metrics
dan numpy sebagai aliasnya yaitu np. Serta, import juga matplotlib.pyplot sebagai aliasnya, plt.
2.Hitung dan print nilai MSE dan RMSE dengan menggunakan argumen y_test dan y_pred,
untuk rmse gunakan np.sqrt()
3.Buat scatter plot yang menggambarkan hasil prediksi (y_pred) dan harga actual (y_test)
''')

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

#Calculating MSE, lower the value better it is. 0 means perfect prediction
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error of testing set:', mse)
#Calculating MAE
mae = mean_absolute_error(y_test, y_pred)
print('Mean absolute error of testing set:', mae)
#Calculating RMSE
rmse = np.sqrt(mse)
print('Root Mean Squared Error of testing set:', rmse)

#Plotting y_test dan y_pred
plt.scatter(y_test, y_pred, c = 'green')
plt.xlabel('Price Actual')
plt.ylabel('Predicted value')
plt.title('True value vs predicted value : Linear Regression')
plt.show()

#Pendahuluan
#eperti yang sudah dijelaskan sebelumnya, Machine Learning terdiri atas 2 tipe yaitu supervised dan unsupervised learning. 
# Kita telah banyak membahas tentang supervised learning yaitu Klasifikasi model dan Regression Model. 
# Sekarang kita akan mempelajari dasar- dasar terkait unsupervised learning
#Unsupervised Learning adalah teknik machine learning dimana tidak terdapat label atau
# output yang digunakan untuk melatih model.
# Jadi, model dengan sendirinya akan bekerja untuk menemukan pola atau informasi dari dataset yang ada. 
# Metode unsupervised learning yang dikenal dengan clustering.
# Sesuai dengan namanya, Clustering memproses data dan mengelompokkannya atau
# mengcluster objek/sample berdasarkan kesamaan antar objek/sampel dalam satu kluster,
# dan objek/sample ini cukup berbeda dengan objek/sample di kluster yang lain.
#Pada awalnya kita tidak mengetahui bagaimana pola dari objek/sample,
# termasuk juga tidak mengetahui bagaimana kesamaan maupun perbedaan antara
# objek yang satu dengan objek yang lain. Setelah dilakukan clustering,
# baru dapat terlihat bawah objek/sample tersebut dapat dikelompokkan ke dalam 3 kluster.
# Untuk menjelaskan tentang metode Clustering,
# kita akan menggunakan metode clustering yang sangat populer,
# yaitu K-Means Algorithm yang akan kita praktikkan nanti

#K-Means Clustering
#"Jadi, Algorithm K-Means itu apa dan bagaimana cara kerjanya?” tanyaku antusias.
# “K-Means merupakan tipe clustering dengan centroid based (titik pusat).
# Artinya kesamaan dari objek/sampel dihitung dari seberapa dekat objek itu dengan centroid atau titik pusat.”
#Untuk menentukan centroid,
# pada awalnya kita perlu mendefinisikan jumlah centroid (K) yang diinginkan,
# semisalnya kita menetapkan jumlah K = 3; maka pada awal iterasi,
# algorithm akan secara random menentukan 3 centroid.
# Setelah itu, objek/sample/data point yang lain akan dikelompokkan sebagai anggota dari salah satu centroid yang terdekat, 
# sehingga terbentuk 3 cluster data.
#Iterasi selanjutnya, titik-titik centroid diupdate atau berpindah ke titik yang lain,
# dan jarak dari data point yang lain ke centroid yang baru dihitung kembali,
# kemudian dikelompokkan kembali berdasarkan jarak terdekat ke centroid yang baru.
# Iterasi akan terus berlanjut hingga diperoleh cluster dengan error terkecil,
# dan posisi centroid tidak lagi berubah
#Secara prosedur, tahap eksplorasi data untuk memahami karakteristik data,
# dan tahap preprocessing tetap dilakukan. Tetapi dalam unsupervised learning,
# kita tidak membagi dataset ke feature dan label; dan juga ke dalam training dan test dataset,
# karena pada dasarnya kita tidak memiliki informasi mengenai label/target data,

#Tugas Praktek
#“Untuk praktik  ini, kita akan menggunakan dataset ‘Mall Customer Segmentation’,” ujar Senja.
#Aku membaca detail latihan yang sudah ia catatkan untukku:
#Dataset ini merupakan data customer suatu mall dan berisi basic informasi customer berupa :
   1.CustomerID, age, gender, annual income, dan spending score.
   2.Adapun tujuan dari clustering adalah untuk memahami customer - customer mana saja yang sering melakukan transaksi sehingga 
   3.informasi ini dapat diberikan kepada marketing team untuk membuat strategi promosi yang sesuai dengan karakteristik customer.
#“Kita akan melakukan segmentasi customer, dengan memanfaatkan fungsi KMeans dari Scikit-Learn.cluster. 
#Silakan berlatih dengan intruksi di catatan tadi ya
1.Import pandas sebagai aliasnya dan KMeans dari sklearn.cluster.
2.Load dataset 'https://storage.googleapis.com/dqlab-dataset/pythonTutorial/mall_customers.csv' dan beri nama dataset
3.Diasumsikan EDA dan preprocessing sudah dilakukan,
selanjutnya kita memilih feature yang akan digunakan untuk membuat model yaitu annual_income dan spending_score. 
Assign dataset dengan feature yang sudah dipilih ke dalam 'X'.
Pada dasarnya terdapat teknik khusus yang dilakukan untuk menyeleksi feature - feature (Feature Selection) mana saja 
yang dapat digunakan untuk machine learning modelling,
karena tidak semua feature itu berguna. Beberapa feature justru bisa menyebabkan performansi model menurun. 
Tetapi untuk problem ini, secara default kita akan menggunakan annual_income dan spending_score.
4.Deklarasikan  KMeans( )  dengan nama cluster_model dan gunakan n_cluster = 5.
n_cluster adalah argumen dari fungsi KMeans( ) yang merupakan jumlah cluster/centroid (K).
random_state = 24.
5.Gunakan fungsi .fit_predict( ) dari cluster_model pada 'X'  untuk proses clustering.

#jawab
#import library
import pandas as pd
from sklearn.cluster import KMeans

#load dataset
dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/mall_customers.csv')

#selecting features
X = dataset[['annual_income','spending_score']]

#Define KMeans as cluster_model
cluster_model = KMeans(n_clusters = 5, random_state = 24)
labels = cluster_model.fit_predict(X)

#Tugas Praktek
#Inspect & Visualizing the Cluster
#“Satu lagi,kalau sudah membuat cluster, tolong  visualisasikan hasil dari clustering yang telah kamu lakukan sebelumnya ya. 
#Langkah-langkahnya sudah saya email,”
1.Pertama - tama, import matplotlib.pyplot dan beri inisial plt.
2.Gunakan fungsi .values untuk mengubah tipe ‘X’ dari dataframe menjadi array
3.Pisahkan X kedalam xs dan ys, di mana xs adalah Kolom index [0] dan ys adalah kolom index [1]
4.Buatlah scatter plot plt.scatter() dari xs dan ys, kemudian tambahkan c = labels untuk secara otomatis memberikan warna yang berbeda pada setiap cluster, dan alpha = 0.5 ke dalam scatter plot argumen.
5.Hitunglah koordinat dari centroid menggunakan .cluster_centers_ dari cluster_model, deklarasikan ke dalam variabel centroids.
6.Pisahkan centroids kedalam centroids_x dan centroids_y, di mana centroids_x adalah kolom index [0] dan centroids_y adalah kolom index [1]
7.Buatlah scatter plot dari centroids_x dan centroids_y , gunakan ‘D’ (diamond) sebagai marker parameter, dengan ukuran 50, s = 50

#hasil
#import library
import matplotlib.pyplot as plt

#convert dataframe to array
X = X.values
#Separate X to xs and ys --> use for chart axis
xs = X[:,0]
ys = X[:,1]
# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = cluster_model.cluster_centers_
# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,marker='D', s=50)
plt.title('KMeans Clustering', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

#Measuring Cluster Criteria
#bagaimana kita tahu bahwa membagi segmentasi ke dalam 5 cluster adalah segmentasi yang paling optimal? 
# Karena jika dilihat pada gambar beberapa data point masih cukup jauh jaraknya dengan centroidnya.
#Clustering yang baik adalah cluster yang data point-nya saling rapat/sangat berdekatan satu sama lain dan cukup berjauhan dengan objek/data point di cluster yang lain. Jadi, objek dalam satu cluster tidak tersebut berjauhan. 
# Nah, untuk mengukur kualitas dari clustering, kita bisa menggunakan inertia,
#Inertia sendiri mengukur seberapa besar penyebaran object/data point data dalam satu cluster, semakin kecil nilai inertia maka semakin baik. Kita tidak perlu bersusah payah menghitung nilai inertia karena secara otomatis, telah dihitung oleh KMeans( ) ketika algorithm di fit ke dataset.
# Untuk mengecek nilai inertia cukup dengan print fungsi .inertia_ dari model yang sudah di fit ke dataset
#Kalau begitu,   bagaimana caranya mengetahui nilai K yang paling baik dengan inertia yang paling kecil? 
# Apakah harus trial Error dengan mencoba berbagai jumlah cluster?
#Benar, kita perlu mencoba beberapa nilai, dan memplot nilai inertia-nya. 
# Semakin banyak cluster maka inertia semakin kecil.
#Meskipun suatu clustering dikatakan baik jika memiliki inertia yang kecil tetapi secara praktikal in real life, terlalu banyak cluster juga tidak diinginkan. 
# Adapun rule untuk memilih jumlah cluster yang optimal adalah dengan memilih jumlah cluster yang terletak pada “elbow” dalam intertia plot, yaitu ketika nilai inertia mulai menurun secara perlahan. 
# Jika dilihat pada gambar maka jumlah cluster yang optimal adalah K = 3

#Tugas Praktek
#Coba kamu membuat inertia plot untuk melihat apakah K = 5 merupakan jumlah cluster yang optimal. 
#Untuk membuat inertia plot, silakan memanfaatkan fungsi looping (for):

1.Pertama - tama, buatlah sebuah list kosong yang dinamakan 'inertia'. List ini akan kita gunakan untuk menyimpan nilai inertia dari setiap nilai K.
2.Gunakan for untuk membuat looping dengan range 1-10. Sebagai index looping gunakan k
3.Di dalam fungsi looping, deklarasikan  KMeans()  dengan nama cluster_model dan gunakan n_cluster = k, dan random_state = 24
4.Gunakan fungsi .fit() dari cluster_model pada 'X'
5.Dari dari cluster_model yang sudah di-fit ke dataset, dapatkan nilai inertia menggunakan inertia_ dan deklarasikan sebagai inertia_value
6.Append inertia_value ke dalam list 'inertia'
7.Setelah iterasi/looping selesai plotlah list 'inertia' tadi sebagai ordinat-nya dan absica-nya adalah range(1, 10).
#maka,
#import library
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Elbow Method - Inertia plot
inertia = []
#looping the inertia calculation for each k
for K in range(1, 10):
    #Assign KMeans as cluster_model
    cluster_model = KMeans(n_clusters = k, random_state = 24)
    #Fit cluster_model to X
    cluster_model.fit(X)
    #Get the inertia value
    inertia_value = cluster_model.inertia_
    #Append the inertia_value to inertia list
    inertia.append(inertia_value)
    
##Inertia plot
plt.plot(range(1, 10), inertia)
plt.title('The Elbow Method - Inertia plot', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('inertia')
plt.show()

#Case Study: Promos for our e-commerce - Part 1
#buatkan machine learning model untuk menyelesaikan permasalahan dari e-commerce divisi kantor.
#Adapun feature - feature dalam dataset ini adalah :
1.'Daily Time Spent on Site' : lama waktu user mengunjungi site (menit)
2.'Age' : usia user (tahun)
3.'Area Income' : rata - rata pendapatan di daerah sekitar user
4.'Daily Internet Usage' : rata - rata waktu yang dihabiskan user di internet dalam sehari (menit)
5.'Ad Topic Line' : topik/konten dari promo banner
5.'City' : kota dimana user mengakses website
7.'Male' : apakah user adalah Pria atau bukan
8.'Country' : negara dimana user mengakses website
9.'Timestamp' : waktu saat user mengklik promo banner atau keluar dari halaman website tanpa mengklik banner
10.'Clicked on Ad' : mengindikasikan user mengklik promo banner atau tidak (0 = tidak; 1 = klik).

#import library
import pandas as pd

# Baca data 'ecommerce_banner_promo.csv'
data = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/ecommerce_banner_promo.csv')

#1. Data eksplorasi dengan head(), info(), describe(), shape
print("\n[1] Data eksplorasi dengan head(), info(), describe(), shape")
print("Lima data teratas:")
print(data.head())
print("Informasi dataset:")
print(data.info())
print("Statistik deskriptif dataset:")
print(data.describe())
print("Ukuran dataset:")
print(data.shape)

#Case Study: Promos for our e-commerce - Part 2
#Sekarang mari melanjutkan dengan ekplorasi data untuk langkah ke-2 dan ke-3:
2.Data eksplorasi dengan dengan mengecek korelasi dari setiap feature menggunakan fungsi corr()
3.Data eksplorasi dengan mengecek distribusi label menggunakan fungsi groupby() dan size()
#2. Data eksplorasi dengan dengan mengecek korelasi dari setiap feature menggunakan fungsi corr()
print("\n[2] Data eksplorasi dengan dengan mengecek korelasi dari setiap feature menggunakan fungsi corr()")
print(data.corr())

#3. Data eksplorasi dengan mengecek distribusi label menggunakan fungsi groupby() dan size()
print("\n[3] Data eksplorasi dengan mengecek distribusi label menggunakan fungsi groupby() dan size()")
print(data.groupby('Clicked on Ad').size())

#Case Study: Promos for our e-commerce - Part 3
#Di proyek ini, aku akan melanjutkan mengeksplorasi data dengan visualisasi dengan tahap - tahap yang perlu dilakukan adalah (langkah ke-4):
4.Data eksplorasi dengan visualisasi:
-Jumlah user dibagi ke dalam rentang usia menggunakan histogram (hist()), 
gunakan bins = data.Age.nunique() sebagai argumen. nunique() adalah fungsi untuk menghitung jumlah data untuk setiap usia (Age).
-Gunakan pairplot() dari seaborn modul untuk menggambarkan hubungan setiap feature. 

#import library
import matplotlib.pyplot as plt
import seaborn as sns

# Seting: matplotlib and seaborn
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

#4. Data eksplorasi dengan visualisasi
#4a. Visualisasi Jumlah user dibagi ke dalam rentang usia (Age) menggunakan histogram (hist()) plot
plt.figure(figsize=(10, 5))
plt.hist(data['Age'], bins = data.Age.nunique())
plt.xlabel('Age')
plt.tight_layout()
plt.show()

#4b. Gunakan pairplot() dari seaborn (sns) modul untuk menggambarkan hubungan setiap feature.
plt.figure()
sns.pairplot(data)
plt.show()

#Case Study: Promos for our e-commerce - Part 4
#Di bagian proyek (langkah ke-5) ini aku akan mengecek apakah terdapat missing value dari data, 
# jika terdapat missing value dapat dilakukan treatment seperti didrop atau diimputasi dan jika tidak maka dapat melanjutkan ke langkah berikutnya.
5.Cek missing value
#5. Cek missing value
print("\n[5] Cek missing value")
print(data.isnull().sum().sum())

#Case Study: Promos for our e-commerce - Part 5
#Pada langkah ke-6 ini aku akan melakukan pemodelan dengan Logistic Regression dengan cara seperti berikut:
6.Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing :
-Deklarasikan data ke dalam X dengan mendrop feature/variabel yang bukan numerik, (type = object) dari data (Logistic Regression hanya dapat memproses numerik variabel). Assign Target/Label feature dan assign sebagai y
-Split X dan y ke dalam training dan testing dataset, gunakan perbandingan 80:20 dan random_state = 42
-Assign classifier sebagai logreg, kemudian fit classifier ke X_train dan predict dengan X_test. Print evaluation score.
#import library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#6.Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing
print("\n[6] Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing")
#6a.Drop Non-Numerical (object type) feature from X, as Logistic Regression can only take numbers, and also drop Target/label, assign Target Variable to y.   
X = data.drop(['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'], axis = 1)
y = data['Clicked on Ad']

#6b. splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

#6c. Modelling
# Call the classifier
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg = logreg.fit(X_train,y_train)
# Prediksi model
y_pred = logreg.predict(X_test)

#6d. Evaluasi Model Performance
print("Evaluasi Model Performance:")
print("Training Accuracy :", logreg.score(X_train,y_train))
print("Testing Accuracy :", logreg.score(X_test,y_test))

#Case Study: Promos for our e-commerce - Part 6
#Di langkah terakhir ini atau langkah ke-7 aku akan melihat performansi model dengan menggunakan confusion matrix dan classification report.
7.Print Confusion matrix dan classification report
# Import library
from sklearn.metrics import confusion_matrix, classification_report

#7. Print Confusion matrix dan classification report
print("\n[7] Print Confusion matrix dan classification report")

#apply confusion_matrix function to y_test and y_pred
print("Confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

#apply classification_report function to y_test and y_pred
print("Classification report:")
cr = classification_report(y_test,y_pred)
print(cr)

#Berdasarkan hasil evaluasi, apakah model yang dibuat cukup baik untuk memprediksi user yang akan mengklik website atau tidak?
#Metrik evaluasi apa yang tepat digunakan untuk mengevaluasi performansi dari model yang telah dilakukan training?
Model sudah sangat baik dalam memprediksi user yang akan mengklik website atau tidak, 
dapat dilihat dari nilai accuracy = 0.90; 
Dataset memiliki jumlah label yang seimbang (balance class), 
sehingga evaluasi performansi dapat menggunakan metrik Accuracy.

#Penutup/Kesimpulan
#Congratulations! Akhirnya selesai satu lagi modul Machine Learning With Python for Beginner. Berdasarkan materi-materi yang telah kupelajari dan praktekkan dalam modul ini, aku telah mendapatkan pengetahuan (knowledge) dan praktek (skill) yang diantaranya
1.Memahami apa itu machine learning dengan jenisnya untuk pemodelan
2.Memahami dan mampu melakukan Eksplorasi Data & Data Pre-processing
3.Memahami dan mampu melakukan proses-proses Pemodelan dengan Scikit-Learn
4.Memahami dan mampu melakukan proses-proses pemodelan dengan menggunakan algoritma pada Supervised Learning
5.Memahami dan mampu melakukan proses-proses pemodelan dengan menggunakan algoritma pada Unsupervised Learning
6.Mengerjakan mini project yang merupakan integrasi keseluruhan materi dan tentunya materi-materi pada modul-modul sebelumnya untuk menyelesaikan persolan bisnis.
