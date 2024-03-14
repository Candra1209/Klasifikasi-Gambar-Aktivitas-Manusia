# Klasifikasi-Gambar-Aktivitas-Manusia
Ini merupakan penelitian yang saya lakukan saat menempu pendidikan S1 - Ilmu Komputer, penelitian ini saya lakukan untuk menyelesaikan tugas akhir pada tahun 2023. pada kesempatan ini saya membuat model deep learning dengan menggunakan arsitektur ***MobileNet V3 small*** dan ***Large***, untuk mengklasifikasi gambar dari **15 aktivitas manusia**.

model akan dilatih dengan skenario seperti gambar dibawah :
![skenario penelitan](Gambar/Rujukan/Drawing1.jpg)

## Dataset Penelitian
Dataset yang digunakan pada penelitian kali ini adalah berupa gambar aktivitas manusia yang terdiri dari **15 kelas** aktivitas dengan total keseluruhan gambar sebanyak **18000** gambar, dimana masing masing kelas memiliki **1200** gambar

dataset dapat di lihat pada [website kaggle.com](https://www.kaggle.com/datasets/emirhanai/human-action-detection-artificial-intelligence)

untuk gambaran mendetail mengenai dataset dapat dilihat dibawah:

| Dataset properti | informasi |
| ------------ | --- |
| Jumlah Kelas | 15 |
| Data latih/kelas | 1000 |
| Data uji/kelas | 200 |
| Total data latih | 15000 |
| Total data uji | 3000 |
| Total gambar | 18000 |
| Sumber | https://www.kaggle.com/datasets/emirhanai/human-action-detection-artificial-intelligence |
| Format | .jpg |
| Size terbesar | 26 KB |
| Size terkecil | 2 KB |
| Size dataset | 142 MB |
| Ukuran terbesar | 400x126 pixel |
| Ukuran terkecil | 95x84 pixel |

untuk kelas-kelas yang terdapat pada dataset ini dapat dilihat di table bawah:

| kelas | data latih | data test | total gambar|
|-------|------------|-----------|-------------|
| calling | 1000 | 200 | 1200 |
| clapping | 1000 | 200 | 1200 |
| cycling | 1000 | 200 | 1200 |
| dancing | 1000 | 200 | 1200 |
| drinking | 1000 | 200 | 1200 |
| eating | 1000 | 200 | 1200 |
| fighting | 1000 | 200 | 1200 |
| hugging | 1000 | 200 | 1200 |
| laughing | 1000 | 200 | 1200 |
| listening to music | 1000 | 200 | 1200 |
| running | 1000 | 200 | 1200 |
| sitting | 1000 | 200 | 1200 |
| sleeping | 1000 | 200 | 1200 |
| texting | 1000 | 200 | 1200 |
| using a | 1000 | 200 | 1200 |
| **Total Gambar** | **15000** | **3000** | **18000** |

## Pre-processing
Proses preprocessing bertujuan untuk menyamaratakan data sehingga semua data memiliki format yang sama untuk dimasukan kedalam model.
beberapa proses pre-processing yang digunakan kali ini adalah :
### resize
dimana semua gambar akan diubah ukuran menjadi 224x224 pixel, dimana teknik resize yang digunakan adalah [bilinear interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation)

sebenarnya kita dapat memilih teknik resize yang kita mau, [resize tensorflow](https://www.tensorflow.org/api_docs/python/tf/image/resize) sendiri menyediakan beberapa teknik resize seperti :
- AREA	'area'
- BICUBIC	'bicubic'
- BILINEAR	'bilinear'
- GAUSSIAN	'gaussian'
- LANCZOS3	'lanczos3'
- LANCZOS5	'lanczos5'
- MITCHELLCUBIC	'mitchellcubic'
- NEAREST_NEIGHBOR	'nearest'

dan kita dapat ubah dengan
```
tf.image.resize(
    images, 
    size, 
    method=ResizeMethod.BILINEAR, # ganti metode yang di inginkan di sini
    preserve_aspect_ratio=False,
    antialias=False,
    name=None
)
```
## Data Augmentation
Untuk mengatasi overfitting saya menggunakan data augmentation agar data yang digunakan selama proses pelatihan lebih beragam. teknik ini sering digunakan dibanyak penelitian dan merupakan cara paling ampuh untuk mengatasi dataset yang sedikit dan kurang keberagaman. beberapa teknik ini dapat digunakan dengan [ImageDataGenerator pada TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) sedangkan untuk random cutout merupakan teknik tambahan (sampai saat respatori ini dibuat belum ada pada tansorflow)

pada kesempatan ini beberapa teknik augmentasi yang digunakan antara lain:
- random rotate dengan besar 30 derajat, 
- random flip (horizontal dan vertical), 
- shear(kemiringan), 
- fill (mengisi ruang kosong) dengan nearest neighbor, 
- [random cutout (melubangi gambar secara random)](https://github.com/yu4u/cutout-random-erasing).

berikut contoh data yang telah melalui data augmentation :
![Tampilan gambar yang melalui augmentation](Gambar/Rujukan/dataAugmented.png)

## Transfer Learning
Akan lebih mudah mengajari seseorang yang memilii pengetahuan dasar untuk mengerjakan tugas spesific, hal ini juga berlaku pada model Deep Learning. Transfer Learning memungkinkan kita untuk mengunakan pengatahuan yang diperoleh dari pelatihan sebelumnya ke dalam model baru sehingga dari pada membiarkan model belajar dari null, setidaknya model memiliki pengetahuan dasar mengenai pola dan bentuk. 
model pada percobaan ini akan mendapatkan transfer pengetahuan dari pelatihan sebelumnya yang dimana pelatihan tersebut dilakukan pada dataset [ImageNet](https://www.image-net.org/about.php) yang merupakan dataset besar dengan 1000 kelas.

kita dapat mengunakan teknik ini pada [Tensorflow saat mendeklarasikan model](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Large)
```
model = tf.keras.applications.MobileNetV3Large(
    input_shape=None,
    alpha=1.0,
    # disini untuk menetukan apakah jenis learning yang dinginkan
    #'none' untuk tidak menggunakan teknik
    # atau masukan path weight file yang ingin digunakan
    weights='imagenet',
    classes=1000,
    pooling=None,
    classifier_activation='softmax',
)
```

## Konfigurasi Arsitektur
Model Deep Learning CNN sederhana terdiri dari tiga layer utama (tergantung model) yaitu **layer convolution**, **layer flatten** dan **classifier**. Layer convolution ini merupakan model CNN yang akan digunakan, dimana pada kesempatan ini saya akan mengguanakan [**MobileNetV3 Small**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small) dan [**MobileNetV3 Large**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Large), 

> perlu dicatat bahwa transfer learning hanya terjadi pada layer convolution, tidak pada flatten dan classifer
> kita dapat menentukan apakah layer convolution yang menerima transfer learning untuk ikut belajar atau tidak dengan menggunakan trainable true atau false
> ```
>pretrained_model= tf.keras.applications.MobileNetV3Small(include_top=False,
>                                                                   alpha=1.0,                                      
>                                                                   input_shape=(224,224,3),
>                                                                   pooling='avg',
>                                                                   classes=15,
>                                                                   weights='imagenet')
>
>for layer in pretrained_model.layers:
>        layer.trainable=True # true untuk ikut belajar (unfreeze), fales untuk tidak ikut belajar (freeze)
> ```

untuk memahami bagaimana bentuk dari model, silahkan buka langsung [paper penelitian oleh Andrew Howard, dkk](https://arxiv.org/abs/1905.02244v5)

## FLatten
[**layer flatten**](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) merupakan layer yang akan mengubah output dari convolution yang merupakan data gambar dengan bentuk data (n x n) menjadi (1 x n) agar dapat digunakan pada layer classifier,

![Layer Flatten](Gambar/Rujukan/flattening.jpg)

### Classifier
 sedangkan classifier merupakan layer yang akan mengkalsifikasi dan mengekstrak informasi dari setiap gambar yang masuk pada penelitian ini classifier yang digunakan adalah [**Multi-layer Perceptron**](https://en.wikipedia.org/wiki/Multilayer_perceptron).
 
Classifier yang akan digunakan pada penelitian adalah multilayer perceptron dengan 3 layer utama, setiap layer terdiri dari [**drop out**](https://databasecamp.de/en/ml/dropout-layer-en#:~:text=The%20dropout%20layer%20is%20a,the%20network%20architecture%20at%20all.) dan [**fully connected layer (FCL)**](https://medium.com/@vaibhav1403/fully-connected-layer-f13275337c7c)

structur MLP yang digunakan sebagai classifer terdiri dari
| layer | unit | drop rate | fungsi aktivasi | Regularization |
|-------|------|-----------|-----------------|----------------|
|flatten|_s_ 1024 / _l_ 1280 |~|~|~|
|drop out|_s_ 1024 / _l_ 1280 |0.4|~|~|
|FCL|728|~|ReLU|L2|
|drop out|728|0.3|~|~|
|FCL|256|~|ReLU|L2|
|drop out|256|0.2|~|~|
|FCL|15|~|softmax|L2|

```
regularizer = 'l2'

classifier = Sequential()
classifier.add(Flatten())
classifier.add(Dropout(0.4))
classifier.add(Dense(units=728, activation="ReLU", kernel_regularizer=regularizer))
classifier.add(Dropout(0.3))
classifier.add(Dense(units=256, density=0.3, activation="ReLU", kernel_regularizer=regularizer))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=15, density=0.2, activation="softmax", kernel_regularizer=regularizer))
```

lalu gabungkan model CNN dan classifer yang baru dibuat
```
model= Sequential()
model.add(pretrained_model) #sesuaikan dengan arsitektur yang dipanggil sebelumnya
model.add(classifier)
```
