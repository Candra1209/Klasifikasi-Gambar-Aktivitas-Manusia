# Klasifikasi-Gambar-Aktivitas-Manusia
ini merupakan penelitian yang saya lakukan saat menempu pendidikan S1 - Ilmu Komputer, penelitian ini saya lakukan untuk menyelesaikan tugas akhir pada tahun 2023. pada kesempatan ini saya membuat model deep learning dengan menggunakan arsitektur ***MobileNet V3 small*** dan ***Large***, untuk mengklasifikasi gambar dari **15 aktivitas manusia**.

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
| Total Gambar | 15000 | 3000 | 18000 |
