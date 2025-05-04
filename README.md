# Real-Time Object Color Detection using CNN Models (VGG16, Xception, NASNet Mobile)

Penellitian ini bertujuan untuk mengembangkan model deteksi warna objek secara real-time berbasis kamera, sebagai solusi bagi penderita buta warna. Model dibangun menggunakan Convolutional Neural Network (CNN) dengan membandingkan tiga arsitektur CNN populer: VGG16, Xception, dan NASNet Mobile.

📌 **Deskripsi Proyek**
Penelitian ini mencakup beberapa tahapan utama:
1. Pengumpulan dataset warna objek.
2. Preprocessing gambar:
    - Resize
    - Augmentasi
    - Normalisasi
3. Pelatihan dan evaluasi model CNN:
    - VGG16
    - Xception
    - NASNet Mobile
4. Pengujian performa model menggunakan metrik:
    - Akurasi
    - Precision
    - Recall
    - F1-Score
5 Implementasi model terbaik (NASNet Mobile) ke dalam model deteksi warna objek secara real-time menggunakan webcam.
6. Evaluasi real-time pada 20 objek nyata.

📊 **Hasil Pengujian**
| Model         | Akurasi | Waktu Prediksi (2904 gambar) |
|---------------|---------|------------------------------|
| VGG16         | 90%     | 10 menit 9 detik             |
| Xception      | 86%     | 3 menit 22 detik             |
| NASNet Mobile | 88%     | 2 menit 22 detik             |

Meskipun VGG16 memiliki akurasi tertinggi, NASNet Mobile dipilih sebagai model terbaik karena memiliki performa prediksi tercepat dengan akurasi yang tetap tinggi. Model ini kemudian diintegrasikan dalam model deteksi warna objek real-time yang dapat menampilkan nama warna langsung di layar saat objek ditunjukkan ke kamera.

📷 **Real-Time Detection**
Model terbaik diterapkan untuk mendeteksi warna objek secara real-time menggunakan webcam. Model dapat mengenali dan menampilkan nama warna dari objek nyata secara langsung di layar.

🔧 **Teknologi yang Digunakan**
1. Python
2. TensorFlow / Keras
3. OpenCV
4. Scikit-learn
5. Numpy
6. Matplotlib
7. CNN Models: VGG16, Xception, NASNet Mobile

💡 **Tujuan Penelitian**
Memberikan solusi teknologi bagi penderita buta warna dengan membangun sistem yang dapat mengenali warna objek secara akurat dan real-time.
