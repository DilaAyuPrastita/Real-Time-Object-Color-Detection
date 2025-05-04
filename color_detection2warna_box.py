# bounding box, ada warna tidak diketahui

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load model yang telah disimpan
model = tf.keras.models.load_model("model_dila_vgg16_var3.h5")

# # Label kelas yang sesuai dengan dataset pelatihan
# class_labels = ["Merah", "Hijau", "Biru", "Kuning"]

# # Warna yang sesuai dengan label (BGR)
# color_map = {
#     "Merah": (0, 0, 255),   # Merah dalam BGR
#     "Hijau": (0, 255, 0),   # Hijau dalam BGR
#     "Biru": (255, 0, 0),    # Biru dalam BGR
#     "Kuning": (0, 255, 255) # Kuning dalam BGR
# }

# # Ambang batas confidence minimal untuk deteksi warna
# CONFIDENCE_THRESHOLD = 0.6  

# # Buka kamera
# cap = cv2.VideoCapture(0)  # 0 untuk webcam utama

# # Tentukan ukuran bounding box di tengah
# frame_width = 640  # Lebar frame
# frame_height = 480  # Tinggi frame
# center_x = frame_width // 2  # Titik tengah X
# center_y = frame_height // 2  # Titik tengah Y
# box_width = 400  # Lebar area deteksi (bounding box)
# box_height = 300  # Tinggi area deteksi (bounding box)

# while True:
#     ret, frame = cap.read()  # Baca frame dari kamera
#     if not ret:
#         break

#     # Gambar bounding box di tengah
#     start_x = center_x - box_width // 2
#     start_y = center_y - box_height // 2
#     end_x = center_x + box_width // 2
#     end_y = center_y + box_height // 2
#     cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)  # Gambar kotak biru di tengah

#     # Crop frame untuk hanya menggunakan area tengah
#     roi = frame[start_y:end_y, start_x:end_x]

#     # Konversi ke grayscale dan blur untuk mendeteksi objek lebih baik
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Deteksi tepi menggunakan Canny Edge Detection
#     edges = cv2.Canny(blurred, 30, 150)

#     # Temukan kontur objek dalam frame
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         if cv2.contourArea(contour) > 500:  # Filter objek kecil agar tidak terdeteksi
#             x, y, w, h = cv2.boundingRect(contour)
#             # Pastikan kontur berada dalam area bounding box tengah
#             if x + w // 2 >= start_x and x + w // 2 <= end_x and y + h // 2 >= start_y and y + h // 2 <= end_y:
#                 roi_object = roi[y:y+h, x:x+w]  # Ambil region of interest (ROI) untuk objek

#                 # Resize ROI agar sesuai dengan input model
#                 img = cv2.resize(roi_object, (224, 224))
#                 img = img_to_array(img)
#                 img = np.expand_dims(img, axis=0)  # Tambahkan batch dimension
#                 img = img / 255.0  # Normalisasi

#                 # Prediksi kelas warna
#                 prediction = model.predict(img)
#                 predicted_class = np.argmax(prediction)
#                 confidence = np.max(prediction)  # Ambil confidence tertinggi

#                 # Cek apakah confidence cukup tinggi
#                 if confidence >= CONFIDENCE_THRESHOLD:
#                     predicted_label = class_labels[predicted_class]
#                     color = color_map[predicted_label]
#                     print("Predicted:", predicted_label, "| Confidence:", confidence)
            
                
#                     # Gambar bounding box di sekitar objek
#                     cv2.rectangle(frame, (start_x + x, start_y + y), (start_x + x + w, start_y + y + h), color, 2)

#                     # Tampilkan hasil prediksi di layar
#                     cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (start_x + x, start_y + y - 10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

#     cv2.imshow("Color Detection", frame)

#     # Tekan tombol 'q' untuk keluar
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Tutup kamera dan jendela
# cap.release()
# cv2.destroyAllWindows()


# Label kelas yang sesuai dengan dataset pelatihan
class_labels = ["Merah", "Hijau", "Biru", "Kuning"]

# Warna yang sesuai dengan label (BGR)
color_map = {
    "Merah": (0, 0, 255),
    "Hijau": (0, 255, 0),
    "Biru": (255, 0, 0),
    "Kuning": (0, 255, 255)  
}

# Ambang batas confidence minimal untuk deteksi warna
CONFIDENCE_THRESHOLD = 0.7

# Buka kamera
cap = cv2.VideoCapture(0)  

# Tentukan ukuran bounding box di tengah
frame_width = 640  
frame_height = 480  
center_x = frame_width // 2  
center_y = frame_height // 2  
box_width = 400  
box_height = 300  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi ke HSV untuk meningkatkan akurasi warna
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Gambar bounding box di tengah
    start_x = center_x - box_width // 2
    start_y = center_y - box_height // 2
    end_x = center_x + box_width // 2
    end_y = center_y + box_height // 2
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

    # Crop frame untuk hanya menggunakan area tengah
    roi = frame[start_y:end_y, start_x:end_x]

    # Konversi ke HSV dan ekstraksi nilai V untuk pencahayaan
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_channel = hsv_roi[:, :, 2]  

    # Equalize histogram untuk memperbaiki kontras
    v_channel = cv2.equalizeHist(v_channel)
    hsv_roi[:, :, 2] = v_channel
    roi = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2BGR)

    # Konversi ke grayscale dan blur untuk meningkatkan deteksi objek
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Deteksi tepi menggunakan Canny Edge Detection
    edges = cv2.Canny(blurred, 30, 150)

    # Temukan kontur objek dalam bounding box
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 300:  # Turunkan threshold agar objek lebih mudah dideteksi
            x, y, w, h = cv2.boundingRect(contour)
            
            # Pastikan kontur berada dalam area bounding box
            if start_x <= x + start_x + w // 2 <= end_x and start_y <= y + start_y + h // 2 <= end_y:
                roi_object = roi[y:y+h, x:x+w]  # Ambil ROI objek

                # Pastikan ukuran ROI valid sebelum resizing
                if roi_object.shape[0] > 0 and roi_object.shape[1] > 0:
                    img = cv2.resize(roi_object, (224, 224))
                    img = img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = img / 255.0  # Normalisasi

                    # Prediksi warna
                    prediction = model.predict(img)
                    predicted_class = np.argmax(prediction)
                    confidence = np.max(prediction)

                    # Cek apakah confidence cukup tinggi
                    if confidence >= CONFIDENCE_THRESHOLD:
                        predicted_label = class_labels[predicted_class]
                        color = color_map[predicted_label]
                    else:
                        continue  # Lewati jika confidence tidak memenuhi threshold

                    # Gambar bounding box dan label pada objek
                    cv2.rectangle(frame, (start_x + x, start_y + y), (start_x + x + w, start_y + y + h), color, 2)
                    cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (start_x + x, start_y + y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    cv2.imshow("Color Detection", frame)

    # Tekan tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
