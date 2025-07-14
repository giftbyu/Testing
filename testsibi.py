import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

# --- KONFIGURASI DAN PEMUATAN MODEL ---
try:
    model = tf.keras.models.load_model('./models/sibi_hybrid_model_v2_finetuned.keras')
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model: {e}")
    exit()

DATA_DIR = './dataset/SIBI'
CLASS_NAMES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"Kelas yang akan dideteksi: {CLASS_NAMES}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# --- LOOP DETEKSI REAL-TIME ---
while True:
    success, frame = cap.read()
    if not success:
        print("Gagal membaca frame dari kamera.")
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # --- PERBAIKAN LOGIKA DIMULAI DI SINI ---

    # Hanya proses jika tangan terdeteksi (Solusi 1: Gerbang Logika)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Gambar landmark untuk visualisasi
        mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # --- PRA-PEMROSESAN PER-FRAME ---
        image_for_cnn = cv2.resize(frame, (128, 128))
        image_for_cnn = np.expand_dims(image_for_cnn, axis=0).astype('float32')

        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        base_point = coords[0]
        relative_coords = coords - base_point
        landmarks_for_mlp = np.expand_dims(relative_coords.flatten(), axis=0)

        # --- PREDIKSI MODEL ---
        prediction = model.predict([image_for_cnn, landmarks_for_mlp], verbose=0)
        confidence = np.max(prediction)
        
        # Hanya tampilkan jika confidence tinggi (Solusi 2: Threshold)
        CONFIDENCE_THRESHOLD = 0.85 # Atur ambang batas di sini (95%)
        
        if confidence > CONFIDENCE_THRESHOLD:
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            
            # --- VISUALISASI HASIL ---
            text = f"{predicted_class_name} ({confidence*100:.2f}%)"
            cv2.putText(display_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

    # Tampilkan frame ke layar (akan selalu berjalan)
    cv2.imshow('Deteksi Bahasa Isyarat Real-time', display_frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()