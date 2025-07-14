import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    classifier_model_path = os.path.join('models', 'sibi_efficientnetb0_best_model.h5')
    classifier_model = tf.keras.models.load_model(classifier_model_path)
    print(f"Model klasifikasi berhasil dimuat dari: {classifier_model_path}")
except Exception as e:
    print(f"Error: Gagal memuat model klasifikasi. Pastikan file '{classifier_model_path}' ada.")
    print(e); exit()

CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
IMG_SIZE = (224, 224)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera."); exit()

print("\nKamera siap. Arahkan tangan Anda ke kamera.")
print("Tekan tombol 'q' untuk keluar.")

while True:
    success, frame = cap.read()
    if not success:
        print("Gagal menangkap frame. Keluar..."); break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

            padding = 30
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.size > 0:
                img_resized = cv2.resize(hand_crop, IMG_SIZE)
                img_array = np.expand_dims(img_resized, axis=0)

                predictions = classifier_model.predict(img_array, verbose=0)
                predicted_index = np.argmax(predictions[0])
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = np.max(predictions[0]) * 100

                text = f"{predicted_class} ({confidence:.1f}%)"
                cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Deteksi Bahasa Isyarat (MediaPipe + TensorFlow) - Tekan 'q'", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
print("Program dihentikan.")