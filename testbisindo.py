import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

# --- PILIH MODEL YANG INGIN DIGUNAKAN ---
MODEL_TO_USE = 'BISINDO'
MODEL_PATH = 'models\Bisindo_hybrid_model_v3_finetuned.keras' # Ganti nama file jika perlu
DATA_DIR = './dataset/BISINDO'
NUM_LANDMARKS = 42 * 3 # HARUS SAMA DENGAN TRAINING

# --- PEMUATAN MODEL & KONFIGURASI (PERUBAHAN) ---
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
mp_hands = mp.solutions.hands
# max_num_hands diubah menjadi 2
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # --- PRA-PEMROSESAN (LOGIKA BARU) ---
    image_for_cnn = np.expand_dims(cv2.resize(frame, (128, 128)), axis=0).astype('float32')
    landmarks_vector = np.zeros(NUM_LANDMARKS, dtype=np.float32)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            handedness = results.multi_handedness[i].classification[0].label
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            relative_coords = (coords - coords[0]).flatten()
            
            if handedness == 'Right':
                landmarks_vector[0:63] = relative_coords
            elif handedness == 'Left':
                landmarks_vector[63:126] = relative_coords
    
    landmarks_for_mlp = np.expand_dims(landmarks_vector, axis=0)

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