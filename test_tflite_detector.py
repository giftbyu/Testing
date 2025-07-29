import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import tkinter as tk
from tkinter import font

# --- Bagian 1: Logika Inti Deteksi (Dibungkus dalam sebuah fungsi) ---
def start_detection(metode):
    """Fungsi utama untuk memulai deteksi webcam dengan model yang dipilih."""
    
    print(f"Memulai deteksi untuk metode: {metode.upper()}")

    # Konfigurasi spesifik untuk setiap metode
    if metode == 'sibi':
        MODEL_PATH = './models/sibi_model_lightweight.tflite'
        DATA_DIR = './dataset/SIBI'
        NUM_LANDMARKS = 21 * 3
        MAX_HANDS = 1
    else: # 'bisindo'
        MODEL_PATH = './models/bisindo_model_lightweight.tflite'
        DATA_DIR = './dataset/BISINDO'
        NUM_LANDMARKS = 42 * 3
        MAX_HANDS = 2

    # Pemuatan Model dan Persiapan
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error memuat model: {e}")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    CLASS_NAMES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_HANDS, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    # Loop Deteksi
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Pra-pemrosesan
        image_for_cnn = np.expand_dims(cv2.resize(frame, (128, 128)), axis=0)
        landmarks_vector = np.zeros(NUM_LANDMARKS, dtype=np.float32)

        if results.multi_hand_landmarks:
            if metode == 'sibi':
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                landmarks_vector = (coords - coords[0]).flatten()
            elif metode == 'bisindo':
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
        
        # Inferensi
        image_input_index = next((i for i, d in enumerate(input_details) if 'image' in d['name']), None)
        landmark_input_index = next((i for i, d in enumerate(input_details) if 'landmark' in d['name']), None)

        if image_input_index is not None and landmark_input_index is not None:
            # --- PERBAIKAN TIPE DATA DI SINI ---
            # Secara eksplisit ubah tipe data menjadi float32 sebelum dikirim ke interpreter
            image_data = image_for_cnn.astype(np.float32)
            landmark_data = landmarks_for_mlp.astype(np.float32)

            interpreter.set_tensor(input_details[image_input_index]['index'], image_data)
            interpreter.set_tensor(input_details[landmark_input_index]['index'], landmark_data)
            
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            confidence = np.max(output_data)
            
            CONFIDENCE_THRESHOLD = 0.90
            if confidence > CONFIDENCE_THRESHOLD:
                predicted_class_index = np.argmax(output_data)
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                text = f"{predicted_class_name} ({confidence*100:.2f}%)"
                cv2.putText(display_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Deteksi Bahasa Isyarat Real-time', display_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Bagian 2: GUI Launcher dengan Tkinter (Tidak ada perubahan) ---
def main():
    def run_detection(metode):
        root.destroy()
        start_detection(metode)

    root = tk.Tk()
    root.title("Pilih Metode Deteksi")
    
    window_width = 400; window_height = 200
    screen_width = root.winfo_screenwidth(); screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width / 2); center_y = int(screen_height/2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    root.configure(bg='#f0f0f0')
    title_font = font.Font(family="Helvetica", size=16, weight="bold")
    button_font = font.Font(family="Helvetica", size=12)

    title_label = tk.Label(root, text="SignBridge Detector", font=title_font, bg='#f0f0f0')
    title_label.pack(pady=20)

    button_frame = tk.Frame(root, bg='#f0f0f0')
    button_frame.pack(pady=10)

    sibi_button = tk.Button(
        button_frame, text="Mulai Deteksi SIBI", font=button_font, command=lambda: run_detection('sibi'),
        bg='#4CAF50', fg='white', relief='flat', padx=10, pady=5
    )
    sibi_button.pack(side='left', padx=10)

    bisindo_button = tk.Button(
        button_frame, text="Mulai Deteksi BISINDO", font=button_font, command=lambda: run_detection('bisindo'),
        bg='#2196F3', fg='white', relief='flat', padx=10, pady=5
    )
    bisindo_button.pack(side='left', padx=10)

    root.mainloop()

if __name__ == '__main__':
    main()