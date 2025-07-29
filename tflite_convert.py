import tensorflow as tf
import os

# --- KONFIGURASI ---
# Pilih model mana yang ingin dikonversi
MODEL_TO_CONVERT = 'SIBI' # Ganti ke 'SIBI' jika perlu

if MODEL_TO_CONVERT == 'SIBI':
    KERAS_MODEL_PATH = './models/sibi_hybrid_model_v_final.keras'
    DATA_DIR = './dataset/SIBI'
    TFLITE_OUTPUT_PATH = './models/sibi_model_lightweight1.tflite' # Nama file output baru
else:
    KERAS_MODEL_PATH = './models/bisindo_hybrid_model_v4_finetuned.keras'
    DATA_DIR = './dataset/BISINDO'
    TFLITE_OUTPUT_PATH = './models/bisindo_model_lightweight1.tflite' # Nama file output baru

LABEL_FILE_PATH = './models/labels.txt'

# --- Langkah 1: Buat File Label (Tetap penting untuk aplikasi Anda!) ---
print(f"Membuat file label '{LABEL_FILE_PATH}' untuk referensi aplikasi...")
CLASS_NAMES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
with open(LABEL_FILE_PATH, 'w') as f:
    f.write('\n'.join(CLASS_NAMES))
print("File label berhasil dibuat. Simpan file ini bersama model Anda.")

# --- Langkah 2: Konversi Inti (Sesuai Dokumentasi Google AI) ---
print(f"Memuat model Keras dari '{KERAS_MODEL_PATH}'...")
# Kita tidak perlu `compile=False` di sini, tapi ini adalah praktik yang aman
model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False) 

print("Mengonversi model ke format TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Terapkan optimasi untuk ukuran file yang lebih kecil
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
tflite_model_bytes = converter.convert()
print("Konversi dasar berhasil.")

# --- Langkah 3: Simpan Model TFLite ---
print(f"Menyimpan model TFLite ringan ke '{TFLITE_OUTPUT_PATH}'...")
with open(TFLITE_OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model_bytes)

print("\nProses selesai!")
print(f"Model TFLite ringan berhasil dibuat di: {TFLITE_OUTPUT_PATH}")