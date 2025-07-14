import cv2
import os
import time

# --- KONFIGURASI ---
DATA_DIR = './dataset'
METHODES = ['SIBI', 'BISINDO']
ALPHABETS = [chr(i) for i in range(65, 91)]
NUM_IMAGES_PER_ALPHABET = 200
COUNTDOWN_SECONDS = 3 # Jeda waktu dalam detik

# --- PEMBUATAN DIREKTORI ---
for methode in METHODES:
    for alphabet in ALPHABETS:
        path = os.path.join(DATA_DIR, methode, alphabet)
        os.makedirs(path, exist_ok=True)

print("Struktur direktori berhasil dibuat/diverifikasi.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# --- PENGAMBILAN GAMBAR ---
for methode in METHODES:
    for alphabet in ALPHABETS:
        path = os.path.join(DATA_DIR, methode, alphabet)
        num_existing_images = len(os.listdir(path))

        if num_existing_images >= NUM_IMAGES_PER_ALPHABET:
            print(f"Melewati: {methode}/{alphabet} (Sudah lengkap)")
            continue
        
        print(f"\nMengumpulkan gambar untuk: Metode -> {methode}, Abjad -> {alphabet}")
        
        # Loop untuk menunggu tombol 'S'
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"SIAPKAN TANGAN UNTUK ABJAD: {alphabet}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Tekan 'S' untuk Mulai...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Pengumpul Dataset', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        
        # --- PERUBAHAN DI SINI: TAMBAHKAN COUNTDOWN ---
        for i in range(COUNTDOWN_SECONDS, 0, -1):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Mulai dalam {i}...", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.imshow('Pengumpul Dataset', frame)
            cv2.waitKey(1000) # Tunggu 1 detik

        # Loop untuk pengambilan gambar
        count = num_existing_images
        while count < NUM_IMAGES_PER_ALPHABET:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            image_path = os.path.join(path, f'{count}.jpg')
            cv2.imwrite(image_path, frame)
            
            cv2.putText(frame, f"Tersimpan: {count+1}/{NUM_IMAGES_PER_ALPHABET}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Pengumpul Dataset', frame)
            
            count += 1
            time.sleep(0.1)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
        
        if 'key' in locals() and key == ord('q'): break
    if 'key' in locals() and key == ord('q'): break

print("\nPengumpulan data selesai atau dihentikan.")
cap.release()
cv2.destroyAllWindows()