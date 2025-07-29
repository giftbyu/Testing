import cv2
import os
import shutil


SOURCE_DATA_DIR = './dataset/BISINDO'  
OUTPUT_DIR_NAME = './dataset_augmented_bisindo'


if os.path.exists(OUTPUT_DIR_NAME):
    print(f"Direktori '{OUTPUT_DIR_NAME}' sudah ada. Harap hapus atau ganti nama sebelum melanjutkan.")
    exit()

os.makedirs(OUTPUT_DIR_NAME, exist_ok=True)
print(f"Direktori output '{OUTPUT_DIR_NAME}' telah dibuat.")


print("Menyalin file asli...")
shutil.copytree(SOURCE_DATA_DIR, os.path.join(OUTPUT_DIR_NAME, os.path.basename(SOURCE_DATA_DIR)))


print("Membuat versi gambar yang di-flip...")
for label in sorted(os.listdir(SOURCE_DATA_DIR)):
    label_dir_source = os.path.join(SOURCE_DATA_DIR, label)
    if not os.path.isdir(label_dir_source):
        continue

    print(f"  Memproses folder: {label}")
    

    for image_name in os.listdir(label_dir_source):
        image_path = os.path.join(label_dir_source, image_name)
        
        try:
            image = cv2.imread(image_path)
            if image is None: continue
            
            flipped_image = cv2.flip(image, 1)
            flipped_image_name = f"flipped_{image_name}"

            output_label_dir = os.path.join(OUTPUT_DIR_NAME, os.path.basename(SOURCE_DATA_DIR), label)
            os.makedirs(output_label_dir, exist_ok=True)
            output_path = os.path.join(output_label_dir, flipped_image_name)
            
            cv2.imwrite(output_path, flipped_image)
            
        except Exception as e:
            print(f"    Gagal memproses {image_name}: {e}")

print("\nProses augmentasi mirroring selesai!")
print(f"Dataset baru Anda yang lebih besar dan tangguh sekarang ada di: '{OUTPUT_DIR_NAME}'")
print("Gunakan direktori ini sebagai DATA_DIR di notebook pelatihan Anda.")