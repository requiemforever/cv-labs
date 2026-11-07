import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ExifTags

# Фіксація випадкового стану для відтворюваності[span_3](start_span)[span_3](end_span)
np.random.seed(42)

# Шлях до твого зображення (відносно папки cv-labs)
image_path = 'lab01/data/IMG_2658.jpeg'

# ==========================================
# КРОК 1. Зчитування та базова інформація[span_4](start_span)[span_4](end_span)
# ==========================================
img = cv2.imread(image_path)

if img is None:
    print(f"Помилка: не вдалося знайти фото за шляхом {image_path}")
    exit()

print("--- Крок 1: Інформація про зображення ---")
print(f"Форма (shape): {img.shape}")
print(f"Тип даних (dtype): {img.dtype}")
print(f"Мінімум: {img.min()}, Максимум: {img.max()}\n")

# ==========================================
# КРОК 2. Відображення (BGR vs RGB)[span_5](start_span)[span_5](end_span)
# ==========================================
plt.figure(figsize=(10, 5))

# Без перетворення (BGR)
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Без перетворення (BGR)")

# З перетворенням (RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(1, 2, 2)
plt.imshow(img_rgb)
plt.title("З перетворенням (RGB)")

plt.suptitle("Крок 2: Відображення зображень")
plt.show() # Закрий вікно графіка, щоб скрипт пішов далі

# ==========================================
# КРОК 3. Переповнення типів (Overflow)[span_6](start_span)[span_6](end_span)
# ==========================================
# Некоректне додавання (арифметика над uint8 без приведення)[span_7](start_span)[span_7](end_span)[span_8](start_span)[span_8](end_span)
bright_incorrect = img_rgb + 50

# Коректне додавання через cv2.add (автоматично обрізає до 255)[span_9](start_span)[span_9](end_span)
bright_correct = cv2.add(img_rgb, 50)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(bright_incorrect)
plt.title("Некоректно (img + 50)")

plt.subplot(1, 2, 2)
plt.imshow(bright_correct)
plt.title("Коректно (cv2.add)")

plt.suptitle("Крок 3: Переповнення типів")
plt.show() # Закрий вікно графіка, щоб скрипт пішов далі

# ==========================================
# КРОК 4. Втрати при стисненні (PSNR)[span_10](start_span)[span_10](end_span)
# ==========================================
print("--- Крок 4: Втрати при стисненні ---")
# Зберігаємо еталонний PNG у папку результатів
png_path = 'lab01/results/reference.png'
cv2.imwrite(png_path, img)
original_size = os.path.getsize(png_path) / 1024 # Розмір у КБ

qualities = [95, 75, 50, 25, 10]
sizes = []
psnr_values = []

# Формула PSNR: 10 * log10(255^2 / MSE)[span_11](start_span)[span_11](end_span)
for q in qualities:
    jpg_path = f'lab01/results/compressed_{q}.jpg'
    # Збереження з певною якістю[span_12](start_span)[span_12](end_span)
    cv2.imwrite(jpg_path, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    
    # Зчитування назад
    compressed_img = cv2.imread(jpg_path)
    size_kb = os.path.getsize(jpg_path) / 1024
    
    # Обчислення MSE та PSNR
    mse = np.mean((img.astype(np.float64) - compressed_img.astype(np.float64)) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)
        
    sizes.append(size_kb)
    psnr_values.append(psnr)
    print(f"Якість: {q}, Розмір: {size_kb:.2f} КБ, PSNR: {psnr:.2f} дБ")

# Побудова графіка PSNR від розміру файлу[span_13](start_span)[span_13](end_span)
plt.figure(figsize=(8, 5))
plt.plot(sizes, psnr_values, marker='o', linestyle='-', color='b')
plt.title("Залежність PSNR від розміру файлу")
plt.xlabel("Розмір файлу (КБ)")
plt.ylabel("PSNR (дБ)")
plt.grid(True)
plt.show()

# ==========================================
# КРОК 5. EXIF дані[span_14](start_span)[span_14](end_span)
# ==========================================
print("\n--- Крок 5: EXIF дані ---")
try:
    pil_img = Image.open(image_path)
    exif = pil_img._getexif()
    if exif is not None:
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            # Шукаємо ISO (ISOSpeedRatings), витримку (ExposureTime), фокусну відстань (FocalLength)[span_15](start_span)[span_15](end_span)
            if tag in ['ISOSpeedRatings', 'ExposureTime', 'FocalLength']:
                print(f"{tag}: {value}")
    else:
        print("EXIF дані відсутні у цьому файлі (можливо, втрачені при конвертації).")
except Exception as e:
    print(f"Помилка зчитування EXIF: {e}")
