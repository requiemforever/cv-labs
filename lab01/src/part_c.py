import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Шлях до першого фото (де багато дрібних текстур)
image_path = 'lab01/data/IMG_2658.jpeg'
img = cv2.imread(image_path)

if img is None:
    print("Помилка: файл не знайдено.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ==========================================
# Крок 12. Аліасинг[span_2](start_span)[span_2](end_span)
# ==========================================
print("--- Крок 12: Аліасинг ---")
h, w = img_rgb.shape[:2]
new_size_4x = (w // 4, h // 4)

# Зменшення з INTER_NEAREST (тут має з'явитися муар на дрібних текстурах)[span_3](start_span)[span_3](end_span)
img_nearest_4x = cv2.resize(img_rgb, new_size_4x, interpolation=cv2.INTER_NEAREST)

# Попереднє розмиття GaussianBlur + зменшення[span_4](start_span)[span_4](end_span)
img_blur = cv2.GaussianBlur(img_rgb, (9, 9), 2.0)
img_blur_resized = cv2.resize(img_blur, new_size_4x, interpolation=cv2.INTER_NEAREST)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_nearest_4x)
plt.title("Зменшення 4x (INTER_NEAREST)")
plt.subplot(1, 2, 2)
plt.imshow(img_blur_resized)
plt.title("Розмиття + Зменшення 4x (Антиаліасинг)")
plt.suptitle("Крок 12: Аліасинг (шукай муар на дрібних деталях)")
plt.show()

# ==========================================
# Крок 13. Інтерполяція[span_5](start_span)[span_5](end_span)
# ==========================================
print("\n--- Крок 13: Інтерполяція ---")
# Вирізаємо фрагмент високої роздільності (800x800)
# Координати [y1:y2, x1:x2] - беремо центр зображення
crop_high_res = img_rgb[h//2-400 : h//2+400, w//2-400 : w//2+400]

# Зменшуємо його у 8 разів (щоб було з чим працювати)[span_6](start_span)[span_6](end_span)
new_size_8x = (100, 100)
crop_low_res = cv2.resize(crop_high_res, new_size_8x, interpolation=cv2.INTER_AREA)

methods = [
    ("INTER_NEAREST", cv2.INTER_NEAREST),
    ("INTER_LINEAR", cv2.INTER_LINEAR),
    ("INTER_CUBIC", cv2.INTER_CUBIC),
    ("INTER_LANCZOS4", cv2.INTER_LANCZOS4)
]

plt.figure(figsize=(15, 4))
for i, (name, method) in enumerate(methods):
    # Збільшуємо назад у 8 разів[span_7](start_span)[span_7](end_span)
    upscaled = cv2.resize(crop_low_res, (800, 800), interpolation=method)
    
    # Обчислення PSNR відносно оригіналу високої роздільності[span_8](start_span)[span_8](end_span)
    mse = np.mean((crop_high_res.astype(np.float64) - upscaled.astype(np.float64)) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse) if mse != 0 else 100
    
    print(f"Метод: {name:<15} | PSNR = {psnr:.2f} дБ")
    
    plt.subplot(1, 4, i+1)
    plt.imshow(upscaled)
    plt.title(f"{name}\nPSNR: {psnr:.2f} дБ")
    plt.axis('off')

plt.suptitle("Крок 13: Збільшення у 8 разів")
plt.tight_layout()
plt.show()

# ==========================================
# Крок 14. Гамма[span_9](start_span)[span_9](end_span)
# ==========================================
print("\n--- Крок 14: Гамма-корекція ---")
gammas = [0.5, 1.0, 2.2]

plt.figure(figsize=(12, 8))

# Побудова передавальних характеристик (графік)[span_10](start_span)[span_10](end_span)
plt.subplot(2, 3, (1, 3))
x = np.arange(0, 256)
for g in gammas:
    y = np.clip(((x / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    plt.plot(x, y, label=f"Gamma {g}")
plt.title("Передавальні характеристики")
plt.xlabel("Вхідне значення")
plt.ylabel("Вихідне значення")
plt.legend()
plt.grid(True)

# Застосування гамми до зображення[span_11](start_span)[span_11](end_span)
img_small = cv2.resize(img_rgb, (w//4, h//4)) # для швидкості
for i, g in enumerate(gammas):
    lut = np.array([np.clip(((p / 255.0) ** g) * 255.0, 0, 255) for p in range(256)], dtype=np.uint8)
    img_gamma = cv2.LUT(img_small, lut)
    
    plt.subplot(2, 3, 4 + i)
    plt.imshow(img_gamma)
    plt.title(f"Gamma = {g}")
    plt.axis('off')

plt.tight_layout()
plt.show()
