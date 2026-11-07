import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(42)
# Використовуємо один із твоїх кадрів
image_path = 'lab01/data/IMG_2658.jpeg'  
img_color = cv2.imread(image_path)
if img_color is None:
    # Шлях може відрізнятися залежно від того, звідки ти запускаєш (можливо '../data/IMG_2658.jpeg')
    img_color = cv2.imread('../data/IMG_2658.jpeg')

# Крок 1: Переведення у відтінки сірого й у float32[span_3](start_span)[span_3](end_span)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_float = np.float32(img_gray)

def custom_harris(img_f32, k=0.04, threshold_ratio=0.01, nms_radius=10, max_points=1000):
    # Крок 2: Похідні Ix, Iy (Sobel, ядро 3)[span_4](start_span)[span_4](end_span)
    Ix = cv2.Sobel(img_f32, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img_f32, cv2.CV_32F, 0, 1, ksize=3)
    
    # Крок 3: Поелементні добутки (до згладжування!)[span_5](start_span)[span_5](end_span)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # Крок 4: Згладжування гауссіаною sigma=1.5[span_6](start_span)[span_6](end_span)
    Sxx = cv2.GaussianBlur(Ixx, (0, 0), sigmaX=1.5)
    Syy = cv2.GaussianBlur(Iyy, (0, 0), sigmaX=1.5)
    Sxy = cv2.GaussianBlur(Ixy, (0, 0), sigmaX=1.5)
    
    # Крок 5: Відгук R[span_7](start_span)[span_7](end_span)
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M ** 2)
    
    # Крок 6: Поріг[span_8](start_span)[span_8](end_span)
    threshold = threshold_ratio * R.max()
    pts = np.argwhere(R > threshold)
    scores = R[pts[:, 0], pts[:, 1]]
    
    # Крок 7: Немаксимальне придушення (NMS)[span_9](start_span)[span_9](end_span)
    sorted_idx = np.argsort(scores)[::-1]
    pts = pts[sorted_idx]
    
    keypoints = []
    for pt in pts:
        if len(keypoints) >= max_points:
            break
        if not keypoints:
            keypoints.append(pt)
            continue
        
        # Перевірка відстані до вже доданих точок[span_10](start_span)[span_10](end_span)
        dist = np.linalg.norm(np.array(keypoints) - pt, axis=1)
        if np.all(dist > nms_radius):
            keypoints.append(pt)
            
    return np.array(keypoints) # Формат [y, x]

print("--- Частина А. Власна реалізація Гарріса ---")
start_t = time.time()
custom_kps = custom_harris(img_float)
print(f"Власний Гарріс: знайдено {len(custom_kps)} точок за {time.time()-start_t:.3f} с.")

# Крок 8: Звірка з бібліотечною функцією[span_11](start_span)[span_11](end_span)
# Параметри: blockSize=3 (аналог згладжування), ksize=3 (Sobel), k=0.04
lib_R = cv2.cornerHarris(img_float, blockSize=3, ksize=3, k=0.04)
lib_pts = np.argwhere(lib_R > 0.01 * lib_R.max())
lib_scores = lib_R[lib_pts[:, 0], lib_pts[:, 1]]
lib_pts = lib_pts[np.argsort(lib_scores)[::-1]][:1000] # Беремо топ-1000

# Рахуємо збіг (відстань < 2 пікселів)[span_12](start_span)[span_12](end_span)
matches = 0
for cpt in custom_kps:
    dists = np.linalg.norm(lib_pts - cpt, axis=1)
    if np.min(dists) < 2.0:
        matches += 1
print(f"Збіг із бібліотечною функцією: {matches / len(custom_kps) * 100:.1f}% (очікується > 90%)\n")

print("--- Частина В. Чутливість до масштабу ---")
# Крок 9-10: Зменшення в 2 рази[span_13](start_span)[span_13](end_span)
h, w = img_float.shape
img_scaled = cv2.resize(img_float, (w // 2, h // 2))
scaled_kps = custom_harris(img_scaled)

# Масштабуємо назад і шукаємо відповідники[span_14](start_span)[span_14](end_span)
mapped_kps = scaled_kps * 2
scale_matches = 0
for mkp in mapped_kps:
    dists = np.linalg.norm(custom_kps - mkp, axis=1)
    if np.min(dists) < 3.0:
        scale_matches += 1
print(f"Повторюваність Гарріса при зміні масштабу: {scale_matches / len(custom_kps) * 100:.1f}%\n")

print("--- Частина C. SIFT і ORB ---")
# Крок 11-12: Створення та обчислення[span_15](start_span)[span_15](end_span)
sift = cv2.SIFT_create(nfeatures=1000)
orb = cv2.ORB_create(nfeatures=1000)

for name, detector in [("SIFT", sift), ("ORB", orb)]:
    t0 = time.time()
    kps, des = detector.detectAndCompute(img_gray, None)
    t1 = time.time()
    print(f"{name}: точок = {len(kps)}, розмірність = {des.shape[1] if des is not None else 0}, тип = {des.dtype if des is not None else 'N/A'}, час = {t1-t0:.3f} с.")

# Крок 13: Візуалізація[span_16](start_span)[span_16](end_span)
kps_sift, _ = sift.detectAndCompute(img_gray, None)
img_sift = cv2.drawKeypoints(img_color, kps_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
plt.title("SIFT Keypoints (DRAW_RICH_KEYPOINTS)")
plt.axis('off')
plt.show()

# --- Додаткова візуалізація для звіту (Частини А і В) ---
# 1. Візуалізація точок Гарріса на оригіналі
img_harris_a = img_color.copy()
for pt in custom_kps:
    cv2.circle(img_harris_a, (int(pt[1]), int(pt[0])), 3, (0, 255, 0), -1)

# 2. Візуалізація точок на зменшеному кадрі
img_harris_b = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2BGR)
for pt in scaled_kps:
    cv2.circle(img_harris_b, (int(pt[1]), int(pt[0])), 2, (0, 0, 255), -1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_harris_a, cv2.COLOR_BGR2RGB))
plt.title("Частина А: Точки Гарріса (Оригінал)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_harris_b, cv2.COLOR_BGR2RGB))
plt.title("Частина В: Точки Гарріса (Зменшено x0.5)")
plt.axis('off')
plt.show()

