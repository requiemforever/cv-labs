import cv2
import numpy as np
import matplotlib.pyplot as plt

# Крок 1. Завантаження двох сусідніх кадрів (з перекриттям ~40%)[span_4](start_span)[span_4](end_span)
img1_path = 'lab01/data/IMG_2658.jpeg'
img2_path = 'lab01/data/IMG_2659.jpeg'

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
img1_color = cv2.imread(img1_path)
img2_color = cv2.imread(img2_path)

if img1 is None or img2 is None:
    print("Помилка: не вдалося завантажити зображення. Перевір шляхи.")
    exit()

# Обчислення SIFT-дескрипторів[span_5](start_span)[span_5](end_span)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Крок 2. Створення cv2.BFMatcher з нормою NORM_L2 та виклик knnMatch з k=2[span_6](start_span)[span_6](end_span)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Крок 3. Критерій відношення відстаней[span_7](start_span)[span_7](end_span)
print("--- Крок 3. Критерій відношення відстаней (Lowe's Ratio Test) ---")
T_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
for T in T_values:
    good_T = [m for m, n in matches if m.distance < T * n.distance]
    print(f"Для T = {T}: знайдено {len(good_T)} кандидатів")

# Робоче значення T для наступних кроків (зазвичай 0.7 або 0.8)
T_work = 0.8
good_matches = [m for m, n in matches if m.distance < T_work * n.distance]

# Крок 4. Взаємна перевірка (Cross-check)[span_8](start_span)[span_8](end_span)
matches_2to1 = bf.match(des2, des1)
mutual_matches = []
for m in good_matches:
    # m.queryIdx - індекс точки на img1, m.trainIdx - індекс точки на img2
    best_match_reverse = matches_2to1[m.trainIdx]
    if best_match_reverse.trainIdx == m.queryIdx:
        mutual_matches.append(m)

print("\n--- Крок 4. Взаємна перевірка ---")
print(f"Після взаємної перевірки (із T=0.8) лишилося кандидатів: {len(mutual_matches)}[span_9](start_span)[span_9](end_span)")

# Крок 5. Візуалізація сирих кандидатів[span_10](start_span)[span_10](end_span)
# Малюємо перші 100 збігів, щоб картинка не перетворилася на суцільну кашу, але помилки було видно
img_matches = cv2.drawMatches(img1_color, kp1, img2_color, kp2, mutual_matches[:100], None, 
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 6))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title("Крок 5. Сирі кандидати (явно видно хибні перехресні лінії)[span_11](start_span)[span_11](end_span)")
plt.axis('off')
plt.show()

# --- Частина В. Чому потрібен робастний метод ---

# Підготовка координат точок для знайдених збігів
src_pts = np.float32([kp1[m.queryIdx].pt for m in mutual_matches])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in mutual_matches])

# Крок 6. Оцінка гомографії методом найменших квадратів (0 = звичайний МНК без відсіву)[span_12](start_span)[span_12](end_span)
H_bad, _ = cv2.findHomography(src_pts, dst_pts, 0)

# Крок 7. Застосування гомографії та обчислення похибки[span_13](start_span)[span_13](end_span)
h, w = img1.shape
img1_warped = cv2.warpPerspective(img1_color, H_bad, (w, h))

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img1_warped, cv2.COLOR_BGR2RGB))
plt.title("Крок 7. Спотворений кадр (Результат МНК без RANSAC)[span_14](start_span)[span_14](end_span)")
plt.axis('off')
plt.show()

# Обчислення середньої похибки репроєкції[span_15](start_span)[span_15](end_span)
src_pts_homo = np.hstack([src_pts, np.ones((len(src_pts), 1))])
proj_pts = (H_bad @ src_pts_homo.T).T
proj_pts = proj_pts[:, :2] / proj_pts[:, 2:]

errors = np.linalg.norm(dst_pts - proj_pts, axis=1)
mean_error = np.mean(errors)

print("\n--- Частина В. Чому потрібен робастний метод ---")
print(f"Середня похибка репроєкції (звичайний МНК): {mean_error:.2f} пікселів[span_16](start_span)[span_16](end_span)")
print("Похибка складає десятки або сотні пікселів через наявність викидів (хибних збігів)[span_17](start_span)[span_17](end_span).")
