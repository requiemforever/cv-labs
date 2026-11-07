import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Параметри шаблону
PATTERN_SIZE = (9, 6) # Внутрішні кути[span_4](start_span)[span_4](end_span)
SQUARE_SIZE = 22.0    # Міліметри[span_5](start_span)[span_5](end_span)

# Крок 8: Масив тривимірних координат (Z завжди 0)[span_6](start_span)[span_6](end_span)
objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = [] # 3D точки реального світу
imgpoints_raw = [] # 2D точки без уточнення
imgpoints_subpix = [] # 2D точки з уточненням
valid_images = []

images = glob.glob('lab02/data/IMG_*.jpeg')
if not images:
    print("Помилка: зображення не знайдено.")
    exit()

print(f"Знайдено файлів для обробки: {len(images)}")
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Крок 6: Пошук кутів[span_7](start_span)[span_7](end_span)
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
    if ret:
        objpoints.append(objp)
        imgpoints_raw.append(corners)
        
        # Крок 7: Уточнення до субпіксельної точності[span_8](start_span)[span_8](end_span)
        corners_sub = cv2.cornerSubPix(gray, corners.copy(), (11, 11), (-1, -1), criteria)
        imgpoints_subpix.append(corners_sub)
        valid_images.append(fname)
    else:
        print(f"Кути не знайдено на кадрі: {fname} (відкладено)[span_9](start_span)[span_9](end_span)")

print(f"Успішно знайдено кути на {len(valid_images)} кадрах.")

img_shape = gray.shape[::-1]

# Крок 7 та 9: Калібрування і порівняння похибки[span_10](start_span)[span_10](end_span)
ret_raw, mtx_raw, dist_raw, _, _ = cv2.calibrateCamera(objpoints, imgpoints_raw, img_shape, None, None)
ret_sub, mtx_sub, dist_sub, rvecs_sub, tvecs_sub = cv2.calibrateCamera(objpoints, imgpoints_subpix, img_shape, None, None)

print(f"\n--- Крок 7: Вплив субпіксельного уточнення ---")
print(f"Загальна похибка БЕЗ уточнення: {ret_raw:.4f} пікселів")
print(f"Загальна похибка З уточненням:  {ret_sub:.4f} пікселів[span_11](start_span)[span_11](end_span)")

# Крок 10: Похибка по кадрах (самостійний розрахунок)[span_12](start_span)[span_12](end_span)
def calculate_errors(obj_pts, img_pts, rvecs, tvecs, mtx, dist):
    errors = []
    for i in range(len(obj_pts)):
        img_pts_proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
        # Середня відстань між виявленими та спроєктованими кутами для одного кадру[span_13](start_span)[span_13](end_span)
        dist_per_point = np.linalg.norm(img_pts[i] - img_pts_proj, axis=2)
        errors.append(np.mean(dist_per_point))
    return np.array(errors)

errors = calculate_errors(objpoints, imgpoints_subpix, rvecs_sub, tvecs_sub, mtx_sub, dist_sub)

plt.figure(figsize=(12, 5))
plt.bar(range(len(errors)), errors, color='skyblue', edgecolor='black')
plt.axhline(np.median(errors), color='green', linestyle='dashed', label='Медіана')
plt.axhline(2 * np.median(errors), color='red', linestyle='dashed', label='Поріг (2 * Медіана)[span_14](start_span)[span_14](end_span)')
plt.title('Крок 10: Похибка репроєкції по кадрах')
plt.xlabel('Індекс кадру')
plt.ylabel('Середня похибка (пікселі)')
plt.legend()
plt.tight_layout()
plt.show()

# Крок 11: Вилучення викидів[span_15](start_span)[span_15](end_span)
median_err = np.median(errors)
threshold = 2 * median_err

objpoints_filtered = []
imgpoints_filtered = []
valid_images_filtered = []

for i in range(len(errors)):
    if errors[i] <= threshold:
        objpoints_filtered.append(objpoints[i])
        imgpoints_filtered.append(imgpoints_subpix[i])
        valid_images_filtered.append(valid_images[i])

removed_count = len(errors) - len(objpoints_filtered)
print(f"\n--- Крок 11: Фільтрація викидів ---")
print(f"Критерій вилучення: похибка > {threshold:.4f} (подвійна медіана)[span_16](start_span)[span_16](end_span)")
print(f"Вилучено кадрів: {removed_count}")

# Перерахунок калібрування на чистому наборі даних[span_17](start_span)[span_17](end_span)
ret_final, mtx_final, dist_final, rvecs_final, tvecs_final = cv2.calibrateCamera(
    objpoints_filtered, imgpoints_filtered, img_shape, None, None
)

print(f"Фінальна загальна похибка після фільтрації: {ret_final:.4f} пікселів")
print("\nГоловна точка (cx, cy):", mtx_final[0, 2], mtx_final[1, 2])
print(f"Відношення fx до fy: {mtx_final[0, 0] / mtx_final[1, 1]:.4f}")
