import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import random
import datetime

np.random.seed(42)

# --- Швидке повторне зчитування ---
PATTERN_SIZE = (9, 6)
SQUARE_SIZE = 22.0
objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

images = glob.glob('lab02/data/IMG_*.jpeg')
valid_objpoints = []
valid_imgpoints = []
gray_shape = None
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray_shape is None:
        gray_shape = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
    if ret:
        corners_sub = cv2.cornerSubPix(gray, corners.copy(), (11, 11), (-1, -1), criteria)
        valid_objpoints.append(objp)
        valid_imgpoints.append(corners_sub)

# ==========================================
# Крок 12. Залежність від кількості кадрів
# ==========================================
print("--- Рахуємо залежність від кількості кадрів (це займе трохи часу) ---")
N_values = [5, 8, 10, 12, 15, 20]
fx_results = {N: [] for N in N_values}
err_results = {N: [] for N in N_values}

for N in N_values:
    # Якщо N більше, ніж знайдених кадрів, обмежуємо
    actual_N = min(N, len(valid_objpoints)) 
    for _ in range(5):
        indices = random.sample(range(len(valid_objpoints)), actual_N)
        sub_obj = [valid_objpoints[i] for i in indices]
        sub_img = [valid_imgpoints[i] for i in indices]
        
        ret, mtx, dist, _, _ = cv2.calibrateCamera(sub_obj, sub_img, gray_shape, None, None)
        fx_results[N].append(mtx[0, 0])
        err_results[N].append(ret)

# Графіки
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
N_arr = np.array(N_values)
fx_means = [np.mean(fx_results[N]) for N in N_values]
fx_stds = [np.std(fx_results[N]) for N in N_values]
plt.errorbar(N_arr, fx_means, yerr=fx_stds, fmt='-o', capsize=5, color='blue')
plt.title('Оцінка fx від кількості кадрів N')
plt.xlabel('N (кадрів)')
plt.ylabel('Focal length fx (пікселі)')
plt.grid(True)

plt.subplot(1, 2, 2)
err_means = [np.mean(err_results[N]) for N in N_values]
err_stds = [np.std(err_results[N]) for N in N_values]
plt.errorbar(N_arr, err_means, yerr=err_stds, fmt='-o', color='red', capsize=5)
plt.title('Середня похибка від кількості кадрів N')
plt.xlabel('N (кадрів)')
plt.ylabel('Похибка репроєкції (пікселі)')
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# Повне фінальне калібрування
# ==========================================
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(valid_objpoints, valid_imgpoints, gray_shape, None, None)

# ==========================================
# Крок 13 та 14. Усунення дисторсії
# ==========================================
# Вкажи тут правильну назву свого тестового файлу
test_img_path = 'lab02/data/test_line.jpeg'
test_img = cv2.imread(test_img_path)

if test_img is not None:
    h, w = test_img.shape[:2]
    # alpha = 0 (обрізати краї)
    newcameramtx_0, roi_0 = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
    # alpha = 1 (залишити всі пікселі)
    newcameramtx_1, roi_1 = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    dst_0 = cv2.undistort(test_img, mtx, dist, None, newcameramtx_0)
    dst_1 = cv2.undistort(test_img, mtx, dist, None, newcameramtx_1)
    
    # Накладаємо ідеально рівну червону лінію близько до правого краю для перевірки
    line_x = int(w * 0.9)
    cv2.line(test_img, (line_x, 0), (line_x, h), (0, 0, 255), 10)
    cv2.line(dst_0, (line_x, 0), (line_x, h), (0, 0, 255), 10)
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt.title('Оригінал (видно вигин)')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(dst_0, cv2.COLOR_BGR2RGB))
    plt.title('alpha = 0 (Обрізано)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(dst_1, cv2.COLOR_BGR2RGB))
    plt.title('alpha = 1 (З чорними краями)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print(f"Помилка: не знайдено файл {test_img_path}")

# ==========================================
# Крок 15. Збереження
# ==========================================
fs = cv2.FileStorage('lab02/camera_params.yml', cv2.FILE_STORAGE_WRITE)
fs.write('camera_matrix', mtx)
fs.write('distortion_coefficients', dist)
fs.write('image_size', gray_shape)
fs.write('reprojection_error', ret)
fs.write('frames_used', len(valid_objpoints))
fs.write('calibration_date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
fs.release()
print("\nУспіх! Файл camera_params.yml збережено у папці lab02.")
