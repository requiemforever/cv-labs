import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import urllib.request
import os
import ssl

# --- Обхід проблеми із сертифікатами на macOS ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- Додаємо User-Agent браузера, щоб обійти захист від ботів (Error 403) ---
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')]
urllib.request.install_opener(opener)

# --- Автоматичне завантаження еталонної стереопари Middlebury (Cones) ---
os.makedirs('lab05/data', exist_ok=True)
urls = {
    'left.png': 'https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png',
    'right.png': 'https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png',
    'disp_gt.png': 'https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/disp2.png'
}

for filename, url in urls.items():
    filepath = f'lab05/data/{filename}'
    if not os.path.exists(filepath):
        print(f"Завантаження {filename}...")
        urllib.request.urlretrieve(url, filepath)

# Крок 7. Створення об'єктів для обчислення карти диспаратності
imgL = cv2.imread('lab05/data/left.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('lab05/data/right.png', cv2.IMREAD_GRAYSCALE)

# Крок 8. Сітка експериментів
numDisparities_list = [64, 128, 192]
blockSize_list = [5, 9, 15, 21]
uniquenessRatio_list = [5, 10, 15]

results = []
best_coverage = 0
best_disp_map = None
best_params = None

print(f"{'numDisp':<10} | {'block':<7} | {'unique':<7} | {'Час (с)':<8} | {'Покриття (%)':<12}")
print("-" * 55)

for nd in numDisparities_list:
    for bs in blockSize_list:
        for ur in uniquenessRatio_list:
            
            # Налаштування StereoSGBM
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=nd,
                blockSize=bs,
                uniquenessRatio=ur,
                P1=8 * 1 * bs**2,
                P2=32 * 1 * bs**2,
                disp12MaxDiff=1
            )
            
            start_time = time.time()
            disp_raw = stereo.compute(imgL, imgR).astype(np.float32)
            calc_time = time.time() - start_time
            
            # Крок 7 (Увага): Ділимо на 16.0 для отримання диспаратності в пікселях
            disp = disp_raw / 16.0
            
            # Рахуємо покриття (частка пікселів, для яких алгоритм знайшов відповідник)
            valid_pixels = disp >= 0
            coverage = np.sum(valid_pixels) / disp.size * 100
            
            print(f"{nd:<10} | {bs:<7} | {ur:<7} | {calc_time:<8.3f} | {coverage:<12.1f}")
            
            results.append({
                'numDisp': nd, 'block': bs, 'unique': ur, 
                'time': calc_time, 'coverage': coverage
            })
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_disp_map = disp.copy()
                best_params = (nd, bs, ur)

print("\n--- Найкращі параметри за покриттям ---")
print(f"numDisparities={best_params[0]}, blockSize={best_params[1]}, uniquenessRatio={best_params[2]}")

# Крок 9. Візуалізація найкращої карти з кольоровою шкалою
disp_vis = np.where(best_disp_map < 0, 0, best_disp_map)

plt.figure(figsize=(10, 6))
img = plt.imshow(disp_vis, cmap='plasma') 
cbar = plt.colorbar(img)
cbar.set_label('Пікселі диспаратності')
plt.title(f"Крок 9. Карта диспаратності\n(numDisp={best_params[0]}, block={best_params[1]}, unique={best_params[2]})")
plt.axis('off')
plt.tight_layout()
plt.show()
