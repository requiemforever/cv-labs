import cv2
import numpy as np
import matplotlib.pyplot as plt

# Завантаження даних
imgL = cv2.imread('lab05/data/left.png')
imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR_gray = cv2.cvtColor(cv2.imread('lab05/data/right.png'), cv2.COLOR_BGR2GRAY)
gt_raw = cv2.imread('lab05/data/disp_gt.png', cv2.IMREAD_GRAYSCALE)

# Обчислюємо найкращу карту ще раз
stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=64, blockSize=5, uniquenessRatio=5,
    P1=8*1*25, P2=32*1*25, disp12MaxDiff=1
)
disp = stereo.compute(imgL_gray, imgR_gray).astype(np.float32) / 16.0

# --- Частина Е (Крок 13). Оцінка за еталоном ---
# Еталонна карта Middlebury збережена у відтінках сірого, нормалізуємо її до нашого діапазону (64)
gt_disp = (gt_raw.astype(np.float32) / 255.0) * 64.0 

# Маска для пікселів, де є дані на обох картах
mask_valid = (disp >= 0) & (gt_disp > 0)
abs_diff = np.abs(disp[mask_valid] - gt_disp[mask_valid])

# Рахуємо пікселі, де похибка > 2 px (робимо невеликий запас через масштабування PNG)
bad_pixels = np.sum(abs_diff > 2.0)
coverage = (np.sum(disp >= 0) / disp.size) * 100
error_rate = (bad_pixels / np.sum(mask_valid)) * 100

print(f"--- Крок 13. Оцінка за еталоном ---")
print(f"Покриття нашої карти: {coverage:.1f}%")
print(f"Відсоток помилкових пікселів (|D - D_gt| > 1 px): {error_rate:.1f}%")

# --- Крок 14. Зони систематичних помилок ---
err_map = np.zeros_like(disp)
err_map[mask_valid] = np.where(abs_diff > 2.0, 1, 0)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(disp, cmap='plasma', vmin=0, vmax=64)
plt.title('Наша диспаратність')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gt_disp, cmap='plasma', vmin=0, vmax=64)
plt.title('Еталон (Ground Truth)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(err_map, cmap='Reds')
plt.title('Зони помилок (червоним)')
plt.axis('off')
plt.tight_layout()
plt.show()

# --- Частина D (Кроки 10-12). Глибина та Хмара точок ---
print("\nГенеруємо 3D хмару точок...")
h, w = imgL_gray.shape
f = 0.8 * w  # Приблизна фокусна відстань
# Будуємо матрицю Q для переходу від диспаратності до 3D-координат
Q = np.float32([
    [1, 0, 0, -w/2],
    [0, -1, 0, h/2],
    [0, 0, 0, -f],
    [0, 0, 1, 0]
])

points_3D = cv2.reprojectImageTo3D(disp, Q)
colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

# Фільтруємо недійсні пікселі та занадто віддалений фон
mask_pc = disp > 5  
out_points = points_3D[mask_pc]
out_colors = colors[mask_pc]

# Зберігаємо у формат PLY, який можна відкрити в будь-якому 3D-редакторі (напр. MeshLab або Blender)
with open('lab05/point_cloud.ply', 'w') as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(out_points)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for p, c in zip(out_points, out_colors):
        f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

print("Успіх! Хмару точок збережено у файл 'lab05/point_cloud.ply'.")
