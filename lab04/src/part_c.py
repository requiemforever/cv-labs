import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time

np.random.seed(42)

img1_path = 'lab01/data/IMG_2658.jpeg'
img2_path = 'lab01/data/IMG_2659.jpeg'
img1_color = cv2.imread(img1_path)
img2_color = cv2.imread(img2_path)
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
matches_2to1 = bf.match(des2, des1)
mutual_matches = [m for m in good_matches if matches_2to1[m.trainIdx].trainIdx == m.queryIdx]

src_pts = np.float32([kp1[m.queryIdx].pt for m in mutual_matches])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in mutual_matches])

# Крок 8. Реалізація власного RANSAC
def custom_ransac(src, dst, tau, max_iters):
    best_inliers_idx = []
    best_H = None
    src_homo = np.hstack([src, np.ones((len(src), 1))])
    
    for _ in range(max_iters):
        # 1. Випадково вибрати 4 пари
        idx = np.random.choice(len(src), 4, replace=False)
        
        # 2. Оцінити H за цими 4 парами
        try:
            H_est = cv2.getPerspectiveTransform(src[idx], dst[idx])
        except:
            continue
            
        # 3. Перенести всі точки
        proj_pts = (H_est @ src_homo.T).T
        proj_pts = proj_pts[:, :2] / (proj_pts[:, 2:] + 1e-8)
        
        # 4. Порахувати відстані та знайти inliers
        dists = np.linalg.norm(dst - proj_pts, axis=1)
        inliers_idx = np.where(dists < tau)[0]
        
        # 5. Запам'ятати найкращу модель
        if len(inliers_idx) > len(best_inliers_idx):
            best_inliers_idx = inliers_idx
            best_H = H_est
            
    # Переоцінити H за всіма узгодженими точками методом 0 (МНК)
    if len(best_inliers_idx) >= 4:
        final_H, _ = cv2.findHomography(src[best_inliers_idx], dst[best_inliers_idx], 0)
        return final_H, best_inliers_idx
    return best_H, best_inliers_idx

# Крок 9. Теоретична кількість ітерацій
# Припускаємо, що частка викидів (eps) складає 60% (0.6)
eps = 0.6
p = 0.99
s = 4
N_theoretical = math.log(1 - p) / math.log(1 - (1 - eps)**s)
N_iters = int(np.ceil(N_theoretical))
print("--- Крок 9. Кількість ітерацій ---")
print(f"Розрахована кількість ітерацій (N): {N_iters}")

# Крок 10. Звірка з cv2.findHomography
tau_work = 3.0

start_t = time.time()
H_custom, inl_custom = custom_ransac(src_pts, dst_pts, tau_work, N_iters)
time_custom = time.time() - start_t

start_t = time.time()
H_lib, mask_lib = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, tau_work)
time_lib = time.time() - start_t
inl_lib = np.where(mask_lib.ravel() == 1)[0]

def calc_error(H, src, dst, inl):
    if H is None or len(inl) == 0: return 0
    src_h = np.hstack([src[inl], np.ones((len(inl), 1))])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / (proj[:, 2:] + 1e-8)
    return np.mean(np.linalg.norm(dst[inl] - proj, axis=1))

err_custom = calc_error(H_custom, src_pts, dst_pts, inl_custom)
err_lib = calc_error(H_lib, src_pts, dst_pts, inl_lib)

print("\n--- Крок 10. Звірка ---")
print(f"Власний RANSAC: узгоджених={len(inl_custom)}, похибка={err_custom:.3f} px, час={time_custom:.3f} с")
print(f"cv2.RANSAC:     узгоджених={len(inl_lib)}, похибка={err_lib:.3f} px, час={time_lib:.3f} с")

# Крок 11. Дослідження порога
taus = [1, 2, 3, 5, 8, 12, 20]
counts = []
errors = []

for t in taus:
    Ht, inl = custom_ransac(src_pts, dst_pts, t, N_iters)
    counts.append(len(inl))
    errors.append(calc_error(Ht, src_pts, dst_pts, inl))

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_xlabel('Поріг tau (пікселі)')
ax1.set_ylabel('Кількість узгоджених точок', color='blue')
ax1.plot(taus, counts, marker='o', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel('Середня похибка репроєкції (пікселі)', color='red')
ax2.plot(taus, errors, marker='s', color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.title('Крок 11. Залежність показників від порога RANSAC')
fig.tight_layout()
plt.show()

# Крок 12. Візуалізація
mask_bytes = np.zeros(len(mutual_matches), dtype=np.uint8)
mask_bytes[inl_custom] = 1

img_ransac = cv2.drawMatches(img1_color, kp1, img2_color, kp2, mutual_matches, None, 
                             matchColor=(0, 255, 0), singlePointColor=None,
                             matchesMask=mask_bytes.tolist(), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 6))
plt.imshow(cv2.cvtColor(img_ransac, cv2.COLOR_BGR2RGB))
plt.title("Крок 12. Узгоджені збіги (тільки Inliers)")
plt.axis('off')
plt.show()
