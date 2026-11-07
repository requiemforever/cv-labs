import cv2
import numpy as np
import matplotlib.pyplot as plt

# Крок 13. Завантаження серії з 5 кадрів
image_paths = [
    'lab01/data/IMG_2658.jpeg',
    'lab01/data/IMG_2659.jpeg',
    'lab01/data/IMG_2660.jpeg', # Опорний кадр (індекс 2)
    'lab01/data/IMG_2661.jpeg',
    'lab01/data/IMG_2662.jpeg'
]

images = [cv2.imread(p) for p in image_paths]
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
ANCHOR_IDX = 2

# Знаходимо ключові точки для всіх кадрів
sift = cv2.SIFT_create()
kps_descs = [sift.detectAndCompute(gray, None) for gray in gray_images]

def get_pairwise_homography(idx1, idx2):
    kp1, des1 = kps_descs[idx1]
    kp2, des2 = kps_descs[idx2]
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.8 * n.distance]
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    
    # Крок 16. Оцінка якості (середня похибка репроєкції на узгоджених точках)
    inliers = np.where(mask.ravel() == 1)[0]
    src_h = np.hstack([src_pts[inliers], np.ones((len(inliers), 1))])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / (proj[:, 2:] + 1e-8)
    error = np.mean(np.linalg.norm(dst_pts[inliers] - proj, axis=1))
    
    return H, error

# Обчислення попарних гомографій
print("--- Обчислення гомографій ---")
H_01, err_01 = get_pairwise_homography(0, 1)
H_12, err_12 = get_pairwise_homography(1, 2)
H_32, err_32 = get_pairwise_homography(3, 2)
H_43, err_43 = get_pairwise_homography(4, 3)

print(f"Похибка 0->1: {err_01:.3f} px")
print(f"Похибка 1->2: {err_12:.3f} px")
print(f"Похибка 3->2: {err_32:.3f} px")
print(f"Похибка 4->3: {err_43:.3f} px")
mean_total_error = np.mean([err_01, err_12, err_32, err_43])
print(f"Загальна середня похибка репроєкції панорами: {mean_total_error:.3f} px")

# Перемноження гомографій для зведення до опорного кадру
H_to_anchor = [None] * 5
H_to_anchor[2] = np.eye(3)                  # Опорний
H_to_anchor[1] = H_12                       # 1 -> 2
H_to_anchor[0] = H_12 @ H_01                # 0 -> 1 -> 2
H_to_anchor[3] = H_32                       # 3 -> 2
H_to_anchor[4] = H_32 @ H_43                # 4 -> 3 -> 2

# Крок 14. Розмір полотна
h, w = images[0].shape[:2]
corners = np.array([
    [0, 0], [w, 0], [w, h], [0, h]
], dtype=np.float32)

all_corners = []
for i in range(5):
    corn_h = np.hstack([corners, np.ones((4, 1))])
    proj_corn = (H_to_anchor[i] @ corn_h.T).T
    proj_corn = proj_corn[:, :2] / proj_corn[:, 2:]
    all_corners.append(proj_corn)

all_corners = np.vstack(all_corners)
x_min, y_min = np.int32(all_corners.min(axis=0) - 0.5)
x_max, y_max = np.int32(all_corners.max(axis=0) + 0.5)

translation = np.array([
    [1, 0, -x_min],
    [0, 1, -y_min],
    [0, 0, 1]
], dtype=np.float32)

canvas_w = x_max - x_min
canvas_h = y_max - y_min

# Крок 15. Перенесення кадрів на полотно
panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# Порядок накладання (від країв до центру)
render_order = [0, 4, 1, 3, 2] 

for i in render_order:
    H_final = translation @ H_to_anchor[i]
    warped_img = cv2.warpPerspective(images[i], H_final, (canvas_w, canvas_h))
    
    # Просте накладання (запис пікселів, де панорама ще чорна)
    mask = (panorama == 0)
    panorama[mask] = warped_img[mask]

# Збереження і візуалізація
cv2.imwrite('lab04/panorama_result.jpeg', panorama)

plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.title(f"Крок 15. Готова панорама (Об'єктивна міра якості: {mean_total_error:.3f} px)")
plt.axis('off')
plt.show()
