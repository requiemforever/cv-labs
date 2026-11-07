import cv2
import numpy as np
import matplotlib.pyplot as plt

# Крок 17. Випадок відмови (Паралакс)
img1 = cv2.imread('lab01/data/fail1.jpeg')
img2 = cv2.imread('lab01/data/fail2.jpeg')

# Робимо кадри однакового розміру, щоб уникнути помилки ValueError
img1 = cv2.resize(img1, (1024, 768))
img2 = cv2.resize(img2, (1024, 768))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.8 * n.distance]

src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

# Шукаємо гомографію
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Зшиваємо
h, w = img1.shape[:2]
warped_img1 = cv2.warpPerspective(img1, H, (w + w//2, h))
panorama_fail = warped_img1.copy()

# Накладаємо друге зображення
mask2 = (panorama_fail[0:h, 0:w] == 0)
panorama_fail[0:h, 0:w] = np.where(mask2, img2, panorama_fail[0:h, 0:w])

plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(panorama_fail, cv2.COLOR_BGR2RGB))
plt.title("Крок 17. Випадок відмови (Ефект паралаксу)")
plt.axis('off')
plt.show()
