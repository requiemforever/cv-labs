import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Шляхи до файлів
img1_path = 'lab01/data/IMG_2680.jpeg'
img2_path = 'lab01/data/IMG_2681.jpeg'
mask_path = 'lab01/data/IMG_2682.png'

# Зчитуємо зображення
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None or ground_truth is None:
    print("Помилка зчитування файлів. Перевір шляхи.")
    exit()

# Вирівнюємо розмір еталонної маски ТА другого фото під розмір першого фото
ground_truth = cv2.resize(ground_truth, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Конвертація в RGB для Matplotlib
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Еталонна маска: робимо бінарною (0 або 255)
_, ground_truth_bin = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)

# Функція для розрахунку IoU
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
    return iou

# Функція для розрахунку частки правильно класифікованих пікселів
def calculate_accuracy(pred_mask, gt_mask):
    correct = np.sum(pred_mask == gt_mask)
    total = gt_mask.size
    return correct / total

# --- Крок 8. Порогування в RGB ---
lower_rgb = np.array([0, 100, 0])
upper_rgb = np.array([120, 255, 100])

mask1_rgb = cv2.inRange(img1_rgb, lower_rgb, upper_rgb)
mask2_rgb = cv2.inRange(img2_rgb, lower_rgb, upper_rgb)

acc1_rgb = calculate_accuracy(mask1_rgb, ground_truth_bin)
iou1_rgb = calculate_iou(mask1_rgb, ground_truth_bin)

acc2_rgb = calculate_accuracy(mask2_rgb, ground_truth_bin)
iou2_rgb = calculate_iou(mask2_rgb, ground_truth_bin)

print("--- Кроки 8-9: RGB ---")
print(f"Кадр 1 (RGB): Accuracy = {acc1_rgb:.4f}, IoU = {iou1_rgb:.4f}")
print(f"Кадр 2 (RGB): Accuracy = {acc2_rgb:.4f}, IoU = {iou2_rgb:.4f}")

# --- Крок 10. Те саме в HSV ---
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

lower_hsv = np.array([35, 50, 50])
upper_hsv = np.array([85, 255, 255])

mask1_hsv = cv2.inRange(img1_hsv, lower_hsv, upper_hsv)
mask2_hsv = cv2.inRange(img2_hsv, lower_hsv, upper_hsv)

acc1_hsv = calculate_accuracy(mask1_hsv, ground_truth_bin)
iou1_hsv = calculate_iou(mask1_hsv, ground_truth_bin)

acc2_hsv = calculate_accuracy(mask2_hsv, ground_truth_bin)
iou2_hsv = calculate_iou(mask2_hsv, ground_truth_bin)

print("\n--- Крок 10: HSV ---")
print(f"Кадр 1 (HSV): Accuracy = {acc1_hsv:.4f}, IoU = {iou1_hsv:.4f}")
print(f"Кадр 2 (HSV): Accuracy = {acc2_hsv:.4f}, IoU = {iou2_hsv:.4f}")

# Візуалізація
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(img1_rgb)
plt.title("Кадр 1 (Оригінал)")

plt.subplot(2, 3, 2)
plt.imshow(mask1_rgb, cmap='gray')
plt.title(f"RGB Маска 1 (IoU: {iou1_rgb:.2f})")

plt.subplot(2, 3, 3)
plt.imshow(mask1_hsv, cmap='gray')
plt.title(f"HSV Маска 1 (IoU: {iou1_hsv:.2f})")

plt.subplot(2, 3, 4)
plt.imshow(img2_rgb)
plt.title("Кадр 2 (Оригінал)")

plt.subplot(2, 3, 5)
plt.imshow(mask2_rgb, cmap='gray')
plt.title(f"RGB Маска 2 (IoU: {iou2_rgb:.2f})")

plt.subplot(2, 3, 6)
plt.imshow(mask2_hsv, cmap='gray')
plt.title(f"HSV Маска 2 (IoU: {iou2_hsv:.2f})")

plt.tight_layout()
plt.show()
