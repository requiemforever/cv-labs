import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
img_path = 'lab01/data/IMG_2658.jpeg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    img = cv2.imread('../data/IMG_2658.jpeg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# Детектори з однаковим лімітом[span_19](start_span)[span_19](end_span)
sift = cv2.SIFT_create(nfeatures=1000)
orb = cv2.ORB_create(nfeatures=1000)

def extract_harris_pts(image):
    # Використовуємо бібліотечну функцію для швидкості експерименту
    img_f32 = np.float32(image)
    R = cv2.cornerHarris(img_f32, 3, 3, 0.04)
    pts = np.argwhere(R > 0.01 * R.max())
    scores = R[pts[:, 0], pts[:, 1]]
    pts = pts[np.argsort(scores)[::-1]]
    
    kps = []
    for pt in pts:
        if len(kps) >= 1000: break
        if not kps:
            kps.append(pt)
            continue
        dist = np.linalg.norm(np.array(kps) - pt, axis=1)
        if np.all(dist > 10):
            kps.append(pt)
    # Повертаємо [x, y] для сумісності з SIFT/ORB
    return np.array([[pt[1], pt[0]] for pt in kps], dtype=np.float32)

def evaluate_repeatability(img1, img2, H, eps=3.0):
    metrics = []
    
    # Отримуємо точки оригіналу
    h_pts1 = extract_harris_pts(img1)
    s_pts1 = np.array([kp.pt for kp in sift.detect(img1, None)], dtype=np.float32)
    o_pts1 = np.array([kp.pt for kp in orb.detect(img1, None)], dtype=np.float32)
    
    # Отримуємо точки перетвореного кадру
    h_pts2 = extract_harris_pts(img2)
    s_pts2 = np.array([kp.pt for kp in sift.detect(img2, None)], dtype=np.float32)
    o_pts2 = np.array([kp.pt for kp in orb.detect(img2, None)], dtype=np.float32)
    
    for pts1, pts2 in [(h_pts1, h_pts2), (s_pts1, s_pts2), (o_pts1, o_pts2)]:
        if len(pts1) == 0 or len(pts2) == 0:
            metrics.append(0.0)
            continue
            
        # Крок 15.2: Перенесення точок через H[span_20](start_span)[span_20](end_span)
        pts1_homo = np.hstack([pts1, np.ones((len(pts1), 1))])
        pts1_mapped = (H @ pts1_homo.T).T
        pts1_mapped = pts1_mapped[:, :2] / pts1_mapped[:, 2:]
        
        # Крок 15.3: Рахуємо збіги < 3 пікселів[span_21](start_span)[span_21](end_span)
        matches = 0
        for pt in pts1_mapped:
            dists = np.linalg.norm(pts2 - pt, axis=1)
            if np.min(dists) < eps:
                matches += 1
                
        # Крок 15.4: Метрика[span_22](start_span)[span_22](end_span)
        repeatability = matches / min(len(pts1), len(pts2))
        metrics.append(repeatability)
        
    return metrics # [Harris, SIFT, ORB]

# Крок 16: Сітка експериментів[span_23](start_span)[span_23](end_span)
rotations = np.arange(0, 100, 10)
scales = np.arange(0.5, 2.25, 0.25)
brightness = np.arange(0.5, 1.6, 0.1)
noise_sigma = np.arange(0, 25, 5)

results_rot = []
for theta in rotations:
    M = cv2.getRotationMatrix2D((w/2, h/2), theta, 1.0)
    H = np.vstack([M, [0, 0, 1]]) # Доповнення до 3х3[span_24](start_span)[span_24](end_span)
    img2 = cv2.warpPerspective(img, H, (w, h))
    results_rot.append(evaluate_repeatability(img, img2, H))

results_scl = []
for s in scales:
    H = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float32) #[span_25](start_span)[span_25](end_span)
    img2 = cv2.warpPerspective(img, H, (int(w*s), int(h*s)))
    results_scl.append(evaluate_repeatability(img, img2, H))

results_br = []
for b in brightness:
    img2 = np.clip(img.astype(np.float32) * b, 0, 255).astype(np.uint8)
    results_br.append(evaluate_repeatability(img, img2, np.eye(3)))

results_noise = []
for sigma in noise_sigma:
    noise = np.random.normal(0, sigma, img.shape)
    img2 = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    results_noise.append(evaluate_repeatability(img, img2, np.eye(3)))

# Побудова графіків[span_26](start_span)[span_26](end_span)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
experiments = [
    (rotations, np.array(results_rot), 'Кут повороту (градуси)'),
    (scales, np.array(results_scl), 'Масштаб (s)'),
    (brightness, np.array(results_br), 'Множник яскравості'),
    (noise_sigma, np.array(results_noise), 'Шум Гауса (sigma)')
]

for ax, (x_vals, res, xlabel) in zip(axs.flat, experiments):
    ax.plot(x_vals, res[:, 0] * 100, marker='o', label='Harris')
    ax.plot(x_vals, res[:, 1] * 100, marker='s', label='SIFT')
    ax.plot(x_vals, res[:, 2] * 100, marker='^', label='ORB')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Повторюваність (%)')
    ax.grid(True)
    ax.legend()

plt.suptitle("Крок 16. Залежність повторюваності від перетворень", fontsize=16)
plt.tight_layout()
plt.show()
