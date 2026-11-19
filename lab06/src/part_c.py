import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

os.makedirs("lab06/output", exist_ok=True)


def run_dense_optical_flow(video_path, output_video_path, sample_img_path):
    """Частина C (Кроки 8–10). Щільний оптичний потік Farneback."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Помилка: не вдалося відкрити {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    ret, prev_frame = cap.read()
    if not ret:
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Підготовка HSV-зображення для візуалізації (Крок 9)
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255  # Максимальна насиченість (S)

    frame_times = []
    best_mag_mean = -1.0
    best_sample_orig = None
    best_sample_flow = None
    best_frame_idx = 0
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t_start = time.perf_counter()

        # Крок 8. Обчислення щільного потоку методом Farneback
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        frame_times.append((time.perf_counter() - t_start) * 1000.0)

        # Крок 9. Переведення в HSV: напрям -> H, величина -> V (після нормалізації)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out.write(bgr_flow)

        # Запам'ятовуємо кадр із найвиразнішим рухом для звіту
        mag_mean = float(np.mean(mag))
        if mag_mean > best_mag_mean:
            best_mag_mean = mag_mean
            best_sample_orig = frame.copy()
            best_sample_flow = bgr_flow.copy()
            best_frame_idx = frame_idx

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()
    out.release()

    avg_time_ms = np.mean(frame_times)
    print("--- Частина C. Результати щільного потоку (Farneback) ---")
    print(f"Оброблено кадрів: {len(frame_times)}")
    print(f"Середній час обчислення на кадр: {avg_time_ms:.2f} мс")
    print(f"Відео щільного потоку збережено у: {output_video_path}")
    print(f"Найбільш активний кадр для візуалізації: #{best_frame_idx}")

    # Збереження ілюстрації для звіту (оригінал + HSV-візуалізація потоку)
    if best_sample_orig is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(best_sample_orig, cv2.COLOR_BGR2RGB))
        plt.title(f"Оригінальний кадр #{best_frame_idx}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(best_sample_flow, cv2.COLOR_BGR2RGB))
        plt.title("Щільний потік Farneback (HSV -> BGR)")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(sample_img_path, dpi=150)
        plt.show()


if __name__ == "__main__":
    run_dense_optical_flow(
        video_path="lab06/data/video_a.mp4",
        output_video_path="lab06/output/video_a_dense_flow.mp4",
        sample_img_path="lab06/output/part_c_dense_sample.png"
    )
