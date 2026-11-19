import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("lab06/output", exist_ok=True)


def run_background_subtraction(video_path, output_video_path, sample_img_path,
                               min_area=2000, orientation="horizontal"):
    """Частина D (Кроки 11–14). Віднімання фону MOG2, морфологія та підрахунок через лінію."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Помилка: не вдалося відкрити {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Координата лінії підрахунку по центру кадру (горизонтальна або вертикальна)
    line_pos = h // 2 if orientation == "horizontal" else w // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Крок 11. Ініціалізація MOG2
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True
    )

    # Ядра для морфологічної постобробки (Крок 13)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Простий трекер центроїдів для запобігання подвійному підрахунку (Крок 14)
    tracks = {}       # track_id -> {'centroid': (x, y), 'counted': bool, 'missed': int}
    next_id = 0
    total_count = 0
    max_dist = 80.0

    saved_sample = False
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Крок 12. Отримання маски та видалення тіней (значення 127)
        fg_mask_raw = bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask_raw, 200, 255, cv2.THRESH_BINARY)

        # Крок 13. Морфологічне відкриття (3x3) та закриття (5x5)
        mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)

        # Пошук зв'язних компонент та фільтрація за площею
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask_clean)

        detections = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cx, cy = centroids[i]
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                detections.append(((cx, cy), (x, y, bw, bh)))

        # Зіставлення детекцій із наявними треками за найближчим центроїдом
        used_detections = set()
        for tid in list(tracks.keys()):
            best_idx = -1
            best_dist = max_dist
            tx, ty = tracks[tid]['centroid']

            for idx, (det_c, _) in enumerate(detections):
                if idx in used_detections:
                    continue
                dist = np.hypot(tx - det_c[0], ty - det_c[1])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx != -1:
                used_detections.add(best_idx)
                new_c = detections[best_idx][0]

                # Обираємо координату залежно від орієнтації лінії (Y для горизонтальної, X для вертикальної)
                prev_coord = tracks[tid]['centroid'][1] if orientation == "horizontal" else tracks[tid]['centroid'][0]
                curr_coord = new_c[1] if orientation == "horizontal" else new_c[0]

                # Перевірка перетину лінії (один раз на трек)
                if not tracks[tid]['counted']:
                    crossed_fwd = (prev_coord <= line_pos and curr_coord > line_pos)
                    crossed_bwd = (prev_coord >= line_pos and curr_coord < line_pos)
                    if crossed_fwd or crossed_bwd:
                        total_count += 1
                        tracks[tid]['counted'] = True

                tracks[tid]['centroid'] = new_c
                tracks[tid]['missed'] = 0
            else:
                tracks[tid]['missed'] += 1
                if tracks[tid]['missed'] > 10:
                    del tracks[tid]

        # Створення нових треків для незіставлених детекцій
        for idx, (det_c, _) in enumerate(detections):
            if idx not in used_detections:
                tracks[next_id] = {
                    'centroid': det_c,
                    'counted': False,
                    'missed': 0
                }
                next_id += 1

        # Візуалізація результату
        vis = frame.copy()
        if orientation == "horizontal":
            cv2.line(vis, (0, line_pos), (w, line_pos), (0, 255, 255), 2)
        else:
            cv2.line(vis, (line_pos, 0), (line_pos, h), (0, 255, 255), 2)

        for (cx, cy), (x, y, bw, bh) in detections:
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        cv2.putText(
            vis, f"Count: {total_count}", (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3
        )
        out.write(vis)

        # Збереження прикладу маски для звіту, коли об'єкт поблизу лінії
        if not saved_sample and len(detections) > 0 and frame_idx > 30:
            near_line = any(
                abs((c[1] if orientation == "horizontal" else c[0]) - line_pos) < 120
                for c, _ in detections
            )
            if near_line:
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                plt.title(f"Детекція та лінія (кадр #{frame_idx})")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(fg_mask_raw, cmap="gray")
                plt.title("Сира маска MOG2 (з тінями)")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(mask_clean, cmap="gray")
                plt.title("Очищена маска (морфологія)")
                plt.axis("off")

                plt.tight_layout()
                plt.savefig(sample_img_path, dpi=150)
                plt.close()
                saved_sample = True

        frame_idx += 1

    cap.release()
    out.release()

    print(f"--- Частина D. Віднімання фону ({video_path}, лінія: {orientation}) ---")
    print(f"Автоматичний підрахунок перетинів лінії: {total_count}")
    print(f"Відео збережено у: {output_video_path}")
    if saved_sample:
        print(f"Ілюстрацію масок збережено у: {sample_img_path}\n")


if __name__ == "__main__":
    # Відео A: рух вертикальний -> горизонтальна лінія
    run_background_subtraction(
        video_path="lab06/data/video_a.mp4",
        output_video_path="lab06/output/video_a_mog2_count.mp4",
        sample_img_path="lab06/output/part_d_video_a_masks.png",
        min_area=2000,
        orientation="horizontal"
    )

    # Відео B: рух горизонтальний -> вертикальна лінія
    run_background_subtraction(
        video_path="lab06/data/video_b.mp4",
        output_video_path="lab06/output/video_b_mog2_count.mp4",
        sample_img_path="lab06/output/part_d_video_b_masks.png",
        min_area=2000,
        orientation="vertical"
    )
