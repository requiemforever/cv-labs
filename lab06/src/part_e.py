import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

os.makedirs("lab06/output", exist_ok=True)


def load_gt(csv_path):
    """Завантаження ручної розмітки (Крок 18)."""
    gt = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[int(row["frame"])] = (float(row["x"]), float(row["y"]))
    return gt


def create_kalman_filter(x, y):
    """Крок 17. Створення фільтра Калмана зі станом (x, y, vx, vy)."""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], np.float32)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kf


def run_tracker(video_path, gt, use_kalman=False, max_missed=5,
                output_video_path=None, min_area=2000):
    """
    Кроки 16–17, 19. Реалізація трекера (простого або з фільтром Калмана)
    та обчислення метрик надійності за ручною розміткою.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Помилка: не вдалося відкрити {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    tracks = {}  # tid -> dict
    next_id = 1
    max_match_dist = 80.0

    # Збереження результатів на розмічених кадрах для обчислення метрик
    gt_matches = []  # список кортежів (frame_idx, matched_tid або None, dist, (tx, ty))
    bbox_sizes = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Детекція об'єктів через MOG2 + морфологія
        fg_raw = bg_sub.apply(frame)
        _, fg_mask = cv2.threshold(fg_raw, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        detections = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cx, cy = centroids[i]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                detections.append(((cx, cy), (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], bw, bh)))
                bbox_sizes.append(max(bw, bh))

        # 1. Прогноз положення для кожного наявного треку
        for tid, tr in tracks.items():
            if use_kalman:
                pred = tr["kf"].predict()
                tr["pred"] = (float(pred[0, 0]), float(pred[1, 0]))
            else:
                px, py = tr["pos"]
                vx, vy = tr["vel"]
                tr["pred"] = (px + vx, py + vy)

        # 2. Зіставлення треків із детекціями за найближчим центроїдом
        used_dets = set()
        for tid in list(tracks.keys()):
            pred_x, pred_y = tracks[tid]["pred"]
            best_idx = -1
            best_dist = max_match_dist

            for idx, (det_c, _) in enumerate(detections):
                if idx in used_dets:
                    continue
                dist = np.hypot(pred_x - det_c[0], pred_y - det_c[1])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx != -1:
                used_dets.add(best_idx)
                nx, ny = detections[best_idx][0]
                if use_kalman:
                    meas = np.array([[np.float32(nx)], [np.float32(ny)]])
                    tracks[tid]["kf"].correct(meas)
                    st = tracks[tid]["kf"].statePost
                    tracks[tid]["pos"] = (float(st[0, 0]), float(st[1, 0]))
                else:
                    ox, oy = tracks[tid]["pos"]
                    tracks[tid]["vel"] = (nx - ox, ny - oy)
                    tracks[tid]["pos"] = (nx, ny)
                tracks[tid]["missed"] = 0
                tracks[tid]["history"].append(tracks[tid]["pos"])
                if len(tracks[tid]["history"]) > 40:
                    tracks[tid]["history"].pop(0)
            else:
                tracks[tid]["missed"] += 1
                tracks[tid]["pos"] = tracks[tid]["pred"]
                if tracks[tid]["missed"] > max_missed:
                    del tracks[tid]

        # 3. Створення нових треків для незіставлених детекцій
        for idx, (det_c, _) in enumerate(detections):
            if idx not in used_dets:
                tr_data = {
                    "pos": det_c,
                    "pred": det_c,
                    "vel": (0.0, 0.0),
                    "missed": 0,
                    "history": [det_c]
                }
                if use_kalman:
                    tr_data["kf"] = create_kalman_filter(det_c[0], det_c[1])
                tracks[next_id] = tr_data
                next_id += 1

        # 4. Перевірка на розмічених кадрах (Крок 19)
        if frame_idx in gt:
            gx, gy = gt[frame_idx]
            best_tid = None
            best_dist = float("inf")
            best_pos = None
            for tid, tr in tracks.items():
                tx, ty = tr["pos"]
                d = np.hypot(tx - gx, ty - gy)
                if d < best_dist:
                    best_dist = d
                    best_tid = tid
                    best_pos = (tx, ty)
            gt_matches.append((frame_idx, best_tid, best_dist, best_pos))

        # 5. Візуалізація відео
        if out is not None:
            vis = frame.copy()
            for tid, tr in tracks.items():
                color = ((tid * 67) % 255, (tid * 137) % 255, (tid * 211) % 255)
                if len(tr["history"]) > 1:
                    pts = np.array(tr["history"], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [pts], False, color, 2)
                tx, ty = int(tr["pos"][0]), int(tr["pos"][1])
                cv2.circle(vis, (tx, ty), 6, color, -1)
                cv2.putText(vis, f"ID {tid}", (tx + 8, ty - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if frame_idx in gt:
                gx, gy = int(gt[frame_idx][0]), int(gt[frame_idx][1])
                cv2.drawMarker(vis, (gx, gy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)

            out.write(vis)

        frame_idx += 1

    cap.release()
    if out is not None:
        out.release()

    # Обчислення метрик (Крок 19)
    # Допуск: половина характерного розміру об'єкта (або мінімум 60 px)
    tol = max(60.0, float(np.median(bbox_sizes)) * 0.5) if bbox_sizes else 60.0

    held_count = 0
    id_switches = 0
    last_valid_id = None
    tracked_frames = []
    tracked_x = []

    for f_idx, tid, dist, pos in gt_matches:
        if tid is not None and dist <= tol:
            held_count += 1
            tracked_frames.append(f_idx)
            tracked_x.append(pos[0])
            if last_valid_id is not None and tid != last_valid_id:
                id_switches += 1
            last_valid_id = tid

    total_gt = len(gt_matches)
    hold_ratio = (held_count / total_gt * 100.0) if total_gt > 0 else 0.0

    return {
        "held_count": held_count,
        "total_gt": total_gt,
        "hold_ratio": hold_ratio,
        "id_switches": id_switches,
        "tolerance_px": tol,
        "tracked_frames": tracked_frames,
        "tracked_x": tracked_x
    }


if __name__ == "__main__":
    gt_data = load_gt("lab06/data/video_b_gt.csv")

    # 1. Базовий трекер (Крок 16: видалення через 5 кадрів без збігу)
    res_basic = run_tracker(
        video_path="lab06/data/video_b.mp4",
        gt=gt_data,
        use_kalman=False,
        max_missed=5,
        output_video_path="lab06/output/video_b_tracker_basic.mp4"
    )

    # 2. Трекер із фільтром Калмана (Крок 17: переживає перекриття)
    res_kalman = run_tracker(
        video_path="lab06/data/video_b.mp4",
        gt=gt_data,
        use_kalman=True,
        max_missed=30,
        output_video_path="lab06/output/video_b_tracker_kalman.mp4"
    )

    print("--- Частина E (Крок 19). Метрики трекера ---")
    print(f"Поріг утримання (розмір об'єкта): {res_basic['tolerance_px']:.1f} px\n")

    print("[1] Базовий трекер (Крок 16, max_missed=5):")
    print(f"  Утримано кадрів: {res_basic['held_count']}/{res_basic['total_gt']} ({res_basic['hold_ratio']:.1f}%)")
    print(f"  Кількість підмін ID (ID switches): {res_basic['id_switches']}\n")

    print("[2] Трекер із фільтром Калмана (Крок 17, max_missed=30):")
    print(f"  Утримано кадрів: {res_kalman['held_count']}/{res_kalman['total_gt']} ({res_kalman['hold_ratio']:.1f}%)")
    print(f"  Кількість підмін ID (ID switches): {res_kalman['id_switches']}\n")

    # Графік порівняння координати X: ручна розмітка vs трекери
    gt_frames = sorted(gt_data.keys())
    gt_x = [gt_data[f][0] for f in gt_frames]

    plt.figure(figsize=(10, 5))
    plt.plot(gt_frames, gt_x, 'ko-', label='Ручна розмітка (GT)', markersize=5)
    plt.plot(res_basic["tracked_frames"], res_basic["tracked_x"], 'b^--', label='Базовий трекер', alpha=0.8)
    plt.plot(res_kalman["tracked_frames"], res_kalman["tracked_x"], 'rs:', label='Трекер із фільтром Калмана', alpha=0.8)
    plt.axvspan(365, 385, color='gray', alpha=0.2, label='Епізод перекриття')
    plt.title('Крок 19. Порівняння траєкторії цілі (координата X)')
    plt.xlabel('Номер кадру')
    plt.ylabel('Координата X (пікселі)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("lab06/output/part_e_tracking.png", dpi=150)
    plt.show()
