import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

os.makedirs("lab06/output", exist_ok=True)


def check_exposure_stability(video_path, num_samples=5):
    """Крок 2. Перевірка стабільності середньої яскравості кадрів."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_samples, dtype=int)

    brightness_vals = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_vals.append(np.mean(gray))
    cap.release()

    std_val = np.std(brightness_vals)
    print(f"--- Крок 2. Перевірка яскравості ({video_path}) ---")
    print("Середня яскравість на зразках:", [round(v, 2) for v in brightness_vals])
    print(f"Стандартне відхилення: {std_val:.2f}\n")


def run_sparse_optical_flow(video_path, output_video_path, plot_path):
    """Частина B (Кроки 3–7). Розріджений потік Лукаса-Канаде."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Помилка: не вдалося відкрити {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    ret, first_frame = cap.read()
    if not ret:
        return
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Крок 3. Вибір початкових точок
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=10
    )
    initial_count = len(p0)

    # Параметри пірамідального Лукаса-Канаде (Крок 4)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Окремі набори точок: без фільтрації та зі зворотною перевіркою
    pts_raw = p0.copy()
    pts_fb = p0.copy()

    # Історія останніх 20 позицій для кожної точки (Крок 7)
    tracks = [[tuple(pt[0])] for pt in pts_fb]

    alive_raw_history = [initial_count]
    alive_fb_history = [initial_count]
    frame_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t_start = time.perf_counter()

        # 1. Відстеження без зворотної фільтрації
        if len(pts_raw) > 0:
            p1_raw, st_raw, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_raw, None, **lk_params)
            good_raw = st_raw.ravel() == 1
            pts_raw = p1_raw[good_raw].reshape(-1, 1, 2)

        # 2. Відстеження зі зворотною перевіркою (Крок 5)
        if len(pts_fb) > 0:
            p1_fb, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_fb, None, **lk_params)
            p0_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1_fb, None, **lk_params)

            # Відстань між початковою точкою та повернутою
            fb_error = np.linalg.norm(pts_fb - p0_back, axis=2).ravel()
            good_fb = (st_fwd.ravel() == 1) & (st_bwd.ravel() == 1) & (fb_error <= 1.0)

            pts_fb = p1_fb[good_fb].reshape(-1, 1, 2)
            # Оновлюємо траєкторії лише для живих точок (зберігаємо останні 20 позицій)
            new_tracks = []
            for track, is_good, new_pt in zip(tracks, good_fb, p1_fb):
                if is_good:
                    track.append(tuple(new_pt[0]))
                    if len(track) > 20:
                        track.pop(0)
                    new_tracks.append(track)
            tracks = new_tracks

        frame_times.append((time.perf_counter() - t_start) * 1000.0)

        alive_raw_history.append(len(pts_raw))
        alive_fb_history.append(len(pts_fb))

        # Крок 7. Малюємо траєкторії поверх відео
        vis = frame.copy()
        for track in tracks:
            if len(track) > 1:
                pts_arr = np.array(track, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts_arr], False, (0, 255, 0), 2)
            x, y = int(track[-1][0]), int(track[-1][1])
            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

        out.write(vis)
        prev_gray = curr_gray

    cap.release()
    out.release()

    # Крок 6. Підсумкова статистика
    surv_raw_pct = (alive_raw_history[-1] / initial_count) * 100
    surv_fb_pct = (alive_fb_history[-1] / initial_count) * 100
    avg_time_ms = np.mean(frame_times)

    print("--- Частина B. Результати розрідженого потоку ---")
    print(f"Початкова кількість точок: {initial_count}")
    print(f"Дожило до кінця (без фільтрації): {alive_raw_history[-1]} ({surv_raw_pct:.1f}%)")
    print(f"Дожило до кінця (зі зворотною перевіркою <= 1 px): {alive_fb_history[-1]} ({surv_fb_pct:.1f}%)")
    print(f"Середній час обчислення на кадр: {avg_time_ms:.2f} мс")
    print(f"Відео з траєкторіями збережено у: {output_video_path}")

    # Побудова графіка (Крок 6)
    plt.figure(figsize=(10, 5))
    plt.plot(alive_raw_history, label='Без фільтрації', linewidth=2)
    plt.plot(alive_fb_history, label='З фільтрацією (зворотна похибка <= 1 px)', linewidth=2)
    plt.title('Крок 6. Кількість живих точок від номера кадру')
    plt.xlabel('Номер кадру')
    plt.ylabel('Кількість живих точок')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    check_exposure_stability("lab06/data/video_a.mp4")
    check_exposure_stability("lab06/data/video_b.mp4")

    run_sparse_optical_flow(
        video_path="lab06/data/video_a.mp4",
        output_video_path="lab06/output/video_a_sparse_flow.mp4",
        plot_path="lab06/output/part_b_survival.png"
    )
