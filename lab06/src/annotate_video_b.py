import cv2
import csv
import os

VIDEO_PATH = "lab06/data/video_b.mp4"
CSV_PATH = "lab06/data/video_b_gt.csv"
STEP = 10  # Кожен десятий кадр

clicked_point = None


def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def main():
    global clicked_point
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Помилка: не вдалося відкрити {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []

    cv2.namedWindow("Розмітка (Клік = центр цілі | Пробіл = пропустити | Q = вихід)")
    cv2.setMouseCallback("Розмітка (Клік = центр цілі | Пробіл = пропустити | Q = вихід)", mouse_callback)

    print("Інструкція:")
    print("  - ЛКМ: клікнути в центр цільової банки (обирай одну й ту саму банку на все відео).")
    print("  - Пробіл (Space): пропустити кадр, якщо цільової банки ще немає в кадрі або вона вже вийшла.")
    print("  - Q або Esc: завершити розмітку достроково.\n")

    for frame_idx in range(0, total_frames, STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        clicked_point = None
        display = frame.copy()
        cv2.putText(
            display,
            f"Frame {frame_idx}/{total_frames - 1} | Click target / Space=skip",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2
        )
        cv2.imshow("Розмітка (Клік = центр цілі | Пробіл = пропустити | Q = вихід)", display)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if clicked_point is not None:
                cx, cy = clicked_point
                rows.append((frame_idx, cx, cy))
                print(f"Кадр #{frame_idx}: ({cx}, {cy})")
                break
            elif key == ord(' '):  # Пропуск кадру
                print(f"Кадр #{frame_idx}: пропущено")
                break
            elif key in (ord('q'), 27):  # Вихід
                cap.release()
                cv2.destroyAllWindows()
                save_csv(rows)
                return

    cap.release()
    cv2.destroyAllWindows()
    save_csv(rows)


def save_csv(rows):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y"])
        writer.writerows(rows)
    print(f"\nГотово! Збережено {len(rows)} розмічених кадрів у {CSV_PATH}")


if __name__ == "__main__":
    main()
