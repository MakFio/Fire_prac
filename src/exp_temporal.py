import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from tqdm import tqdm

MODEL_PATH   = r"D:\MLPrac\outputs\runs\yolov11n_dfire\weights\best.pt"
VIDEO_DIR    = Path(r"D:\MLPrac\video")
OUT_DIR      = Path(r"D:\MLPrac\outputs\temporal_results_v11n")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESH  = 0.25
WINDOW_SIZES = [1, 3, 5, 7, 10]

model = YOLO(MODEL_PATH)

video_files = (
    list(VIDEO_DIR.glob("*.mp4")) +
    list(VIDEO_DIR.glob("*.avi")) +
    list(VIDEO_DIR.glob("*.mov"))
)
print(f"Найдено видео: {len(video_files)}")

all_stats = []

for W in WINDOW_SIZES:
    total_alerts = 0
    total_frames = 0

    for vf in tqdm(video_files, desc=f"W={W}"):
        cap    = cv2.VideoCapture(str(vf))
        window = deque(maxlen=W)
        prev_alarm = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            results  = model.predict(frame, conf=CONF_THRESH, verbose=False)
            detected = len(results[0].boxes) > 0
            window.append(detected)

            # Тревога - все W кадров содержат детекцию
            alarm_now = len(window) == W and all(window)
            if alarm_now and not prev_alarm:
                total_alerts += 1
            prev_alarm = alarm_now

        cap.release()

    stat = {
        "window_W":         W,
        "total_alerts":     total_alerts,
        "total_frames":     total_frames,
        "alerts_per_video": round(total_alerts / max(len(video_files), 1), 2),
    }
    all_stats.append(stat)
    print(f"W={W}: тревог={total_alerts}, кадров={total_frames}")

with open(OUT_DIR / "temporal_stats.json", "w", encoding="utf-8") as f:
    json.dump(all_stats, f, indent=2, ensure_ascii=False)

ws     = [s["window_W"]     for s in all_stats]
alerts = [s["total_alerts"] for s in all_stats]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ws, alerts, marker="o", color="#e74c3c", linewidth=2)
ax.set_xlabel("Размер окна W (кадров)")
ax.set_ylabel("Число тревог")
ax.set_title("Temporal Smoothing: тревоги / размер окна")
ax.grid(True, alpha=0.3)
for w, a in zip(ws, alerts):
    ax.annotate(str(a), (w, a), textcoords="offset points", xytext=(4, 6), fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "temporal_plot.png", dpi=150)
plt.close()

print(f"\nРезультаты: {OUT_DIR}")
print(f"График: {OUT_DIR / 'temporal_plot.png'}")