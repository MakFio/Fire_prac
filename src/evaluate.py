import time
import pandas as pd
from pathlib import Path
from ultralytics import YOLO, RTDETR

if __name__ == '__main__':

    TEST_DATA   = "D:/MLPrac/configs/dfire.yaml"
    RESULTS_DIR = Path(r"D:\MLPrac\outputs\eval")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    MODELS = [
        ("YOLOv8m",   r"D:\MLPrac\outputs\runs\yolov8m_dfire\weights\best.pt",  "yolo"),
        ("YOLOv11n",  r"D:\MLPrac\outputs\runs\yolov11n_dfire\weights\best.pt",  "yolo"),
        ("YOLOv10s",  r"D:\MLPrac\outputs\runs\yolov10s_dfire\weights\best.pt", "yolo"),
        ("RT-DETR-L", r"D:\MLPrac\outputs\runs\rtdetr_l_dfire\weights\best.pt", "rtdetr"),
    ]

    rows = []
    for name, weights, mtype in MODELS:
        print(f"Оценка: {name}\n")

        model = RTDETR(weights) if mtype == "rtdetr" else YOLO(weights)

        t0 = time.time()
        val = model.val(
            data=TEST_DATA,
            split="test",
            imgsz=640,
            batch=16,
            device=0,
            workers=4,
            verbose=True,
            save_json=True,
            project=str(RESULTS_DIR),
            name=name,
            exist_ok=True,
        )
        elapsed = time.time() - t0

        m = val.results_dict
        speed = val.speed  # {'preprocess': ms, 'inference': ms, 'postprocess': ms}

        row = {
            "Модель":         name,
            "mAP@0.5":        round(m.get("metrics/mAP50(B)",    0), 4),
            "mAP@0.5:0.95":   round(m.get("metrics/mAP50-95(B)", 0), 4),
            "Precision":      round(m.get("metrics/precision(B)", 0), 4),
            "Recall":         round(m.get("metrics/recall(B)",    0), 4),
            "Latency_ms":     round(speed.get("inference", 0), 2),
            "Eval_time_min":  round(elapsed / 60, 2),
            "Weights":        weights,
        }
        rows.append(row)
        print(f"  mAP@0.5 = {row['mAP@0.5']}  |  Latency = {row['Latency_ms']} ms/img")

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "comparison_table.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ")
    print(df.to_string(index=False))
    print(f"\nТаблица сохранена: {csv_path}")