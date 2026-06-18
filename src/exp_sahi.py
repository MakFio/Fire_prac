import glob
import time
import json
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction

MODEL_PATH  = r"D:\MLPrac\outputs\runs\yolov11n_dfire\weights\best.pt"
TEST_IMAGES = glob.glob(r"D:\MLPrac\D-Fire\test\images\*.jpg")[:200]
OUT_DIR     = Path(r"D:\MLPrac\outputs\sahi_results_rtdetr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Тест SAHI на {len(TEST_IMAGES)} изображениях")

det_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=0.25,
    device="cuda:0",
)

results_base, results_sahi = [], []

for img_path in TEST_IMAGES:
    t0 = time.perf_counter()
    pred_base = get_prediction(img_path, det_model)
    lat_base  = (time.perf_counter() - t0) * 1000

    # SAHI 640×640, перекрытие 20%
    t0 = time.perf_counter()
    pred_sahi = get_sliced_prediction(
        img_path, det_model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
    )
    lat_sahi = (time.perf_counter() - t0) * 1000

    results_base.append({"n": len(pred_base.object_prediction_list), "ms": lat_base})
    results_sahi.append({"n": len(pred_sahi.object_prediction_list), "ms": lat_sahi})

n = len(TEST_IMAGES)
summary = {
    "model":         MODEL_PATH,
    "images_tested": n,
    "base": {
        "avg_detections": round(sum(r["n"]  for r in results_base) / n, 2),
        "avg_latency_ms": round(sum(r["ms"] for r in results_base) / n, 2),
    },
    "sahi": {
        "avg_detections": round(sum(r["n"]  for r in results_sahi) / n, 2),
        "avg_latency_ms": round(sum(r["ms"] for r in results_sahi) / n, 2),
    },
}

with open(OUT_DIR / "sahi_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nРезультаты: {OUT_DIR / 'sahi_summary.json'}")