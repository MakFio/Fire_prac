import shutil
import json
from pathlib import Path
from datetime import datetime

# модель с наибольшим mAP@0.5
BEST_WEIGHTS = r"D:\MLPrac\outputs\runs\yolov8m_dfire\weights\best.pt"
BEST_NAME    = "YOLOv8m"

DEST = Path(r"D:\MLPrac\models\best_model")
DEST.mkdir(parents=True, exist_ok=True)

shutil.copy(BEST_WEIGHTS, DEST / "weights.pt")

metadata = {
    "model_name":      f"{BEST_NAME}_DFire",
    "architecture":    BEST_NAME,
    "dataset":         "D-Fire",
    "classes":         ["fire", "smoke"],
    "input_size":      640,
    "conf_threshold":  0.25,
    "training_date":   datetime.now().isoformat(),
    "weights_file":    "weights.pt",
    "train_images":    14638,
    "val_images":      2583,
    "test_images":     4306,
}
(DEST / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
)
(DEST / "config.yaml").write_text(
    "model: /app/models/best_model/weights.pt\n"
    "nc: 2\nnames:\n  0: fire\n  1: smoke\n"
    "conf: 0.25\niou: 0.45\nimgsz: 640\n"
)

print(f"Модель сохранена: {DEST}")
for f in DEST.iterdir():
    print(f"  {f.name}")