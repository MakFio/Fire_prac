import json
import pandas as pd
from pathlib import Path

EVAL_DIR   = Path(r"D:\MLPrac\outputs\eval")
FRCNN_JSON = EVAL_DIR / "FasterRCNN" / "frcnn_metrics.json"
YOLO_CSV   = EVAL_DIR / "comparison_table.csv"
OUT_CSV    = EVAL_DIR / "comparison_table_5models.csv"

df_yolo = pd.read_csv(YOLO_CSV)
df_yolo = df_yolo.rename(columns={
    "mAP@0.5":      "mAP@0.5",
    "mAP@0.5:0.95": "mAP@0.5:0.95",
})

keep = ["Модель", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "Latency_ms"]
for col in keep:
    if col not in df_yolo.columns:
        df_yolo[col] = None
df_yolo = df_yolo[keep]

with open(FRCNN_JSON, encoding="utf-8") as f:
    frcnn = json.load(f)

df_frcnn = pd.DataFrame([{
    "Модель":       frcnn["Модель"],
    "mAP@0.5":      frcnn["mAP@0.5"],
    "mAP@0.5:0.95": frcnn.get("mAP@0.5:0.95"),
    "Precision":    frcnn["Precision"],
    "Recall":       frcnn["Recall"],
    "Latency_ms":   frcnn["Latency_ms"],
}])

df_all = pd.concat([df_yolo, df_frcnn], ignore_index=True)
df_all.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(df_all.to_string(index=False))
print(f"\nСохранена: {OUT_CSV}")