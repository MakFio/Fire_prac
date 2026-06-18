import time
import json
import csv
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT       = Path(__file__).resolve().parent.parent
WEIGHTS_PT = ROOT / "models" / "best_model" / "weights.pt"
DATA_YAML  = ROOT / "configs" / "dfire.yaml"
TEST_IMGS  = sorted((ROOT / "D-Fire" / "test" / "images").glob("*.jpg"))[:100]
OUT_DIR    = ROOT / "models" / "exported"
TABLE_DIR  = ROOT / "outputs" / "tables"


def measure_latency_gpu(yolo_model, images, device, n_warmup=50, n_measure=100):
    imgs = [str(p) for p in images]
    print(f"    {n_warmup} итераций", end=" ", flush=True)
    warmup_list = (imgs * (n_warmup // len(imgs) + 1))[:n_warmup]
    for img in warmup_list:
        yolo_model.predict(img, device=device, verbose=False)
    print("готово")

    measure_list = (imgs * (n_measure // len(imgs) + 1))[:n_measure]
    lats = []

    if device == "cuda":
        for img in measure_list:
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            yolo_model.predict(img, device=device, verbose=False)
            e.record()
            torch.cuda.synchronize()
            lats.append(s.elapsed_time(e))      # мс
    else:
        for img in measure_list:
            t0 = time.perf_counter()
            yolo_model.predict(img, device=device, verbose=False)
            lats.append((time.perf_counter() - t0) * 1000)

    return round(float(np.mean(lats)), 2), round(float(np.std(lats)), 2)


def run_val(weights, label, data_yaml, device):
    print(f"  val [{label}]...", flush=True)
    m = YOLO(weights)
    r = m.val(
        data=str(data_yaml),
        imgsz=640,
        device=device,
        workers=0,
        verbose=False,
    )
    return {
        "mAP50":     round(float(r.box.map50), 4),
        "mAP50_95":  round(float(r.box.map),   4),
        "precision": round(float(r.box.mp),    4),
        "recall":    round(float(r.box.mr),    4),
    }


if __name__ == "__main__":

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Устройство: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(WEIGHTS_PT)

    print("Экспорт в ONNX")
    t0 = time.time()
    onnx_path = Path(model.export(
        format="onnx", imgsz=640, dynamic=False,
        simplify=True, device=DEVICE,
    ))
    onnx_export_t = round(time.time() - t0, 1)
    print(f"  > {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} МБ)  [{onnx_export_t}s]\n")

    print("Экспорт в TorchScript")
    t0 = time.time()
    ts_path = Path(model.export(
        format="torchscript", imgsz=640, device=DEVICE,
    ))
    ts_export_t = round(time.time() - t0, 1)
    print(f"  → {ts_path}  ({ts_path.stat().st_size/1e6:.1f} МБ)  [{ts_export_t}s]\n")

    print("Замер латентности")

    print("  [PyTorch .pt]")
    pt_model  = YOLO(WEIGHTS_PT)
    pt_mean,   pt_std   = measure_latency_gpu(pt_model,  TEST_IMGS, DEVICE)

    print("  [ONNX Runtime]")
    onnx_model = YOLO(onnx_path)
    onnx_mean, onnx_std = measure_latency_gpu(onnx_model, TEST_IMGS, DEVICE)

    print("  [TorchScript]")
    ts_model   = YOLO(ts_path)
    ts_mean,   ts_std   = measure_latency_gpu(ts_model,  TEST_IMGS, DEVICE)

    print("\nПроверка паритета mAP (val)")
    val_pt   = run_val(WEIGHTS_PT, "PyTorch .pt",  DATA_YAML, DEVICE)
    val_onnx = run_val(onnx_path,  "ONNX",         DATA_YAML, DEVICE)
    val_ts   = run_val(ts_path,    "TorchScript",  DATA_YAML, DEVICE)

    delta_onnx = round(val_onnx["mAP50"] - val_pt["mAP50"], 4)
    delta_ts   = round(val_ts["mAP50"]   - val_pt["mAP50"], 4)
    print(f"\n  ΔmAP50 ONNX vs PT:        {delta_onnx:+.4f}")
    print(f"  ΔmAP50 TorchScript vs PT: {delta_ts:+.4f}")
    if abs(delta_onnx) > 0.005:
        print("Расхождение mAP > 0.5 пп")
    else:
        print("Паритет качества")

    rows = [
        {
            "Формат":         "PyTorch (.pt)",
            "Размер_МБ":      round(WEIGHTS_PT.stat().st_size / 1e6, 1),
            "Latency_мс":     pt_mean,
            "Latency_std_мс": pt_std,
            "mAP@0.5":        val_pt["mAP50"],
            "mAP@0.5:0.95":   val_pt["mAP50_95"],
            "Precision":      val_pt["precision"],
            "Recall":         val_pt["recall"],
            "ΔmAP50_vs_PT":   0.0,
        },
        {
            "Формат":         "ONNX",
            "Размер_МБ":      round(onnx_path.stat().st_size / 1e6, 1),
            "Latency_мс":     onnx_mean,
            "Latency_std_мс": onnx_std,
            "mAP@0.5":        val_onnx["mAP50"],
            "mAP@0.5:0.95":   val_onnx["mAP50_95"],
            "Precision":      val_onnx["precision"],
            "Recall":         val_onnx["recall"],
            "ΔmAP50_vs_PT":   delta_onnx,
        },
        {
            "Формат":         "TorchScript",
            "Размер_МБ":      round(ts_path.stat().st_size / 1e6, 1),
            "Latency_мс":     ts_mean,
            "Latency_std_мс": ts_std,
            "mAP@0.5":        val_ts["mAP50"],
            "mAP@0.5:0.95":   val_ts["mAP50_95"],
            "Precision":      val_ts["precision"],
            "Recall":         val_ts["recall"],
            "ΔmAP50_vs_PT":   delta_ts,
        },
    ]

    print("\nИтоговая таблица")
    header = f"  {'Формат':<20} {'Разм.МБ':>8} {'Lат.мс':>9} {'±std':>7} {'mAP50':>7} {'ΔmAP50':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(
            f"  {r['Формат']:<20} {r['Размер_МБ']:>8.1f} "
            f"{r['Latency_мс']:>9.2f} {r['Latency_std_мс']:>7.2f} "
            f"{r['mAP@0.5']:>7.4f} {r['ΔmAP50_vs_PT']:>+8.4f}"
        )

    json_out = {
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
        "export": {
            "onnx":        {"path": str(onnx_path), "size_mb": rows[1]["Размер_МБ"], "export_time_s": onnx_export_t},
            "torchscript": {"path": str(ts_path),   "size_mb": rows[2]["Размер_МБ"], "export_time_s": ts_export_t},
        },
        "benchmark": rows,
    }
    json_path = OUT_DIR / "export_comparison.json"
    json_path.write_text(json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  JSON : {json_path}")

    csv_path = TABLE_DIR / "onnx_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  → {csv_path}")