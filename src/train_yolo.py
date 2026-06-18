import yaml
import time
import mlflow
from ultralytics import YOLO
from ultralytics import settings as yolo_settings

yolo_settings.update({"mlflow": False})

if __name__ == '__main__':
    mlflow.set_tracking_uri("file:///D:/MLPrac/outputs/mlruns")
    mlflow.set_experiment("fire_detection_comparison")

    RUNS = [
        ("yolov8m.pt",    r"D:\MLPrac\configs\run_yolov8m.yaml",    "YOLOv8m"),
        ("yolo11n.pt",    r"D:\MLPrac\configs\run_yolov11n.yaml",   "YOLOv11n"),
        ("yolov10s.pt",   r"D:\MLPrac\configs\run_yolov10s.yaml",   "YOLOv10s"),
        ("rtdetr-l.pt", r"D:\MLPrac\configs\run_rtdetr_l.yaml", "RT-DETR-L"),
    ]

    for model_weights, cfg_path, run_name in RUNS:
        print(f"  Запуск: {run_name}")

        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        mlflow.end_run()

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(cfg)

            model = YOLO(model_weights)
            start = time.time()
            results = model.train(**{k: v for k, v in cfg.items() if k != "model"})
            elapsed = time.time() - start

            mlflow.end_run()

            metrics = results.results_dict
            with mlflow.start_run(run_name=run_name, nested=False):
                mlflow.log_metrics({
                    "mAP50":          metrics.get("metrics/mAP50(B)",    0),
                    "mAP50_95":       metrics.get("metrics/mAP50-95(B)", 0),
                    "precision":      metrics.get("metrics/precision(B)", 0),
                    "recall":         metrics.get("metrics/recall(B)",    0),
                    "train_time_min": round(elapsed / 60, 2),
                })
                mlflow.log_artifact(cfg_path)

            print(f"{run_name} завершён за {elapsed/60:.1f} мин.")
            print(f"mAP@50 = {metrics.get('metrics/mAP50(B)', 0):.4f}")