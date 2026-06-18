import os
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw

MODEL_PATH  = os.getenv("MODEL_PATH", r"D:\MLPrac\models\best_model\weights.pt")
CLASS_NAMES = ["fire", "smoke"]
COLORS      = {"fire": (220, 50, 50), "smoke": (160, 160, 160)}

class FireDetector:
    def __init__(self):
        self.model      = YOLO(MODEL_PATH)
        self.model_name = os.path.basename(MODEL_PATH)
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf       = float(os.getenv("CONF_THRESHOLD", "0.25"))

    def predict(self, img: Image.Image):
        results    = self.model.predict(img, conf=self.conf, verbose=False)
        boxes      = results[0].boxes
        detections = []
        annotated  = img.copy()
        draw       = ImageDraw.Draw(annotated)

        for box in boxes:
            cls_id   = int(box.cls[0])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
            conf_val = float(box.conf[0])
            xyxy     = box.xyxy[0].cpu().numpy().tolist()
            detections.append({
                "class":      cls_name,
                "confidence": round(conf_val, 3),
                "bbox":       [round(v, 1) for v in xyxy],
            })
            color = COLORS.get(cls_name, (255, 200, 0))
            draw.rectangle(xyxy[:4], outline=color, width=3)
            draw.text((xyxy[0], max(xyxy[1] - 14, 0)),
                      f"{cls_name} {conf_val:.2f}", fill=color)

        return detections, annotated