import glob
import random
import cv2
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

WEIGHTS_PATH   = Path(r"D:\MLPrac\outputs\runs\frcnn_dfire\best.pt")
TEST_IMG_DIR   = r"D:\MLPrac\D-Fire\test\images"
TEST_LABEL_DIR = r"D:\MLPrac\D-Fire\test\labels"
OUT_DIR        = Path(r"D:\MLPrac\outputs\examples\FasterRCNN")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3
IMGSZ       = 640
CONF_THRESH = 0.25
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {1: "fire", 2: "smoke"}
PRED_COLORS = {1: (0, 0, 220), 2: (130, 130, 130)}
GT_COLOR    = (0, 200, 0)

model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE); model.eval()

test_imgs = glob.glob(f"{TEST_IMG_DIR}/*.jpg")
random.seed(42); random.shuffle(test_imgs)

good_cases, bad_cases = [], []

for img_path in test_imgs:
    if len(good_cases) >= 5 and len(bad_cases) >= 5:
        break

    label_path = Path(TEST_LABEL_DIR) / (Path(img_path).stem + ".txt")
    if not label_path.exists():
        continue

    img_pil = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img_pil.size
    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]

    # Ground truth
    gt_boxes = []
    content = label_path.read_text(encoding="utf-8").strip()
    if content:
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0]) + 1  # YOLO 0/1 → COCO 1/2
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            x1 = int((xc - bw / 2) * w); y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w); y2 = int((yc + bh / 2) * h)
            gt_boxes.append((cls, x1, y1, x2, y2))

    if not gt_boxes:
        continue

    # Предсказания через torchvision Faster R-CNN
    img_resized = img_pil.resize((IMGSZ, IMGSZ))
    img_tensor  = TF.to_tensor(img_resized).unsqueeze(0).to(DEVICE)
    scale_x = w / IMGSZ; scale_y = h / IMGSZ

    with torch.no_grad():
        preds = model(img_tensor)[0]

    pred_classes = []
    pred_boxes   = []
    mask = preds["scores"] >= CONF_THRESH
    for cls_id, box, score in zip(
        preds["labels"][mask].tolist(),
        preds["boxes"][mask].tolist(),
        preds["scores"][mask].tolist(),
    ):
        bx1 = int(box[0] * scale_x); by1 = int(box[1] * scale_y)
        bx2 = int(box[2] * scale_x); by2 = int(box[3] * scale_y)
        pred_classes.append(cls_id)
        pred_boxes.append((cls_id, score, bx1, by1, bx2, by2))

    gt_cls_set   = {cls for cls, *_ in gt_boxes}
    correct = any(c in pred_classes for c in gt_cls_set)

    canvas = img_bgr.copy()
    for cls, x1, y1, x2, y2 in gt_boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GT_COLOR, 2)
        cv2.putText(canvas, f"GT:{CLASS_NAMES.get(cls, '?')}",
                    (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1)
    for cls, score, bx1, by1, bx2, by2 in pred_boxes:
        color = PRED_COLORS.get(cls, (255, 200, 0))
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(canvas, f"PR:{CLASS_NAMES.get(cls,'?')} {score:.2f}",
                    (bx1, min(by2 + 15, h - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    stem = Path(img_path).stem
    if correct and len(good_cases) < 5:
        cv2.imwrite(str(OUT_DIR / f"good_{len(good_cases)+1}_{stem}.jpg"), canvas)
        good_cases.append(img_path)
        print(f"[GOOD {len(good_cases)}] {stem}")
    elif not correct and len(bad_cases) < 5:
        cv2.imwrite(str(OUT_DIR / f"bad_{len(bad_cases)+1}_{stem}.jpg"), canvas)
        bad_cases.append(img_path)
        print(f"[BAD  {len(bad_cases)}] {stem}")

print(f"\nУдачных: {len(good_cases)}, ошибочных: {len(bad_cases)}")
print(f"Сохранены в: {OUT_DIR}")