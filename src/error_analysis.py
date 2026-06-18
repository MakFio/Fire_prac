import glob
import random
import cv2
from pathlib import Path
from ultralytics import YOLO

# Лучшая модель по mAP@0.5
MODEL_PATH     = r"D:\MLPrac\outputs\runs\rtdetr_l_dfire\weights\best.pt"
TEST_IMG_DIR   = r"D:\MLPrac\D-Fire\test\images"
TEST_LABEL_DIR = r"D:\MLPrac\D-Fire\test\labels"
OUT_DIR        = Path(r"D:\MLPrac\outputs\examples\RtdetrL")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "fire", 1: "smoke"}
# BGR цвета
PRED_COLORS = {0: (0, 0, 220), 1: (130, 130, 130)}
GT_COLOR    = (0, 200, 0)  # зелёный для Ground Truth

model     = YOLO(MODEL_PATH)
test_imgs = glob.glob(f"{TEST_IMG_DIR}/*.jpg")
random.seed(42)
random.shuffle(test_imgs)

good_cases, bad_cases = [], []

for img_path in test_imgs:
    if len(good_cases) >= 5 and len(bad_cases) >= 5:
        break

    label_path = Path(TEST_LABEL_DIR) / (Path(img_path).stem + ".txt")
    if not label_path.exists():
        continue

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
    h, w = img_bgr.shape[:2]

    # Ground truth boxes
    gt_boxes = []
    content = label_path.read_text(encoding="utf-8").strip()
    if content:
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            x1 = int((xc - bw / 2) * w); y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w); y2 = int((yc + bh / 2) * h)
            gt_boxes.append((cls, x1, y1, x2, y2))

    if not gt_boxes:
        continue

    results    = model.predict(img_path, conf=0.25, verbose=False)
    pred_boxes = results[0].boxes

    pred_classes = [int(b.cls[0]) for b in pred_boxes] if pred_boxes else []
    gt_classes   = [cls for cls, *_ in gt_boxes]
    correct = any(c in pred_classes for c in gt_classes)

    canvas = img_bgr.copy()

    # GT - зелёные рамки
    for cls, x1, y1, x2, y2 in gt_boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GT_COLOR, 2)
        cv2.putText(canvas, f"GT:{CLASS_NAMES.get(cls, '?')}",
                    (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1)

    # Predictions - цветные рамки
    for box in pred_boxes:
        bxy  = box.xyxy[0].cpu().numpy().astype(int)
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        color = PRED_COLORS.get(cls, (255, 200, 0))
        cv2.rectangle(canvas, (bxy[0], bxy[1]), (bxy[2], bxy[3]), color, 2)
        cv2.putText(canvas, f"PR:{CLASS_NAMES.get(cls, '?')} {conf:.2f}",
                    (bxy[0], min(bxy[3] + 15, h - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    stem = Path(img_path).stem
    if correct and len(good_cases) < 5:
        out_path = OUT_DIR / f"good_{len(good_cases)+1}_{stem}.jpg"
        cv2.imwrite(str(out_path), canvas)
        good_cases.append(img_path)
        print(f"[GOOD {len(good_cases)}] {stem}")
    elif not correct and len(bad_cases) < 5:
        out_path = OUT_DIR / f"bad_{len(bad_cases)+1}_{stem}.jpg"
        cv2.imwrite(str(out_path), canvas)
        bad_cases.append(img_path)
        print(f"[BAD  {len(bad_cases)}] {stem}")

print(f"\nУдачных: {len(good_cases)}, ошибочных: {len(bad_cases)}")
print(f"Сохранены в: {OUT_DIR}")