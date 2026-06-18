import json
import time
from pathlib import Path
from PIL import Image

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

WEIGHTS_PATH = Path(r"D:\MLPrac\outputs\runs\frcnn_dfire\best.pt")
COCO_DIR     = Path(r"D:\MLPrac\data\coco")
EVAL_DIR     = Path(r"D:\MLPrac\outputs\eval\FasterRCNN")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES  = 3
IMGSZ        = 640
CONF_THRESH  = 0.25
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DFireCOCODataset(Dataset):
    def __init__(self, ann_file: Path, imgsz: int = 640):
        with open(ann_file, encoding="utf-8") as f:
            coco = json.load(f)
        self.imgsz = imgsz
        self.images = {img["id"]: img for img in coco["images"]}
        self.ann_by_img = {img_id: [] for img_id in self.images}
        for ann in coco["annotations"]:
            if ann["image_id"] in self.ann_by_img:
                self.ann_by_img[ann["image_id"]].append(ann)
        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_info = self.images[img_id]
        img = Image.open(img_info["file_name"]).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.imgsz, self.imgsz))
        scale_x = self.imgsz / orig_w
        scale_y = self.imgsz / orig_h
        img_tensor = TF.to_tensor(img)
        anns = self.ann_by_img[img_id]
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1 = x * scale_x; y1 = y * scale_y
            x2 = (x + w) * scale_x; y2 = (y + h) * scale_y
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            x2 = min(float(self.imgsz), x2); y2 = min(float(self.imgsz), y2)
            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann["category_id"]))
        if boxes:
            boxes_t  = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)
        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([img_id])}
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))


def iou(box_a, box_b):
    xA = max(box_a[0], box_b[0]); yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2]); yB = min(box_a[3], box_b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = (box_a[2]-box_a[0])*(box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0])*(box_b[3]-box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def compute_map50(all_preds, all_targets, num_cls=2, iou_thr=0.5):
    aps = []
    for cls_id in range(1, num_cls + 1):
        tp_list, n_gt = [], 0
        for preds, targets in zip(all_preds, all_targets):
            gt_boxes = targets["boxes"][targets["labels"] == cls_id].tolist()
            n_gt += len(gt_boxes)
            mask = (preds["labels"] == cls_id) & (preds["scores"] >= CONF_THRESH)
            pb   = preds["boxes"][mask].tolist()
            sc   = preds["scores"][mask].tolist()
            matched = [False] * len(gt_boxes)
            for score, pred_box in sorted(zip(sc, pb), reverse=True):
                best_iou, best_j = 0.0, -1
                for j, gt_box in enumerate(gt_boxes):
                    v = iou(pred_box, gt_box)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= iou_thr and best_j >= 0 and not matched[best_j]:
                    tp_list.append((score, 1)); matched[best_j] = True
                else:
                    tp_list.append((score, 0))
        if n_gt == 0:
            aps.append(0.0); continue
        tp_list.sort(key=lambda x: -x[0])
        tp_acc = fp_acc = 0
        prec, rec = [], []
        for _, is_tp in tp_list:
            tp_acc += is_tp; fp_acc += (1 - is_tp)
            prec.append(tp_acc / (tp_acc + fp_acc))
            rec.append(tp_acc / n_gt)
        ap = sum(max((p for p, r in zip(prec, rec) if r >= t/10), default=0.0)
                 for t in range(11)) / 11
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0.0

def compute_precision_recall(all_preds, all_targets, num_cls=2):
    tp_total = fp_total = fn_total = 0
    for preds, targets in zip(all_preds, all_targets):
        for cls_id in range(1, num_cls + 1):
            gt_boxes = targets["boxes"][targets["labels"] == cls_id].tolist()
            mask = (preds["labels"] == cls_id) & (preds["scores"] >= CONF_THRESH)
            pb   = preds["boxes"][mask].tolist()
            matched = [False] * len(gt_boxes)
            for pred_box in pb:
                best_iou, best_j = 0.0, -1
                for j, gt_box in enumerate(gt_boxes):
                    v = iou(pred_box, gt_box)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= 0.5 and best_j >= 0 and not matched[best_j]:
                    tp_total += 1; matched[best_j] = True
                else:
                    fp_total += 1
            fn_total += matched.count(False)
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    return round(precision, 4), round(recall, 4)


if __name__ == "__main__":
    print(f"Оценка Faster R-CNN на тестовой выборке  [device={DEVICE}]")

    # Загрузка модели
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    test_ds     = DFireCOCODataset(COCO_DIR / "dfire_test.json", IMGSZ)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=2, collate_fn=collate_fn)

    all_preds, all_targets = [], []
    latencies = []

    with torch.no_grad():
        for images, targets in test_loader:
            images_dev = [img.to(DEVICE) for img in images]

            # Warmup первых 10 изображений не учитываем
            t0   = time.perf_counter()
            preds = model(images_dev)
            lat  = (time.perf_counter() - t0) * 1000

            if len(all_preds) >= 10:
                latencies.append(lat)

            all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds])
            all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

    map50      = compute_map50(all_preds, all_targets, num_cls=2)
    precision, recall = compute_precision_recall(all_preds, all_targets, num_cls=2)
    avg_lat    = sum(latencies) / len(latencies) if latencies else 0.0
    weight_mb  = WEIGHTS_PATH.stat().st_size / 1024 / 1024

    result = {
        "Модель":        "Faster R-CNN ResNet50",
        "mAP@0.5":       round(map50, 4),
        "mAP@0.5:0.95":  None,
        "Precision":     precision,
        "Recall":        recall,
        "Latency_ms":    round(avg_lat, 2),
        "Weights_MB":    round(weight_mb, 1),
        "Weights":       str(WEIGHTS_PATH),
    }

    print("\nРезультаты")
    for k, v in result.items():
        print(f"  {k}: {v}")

    with open(EVAL_DIR / "frcnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nМетрики сохранены: {EVAL_DIR / 'frcnn_metrics.json'}")