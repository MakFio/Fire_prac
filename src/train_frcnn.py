import json
import time
import mlflow
from pathlib import Path
from PIL import Image

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

COCO_DIR  = Path(r"D:\MLPrac\data\coco")
WORK_DIR  = Path(r"D:\MLPrac\outputs\runs\frcnn_dfire")
WORK_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES  = 3       # 0=background, 1=fire, 2=smoke
EPOCHS       = 5
BATCH_SIZE   = 16
LR           = 0.005
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0005
IMGSZ        = 640
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS  = 4

print(f"Device: {DEVICE}")
print(f"Torchvision: {torchvision.__version__}")


class DFireCOCODataset(Dataset):
    def __init__(self, ann_file: Path, imgsz: int = 640):
        with open(ann_file, encoding="utf-8") as f:
            coco = json.load(f)

        self.imgsz = imgsz
        self.images = {img["id"]: img for img in coco["images"]}

        # Группируем аннотации по image_id
        self.ann_by_img: dict[int, list] = {img_id: [] for img_id in self.images}
        for ann in coco["annotations"]:
            if ann["image_id"] in self.ann_by_img:
                self.ann_by_img[ann["image_id"]].append(ann)

        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_info = self.images[img_id]
        img_path = img_info["file_name"]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        img = img.resize((self.imgsz, self.imgsz))
        scale_x = self.imgsz / orig_w
        scale_y = self.imgsz / orig_h

        img_tensor = TF.to_tensor(img)  # [3, H, W], float32 [0,1]

        anns = self.ann_by_img[img_id]
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # Масштабируем в новый размер
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            # Защита от нулевых боксов
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            x2 = min(float(self.imgsz), x2); y2 = min(float(self.imgsz), y2)
            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann["category_id"]))  # 1=fire, 2=smoke

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([img_id]),
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def iou(box_a, box_b):
    xA = max(box_a[0], box_b[0]); yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2]); yB = min(box_a[3], box_b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = (box_a[2]-box_a[0])*(box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0])*(box_b[3]-box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def compute_map50(all_preds: list, all_targets: list, num_cls: int = 2, iou_thr: float = 0.5):
    """Возвращает mean AP@0.5 по классам 1..num_cls."""
    aps = []
    for cls_id in range(1, num_cls + 1):
        tp_list, scores_list, n_gt = [], [], 0
        for preds, targets in zip(all_preds, all_targets):
            gt_boxes = targets["boxes"][targets["labels"] == cls_id].tolist()
            n_gt += len(gt_boxes)
            mask = preds["labels"] == cls_id
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
                    tp_list.append((score, 1))
                    matched[best_j] = True
                else:
                    tp_list.append((score, 0))
                scores_list.append(score)
        if n_gt == 0:
            aps.append(0.0)
            continue
        tp_list.sort(key=lambda x: -x[0])
        tp_cumsum, fp_cumsum = [], []
        tp_acc = fp_acc = 0
        for _, is_tp in tp_list:
            tp_acc += is_tp; fp_acc += (1 - is_tp)
            tp_cumsum.append(tp_acc); fp_cumsum.append(fp_acc)
        prec = [tp / (tp + fp) if (tp + fp) > 0 else 0
                for tp, fp in zip(tp_cumsum, fp_cumsum)]
        rec  = [tp / n_gt for tp in tp_cumsum]
        # Интерполяция 11 точек
        ap = 0.0
        for thr in [t / 10 for t in range(11)]:
            p_max = max((p for p, r in zip(prec, rec) if r >= thr), default=0.0)
            ap += p_max / 11
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0.0


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///D:/MLPrac/outputs/mlruns")
    mlflow.set_experiment("fire_detection_comparison")

    train_ds = DFireCOCODataset(COCO_DIR / "dfire_train.json", IMGSZ)
    val_ds   = DFireCOCODataset(COCO_DIR / "dfire_val.json",   IMGSZ)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = build_model(NUM_CLASSES).to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[12, 17], gamma=0.1
    )

    best_map50 = 0.0
    history    = []

    with mlflow.start_run(run_name="Faster_R-CNN_ResNet50"):
        mlflow.log_params({
            "model": "fasterrcnn_resnet50_fpn", "epochs": EPOCHS,
            "batch": BATCH_SIZE, "lr": LR, "imgsz": IMGSZ,
            "optimizer": "SGD", "dataset": "D-Fire",
        })

        total_start = time.time()

        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            for step, (images, targets) in enumerate(train_loader):
                images  = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                loss_dict  = model(images, targets)
                total_loss = sum(loss_dict.values())

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                if (step + 1) % 100 == 0:
                    print(f"  Epoch {epoch}/{EPOCHS}  step {step+1}/{len(train_loader)}"
                          f"  loss={total_loss.item():.4f}")

            lr_scheduler.step()
            avg_loss = epoch_loss / len(train_loader)

            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(DEVICE) for img in images]
                    preds  = model(images)
                    all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds])
                    all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

            map50 = compute_map50(all_preds, all_targets, num_cls=2)
            mlflow.log_metrics({"loss": avg_loss, "val_mAP50": map50}, step=epoch)
            history.append({"epoch": epoch, "loss": avg_loss, "mAP50": map50})
            print(f"Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  val_mAP50={map50:.4f}")

            if map50 > best_map50:
                best_map50 = map50
                best_path = WORK_DIR / "best.pt"
                torch.save(model.state_dict(), best_path)
                print(f"  Новый best: mAP50={best_map50:.4f}, сохранён {best_path}")

            if epoch % 5 == 0:
                ckpt_path = WORK_DIR / f"epoch_{epoch}.pt"
                torch.save(model.state_dict(), ckpt_path)

        elapsed = time.time() - total_start
        mlflow.log_metrics({
            "best_mAP50":     best_map50,
            "train_time_min": round(elapsed / 60, 2),
        })
        print(f"\nОбучение завершено за {elapsed/60:.1f} мин.")
        print(f"Best mAP50 (val) = {best_map50:.4f}")
        print(f"Веса сохранены: {WORK_DIR / 'best.pt'}")

    with open(WORK_DIR / "train_history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_mAP50": best_map50,
                   "train_time_min": round(elapsed / 60, 2)}, f, indent=2)