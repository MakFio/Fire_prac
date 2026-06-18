import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

DATASET_ROOT = Path(r"D:\MLPrac\D-Fire")
OUT_DIR      = Path(r"D:\MLPrac\data\coco")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS_TXT = {
    "train": Path(r"D:\MLPrac\data\splits\train.txt"),
    "val":   Path(r"D:\MLPrac\data\splits\val.txt"),
    "test":  Path(r"D:\MLPrac\data\splits\test.txt"),
}

CATEGORIES = [
    {"id": 1, "name": "fire",  "supercategory": "hazard"},
    {"id": 2, "name": "smoke", "supercategory": "hazard"},
]

def convert_split(split_name, img_paths_txt):
    img_paths = [Path(p.strip()) for p in img_paths_txt.read_text(encoding="utf-8").splitlines() if p.strip()]

    images, annotations = [], []
    ann_id = 1

    for img_id, img_path in enumerate(tqdm(img_paths, desc=split_name), start=1):
        label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")

        try:
            img = Image.open(img_path)
            w, h = img.size
        except Exception:
            continue

        images.append({
            "id":        img_id,
            "file_name": str(img_path),
            "width":     w,
            "height":    h,
        })

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    continue

                # (x_min, y_min, width, height) в пикселях
                abs_w = bw * w
                abs_h = bh * h
                x_min = (xc - bw / 2) * w
                y_min = (yc - bh / 2) * h

                if abs_w <= 0 or abs_h <= 0:
                    continue

                annotations.append({
                    "id":           ann_id,
                    "image_id":     img_id,
                    "category_id":  cls_id + 1,  # 0→1 (fire), 1→2 (smoke)
                    "bbox":         [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)],
                    "area":         round(abs_w * abs_h, 2),
                    "iscrowd":      0,
                })
                ann_id += 1

    coco_dict = {
        "info":        {"description": f"D-Fire {split_name}", "version": "1.0"},
        "categories":  CATEGORIES,
        "images":      images,
        "annotations": annotations,
    }

    out_path = OUT_DIR / f"dfire_{split_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, ensure_ascii=False)

    print(f"[{split_name}] изображений: {len(images)}, аннотаций: {len(annotations)} → {out_path}")
    return out_path

for split, txt_path in SPLITS_TXT.items():
    convert_split(split, txt_path)

print("\nКонвертация завершена. Файлы в D:\\MLPrac\\data\\coco\\")