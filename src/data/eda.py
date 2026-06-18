from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

DATASET_ROOT = Path(r"D:\MLPrac\D-Fire")
PLOTS_OUT    = Path(r"D:\MLPrac\outputs\plots")
PLOTS_OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "fire", 1: "smoke"}

SPLIT_PATHS = {
    "train": {
        "images": DATASET_ROOT / "train" / "images",
        "labels": DATASET_ROOT / "train" / "labels",
    },
    "test": {
        "images": DATASET_ROOT / "test" / "images",
        "labels": DATASET_ROOT / "test" / "labels",
    },
}

def analyze_split(split_name, img_dir, label_dir):
    img_dir   = Path(img_dir)
    label_dir = Path(label_dir)

    label_files = sorted(label_dir.glob("*.txt"))
    image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    class_counts  = Counter()
    bbox_areas    = []
    empty_count   = 0
    invalid_count = 0

    for lf in label_files:
        content = lf.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            empty_count += 1
            continue
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                invalid_count += 1
                continue
            try:
                cls = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                invalid_count += 1
                continue

            # Проверка корректности bbox
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                invalid_count += 1
                continue

            class_counts[CLASS_NAMES.get(cls, f"unknown_{cls}")] += 1
            bbox_areas.append(bw * bh)

    print(f"  Сплит: {split_name.upper()}\n")
    print(f"  Изображений:        {len(image_files)}")
    print(f"  Файлов разметки:    {len(label_files)}")
    print(f"  Пустых аннотаций:   {empty_count}")
    print(f"  Некорректных bbox:  {invalid_count}")
    print(f"  Bbox по классам:    {dict(class_counts)}")

    if bbox_areas:
        arr = np.array(bbox_areas)
        print(f"  Площадь bbox:")
        print(f"    min  = {arr.min():.5f}")
        print(f"    mean = {arr.mean():.5f}")
        print(f"    max  = {arr.max():.5f}")
        small = (arr < 0.01).sum()
        print(f"  Малых объектов (<1% кадра): {small} ({100*small/len(arr):.1f}%)")

    return class_counts, bbox_areas

all_counts = {}
all_areas  = {}
for sp, paths in SPLIT_PATHS.items():
    counts, areas = analyze_split(sp, paths["images"], paths["labels"])
    all_counts[sp] = counts
    all_areas[sp]  = areas

splits = list(SPLIT_PATHS.keys())
fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4))
if len(splits) == 1:
    axes = [axes]

for i, sp in enumerate(splits):
    names = [CLASS_NAMES[k] for k in sorted(CLASS_NAMES.keys()) if CLASS_NAMES[k] in all_counts[sp]]
    vals  = [all_counts[sp].get(n, 0) for n in names]
    axes[i].bar(names, vals, color=["#e74c3c", "#95a5a6"], alpha=0.9, edgecolor="white")
    axes[i].set_title(f"Баланс классов - {sp}", fontsize=12)
    axes[i].set_ylabel("Число bbox")
    for j, v in enumerate(vals):
        axes[i].text(j, v + max(vals)*0.01, str(v), ha="center", fontsize=10)

plt.tight_layout()
out_path = PLOTS_OUT / "class_balance.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nГрафик сохранён: {out_path}")

fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4))
if len(splits) == 1:
    axes = [axes]

for i, sp in enumerate(splits):
    areas = all_areas[sp]
    if areas:
        axes[i].hist(areas, bins=50, color="#e74c3c", alpha=0.75, edgecolor="white")
        axes[i].axvline(0.01, color="black", linestyle="--", linewidth=1.2, label="1% кадра")
        axes[i].set_title(f"Распределение площадей bbox - {sp}", fontsize=11)
        axes[i].set_xlabel("Нормализованная площадь (w×h)")
        axes[i].set_ylabel("Количество")
        axes[i].legend()

plt.tight_layout()
out_path2 = PLOTS_OUT / "bbox_area_distribution.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"График сохранён: {out_path2}")
print("\nEDA завершён успешно.")