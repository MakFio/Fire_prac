import random
from pathlib import Path

DATASET_ROOT = Path(r"D:\MLPrac\D-Fire")
VALID_SRC    = Path(r"D:\MLPrac\valid")
SPLITS_OUT   = Path(r"D:\MLPrac\data\splits")
SPLITS_OUT.mkdir(parents=True, exist_ok=True)

SEED     = 42
VAL_FRAC = 0.15

print("Содержимое D:\\MLPrac\\valid\\ ")
for f in sorted(VALID_SRC.glob("*.txt")):
    lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
    print(f"  {f.name}: {len(lines)} строк")

train_imgs = sorted(list((DATASET_ROOT / "train" / "images").glob("*.jpg")) +
                    list((DATASET_ROOT / "train" / "images").glob("*.png")))
test_imgs  = sorted(list((DATASET_ROOT / "test"  / "images").glob("*.jpg")) +
                    list((DATASET_ROOT / "test"  / "images").glob("*.png")))

print(f"\nИзображений в train: {len(train_imgs)}")
print(f"Изображений в test:  {len(test_imgs)}")

random.seed(SEED)
random.shuffle(train_imgs)
val_size   = int(len(train_imgs) * VAL_FRAC)
val_imgs   = train_imgs[:val_size]
train_imgs_final = train_imgs[val_size:]

print(f"\nПосле разбивки (seed={SEED}, val={VAL_FRAC*100:.0f}%):")
print(f"  train: {len(train_imgs_final)}")
print(f"  val:   {len(val_imgs)}")
print(f"  test:  {len(test_imgs)}")

def write_split(name, paths):
    out = SPLITS_OUT / f"{name}.txt"
    out.write_text("\n".join(str(p) for p in paths), encoding="utf-8")
    print(f"Записан: {out}")

write_split("train", train_imgs_final)
write_split("val",   val_imgs)
write_split("test",  test_imgs)
print("\nСплиты сформированы успешно.")