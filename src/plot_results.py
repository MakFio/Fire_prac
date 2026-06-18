import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

df       = pd.read_csv(r"D:\MLPrac\outputs\eval\comparison_table_5models.csv")
PLOT_DIR = Path(r"D:\MLPrac\outputs\plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

models = df["Модель"].tolist()
x      = np.arange(len(models))
width  = 0.35
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][:len(models)]

# mAP сравнение
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, df["mAP@0.5"],       width, label="mAP@0.5",       color="#e74c3c", alpha=0.9)
ax.bar(x + width/2, df["mAP@0.5:0.95"],  width, label="mAP@0.5:0.95",  color="#c0392b", alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.set_ylabel("mAP")
ax.set_title("Сравнение mAP по моделям")
ax.legend()
ax.set_ylim(0, 1.0)
for i, v in enumerate(df["mAP@0.5"]):
    ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / "map_comparison.png", dpi=150)
plt.close()
print("Сохранён: map_comparison.png")

# Precision / Recall
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, df["Precision"], width, label="Precision", color="#2980b9", alpha=0.9)
ax.bar(x + width/2, df["Recall"],    width, label="Recall",    color="#27ae60", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.set_ylabel("Значение")
ax.set_title("Precision и Recall по моделям")
ax.legend()
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(PLOT_DIR / "precision_recall.png", dpi=150)
plt.close()
print("Сохранён: precision_recall.png")

# Точность / Скорость
fig, ax = plt.subplots(figsize=(8, 5))
for i, row in df.iterrows():
    c = colors[i % len(colors)]
    ax.scatter(row["Latency_ms"], row["mAP@0.5"], s=180, c=c,
               zorder=3, edgecolors="white", linewidths=0.8)
    ax.annotate(row["Модель"], (row["Latency_ms"], row["mAP@0.5"]),
                textcoords="offset points", xytext=(8, 4), fontsize=9)
ax.set_xlabel("Latency (мс/изображение)")
ax.set_ylabel("mAP@0.5")
ax.set_title("Точность / Скорость инференса")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "accuracy_vs_latency.png", dpi=150)
plt.close()
print("Сохранён: accuracy_vs_latency.png")

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(models, df["Latency_ms"], color=colors)
ax.set_xlabel("Latency (мс/изображение)")
ax.set_title("Скорость инференса моделей")
for i, v in enumerate(df["Latency_ms"]):
    ax.text(v + 0.02, i, f"{v:.2f} мс", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(PLOT_DIR / "latency_comparison.png", dpi=150)
plt.close()
print("Сохранён: latency_comparison.png")

print(f"\nВсе графики сохранены в: {PLOT_DIR}")