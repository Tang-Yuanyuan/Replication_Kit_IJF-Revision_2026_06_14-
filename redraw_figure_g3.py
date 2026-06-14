"""Regenerate Figure G.3 (weighted prediction accuracy) from saved CSV data."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root       = Path(__file__).resolve().parent
w_out_dir  = root / "results" / "empirical4.1_4.4_weighted"
data_csv   = w_out_dir / "Figure_G.3_prediction_accuracy_data.csv"

if not data_csv.exists():
    sys.exit(f"Data not found: {data_csv}\nRun the weighted analysis first to generate it.")

w_acc = pd.read_csv(data_csv, index_col=0)["Accuracy"]

fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
x_pos = np.arange(len(w_acc))
bars = ax.bar(x_pos, w_acc.values, 0.4, color="black")
ax.set_ylim(60, max(w_acc.max() + 2, 62))
ax.set_ylabel("Prediction Accuracy (%)", fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(w_acc.index, rotation=15, ha="right", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.2f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5), textcoords="offset points",
        ha="center", va="bottom", fontsize=11)
fig.tight_layout()

out = w_out_dir / "Figure_G.3_prediction_accuracy.png"
fig.savefig(out, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved: {out}")
