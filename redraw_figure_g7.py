"""Regenerate Figure G.7 from saved CSV data with left y-axis starting at 10."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
w_sim_dir    = root / "results" / "empirical4.5_weighted"
w_output_dir = root / "results" / "empirical4.1_4.4_weighted"

w_sim_summary = pd.read_csv(w_sim_dir / "Figure_G.7_knowledge_growth_summary.csv", index_col=0)
w_df_plot     = pd.read_csv(w_output_dir / "Figure_G.4_assignment_outcomes_data.csv", index_col=0)

w_sim_order = [
    ("random_sm",    "Random assignment"),
    ("eco_all_sm",   "Logistic regression II"),
    ("eco_demos_sm", "Logistic regression I"),
    ("xgb_all_sm",   "XGBoost algorithm II"),
    ("xgb_demos_sm", "XGBoost algorithm I"),
    ("ideal_sm",     "Perfect assignment"),
]
w_sim_labels  = [lb for _, lb in w_sim_order]
w_bl_cost     = w_df_plot.loc[w_sim_labels, "Cost"].values
w_bl_rate     = w_df_plot.loc[w_sim_labels, "Rate"].values
w_cf_cost     = np.array([w_sim_summary.loc[k, "Cost_Mean"]  for k, _ in w_sim_order])
w_cf_cost_err = np.array([2 * w_sim_summary.loc[k, "Cost_SD"] for k, _ in w_sim_order])
w_cf_rate     = np.array([w_sim_summary.loc[k, "Rate_Mean"]  for k, _ in w_sim_order])
w_cf_rate_low = np.array([w_sim_summary.loc[k, "Rate_Low"]   for k, _ in w_sim_order])
w_cf_rate_hi  = np.array([w_sim_summary.loc[k, "Rate_High"]  for k, _ in w_sim_order])

fig, ax1 = plt.subplots(figsize=(14, 7), dpi=120)
ax2 = ax1.twinx()
xp = np.arange(len(w_sim_labels))
wd = 0.32

ax1.bar(xp - wd / 2, w_bl_cost, wd, color="#d9d9d9", edgecolor="black", label="Benchmark Average Compensation")
ax1.bar(xp + wd / 2, w_cf_cost, wd, color="#666666", edgecolor="black",
        yerr=w_cf_cost_err, capsize=4, label="Counterfactual Average Compensation")
ax2.plot(xp, w_bl_rate,  color="gray",  linestyle="--", marker="o", label="Benchmark Acceptance Rate")
ax2.fill_between(xp, w_cf_rate_low, w_cf_rate_hi, color="gray", alpha=0.2, label="95% Confidence Interval")
ax2.plot(xp, w_cf_rate, color="black", linestyle="-",  marker="s", label="Counterfactual Acceptance Rate")

for idx, val in enumerate(w_bl_cost):
    ax1.annotate(f"{val:.2f}", xy=(idx - wd / 2, val), xytext=(0, 5),
                 textcoords="offset points", ha="center", fontsize=9)
for idx, val in enumerate(w_cf_cost):
    ax1.annotate(f"{val:.2f}", xy=(idx + wd / 2, val), xytext=(0, 5),
                 textcoords="offset points", ha="center", fontsize=9)
for idx, val in enumerate(w_cf_rate):
    ax2.annotate(f"{val:.2f}%", xy=(idx, val), xytext=(0, 8),
                 textcoords="offset points", ha="center", fontsize=9, weight="bold")

ax1.set_ylim(15, max(w_bl_cost.max(), w_cf_cost.max()) + 2)
ax1.set_ylabel("Average Compensation (¥)", fontsize=12)
ax2.set_ylabel("Acceptance Rate (%)", fontsize=12)
ax1.set_xticks(xp)
ax1.set_xticklabels(w_sim_labels, rotation=15, ha="right")
ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
fig.tight_layout()

out = w_sim_dir / "Figure_G.7_knowledge_growth.png"
fig.savefig(out, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved: {out}")
