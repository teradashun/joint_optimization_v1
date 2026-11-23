from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
CONDITION = "0.5_20|40"

paths = RESULTS.glob(f"{CONDITION}_gamma=*/B/exit_sum_cdf.csv")
curves = sorted((float(p.parents[1].name.split("_gamma=")[1]), p) for p in paths)
if not curves:
    raise FileNotFoundError(f"{CONDITION} の手法Bの exit_sum_cdf.csv がありません")

fig, ax = plt.subplots(figsize=(6.4, 4.2))

for gamma, path in curves:
    df = pd.read_csv(path).dropna(subset=["cdf_pct"]).sort_values("exit_index_sum")
    label = rf"$\gamma={gamma:g}$ s"
    if df.empty:
        label += " (no updates)"
    ax.step(df["exit_index_sum"], df["cdf_pct"], where="post", label=label)

ax.set(xlabel="Sum of selected Exit indices", ylabel="Empirical CDF (%)",
       ylim=(0, 102))
ax.legend(title="Method B")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / f"{CONDITION}_B_gamma_cdf.png", dpi=300)