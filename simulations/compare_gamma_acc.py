from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
CONDITION = "0.3_20|40"
ERROR_FREE = RESULTS / "error_free"

paths = RESULTS.glob(f"{CONDITION}_gamma=*/B/round_metrics.csv")
curves = sorted((float(p.parents[1].name.split("_gamma=")[1]), p) for p in paths)
if not curves:
    raise FileNotFoundError(f"{CONDITION} の手法Bの round_metrics.csv がありません")

fig, ax = plt.subplots(figsize=(6.4, 4.2))

for gamma, path in curves:
    df = pd.read_csv(path).sort_values("round")
    ax.plot(df["round"], df["accuracy"], label=rf"$\gamma={gamma:g}$ s")

ref = pd.read_csv(ERROR_FREE / "round_metrics.csv").sort_values("round")
ref_k = int(pd.read_csv(ERROR_FREE / "skip_rate.csv").at[0, "cohort"])

ax.plot(ref["round"], ref["accuracy"],color="black", linestyle="--", linewidth=2, label=f"エラーフリー")

ax.set(xlabel="グローバルラウンド", ylabel="テスト精度 (%)", ylim=(0, 100))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / f"{CONDITION}_B_gamma_accuracy.png", dpi=300)
fig.savefig(RESULTS / f"{CONDITION}_B_gamma_accuracy.pdf")