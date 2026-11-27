from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

root = Path(sys.argv[1])
methods = []
rates_by_method = []

for path in sorted(root.glob("[A-G]/round_metrics.csv")):
    df = pd.read_csv(path, usecols=["round", "skipped"])

    # ラウンド0は学習前の初期評価なので除く
    rates = 100 * df.loc[df["round"] > 0, "skipped"]
    if rates.empty:
        raise ValueError(f"学習ラウンドの記録がありません: {path}")

    methods.append(path.parent.name)
    rates_by_method.append(rates.to_numpy())

if not methods:
    raise FileNotFoundError(f"手法A〜Gの round_metrics.csv がありません: {root}")

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.boxplot(rates_by_method, widths=0.55)
ax.set_xticks(range(1, len(methods) + 1), methods)
ax.set(xlabel="Method", ylabel="Skip rate per round (%)", ylim=(-5, 105))
ax.set_yticks(range(0, 101, 20))
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(root / "skip_rate_boxplot.png", dpi=300)
fig.savefig(root / "skip_rate_boxplot.pdf")