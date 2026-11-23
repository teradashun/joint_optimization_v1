# python simulations/plot_skip_rate.py "$ROOT"

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

root = Path(sys.argv[1])
fig, ax = plt.subplots()

for path in sorted(root.glob("[A-F]/skip_rate.csv")):
    rate = pd.read_csv(path).at[0, "skip_rate_pct"]
    bar = ax.bar(path.parent.name, rate)
    ax.bar_label(bar, labels=[f"{rate:.1f}%"], padding=3)

ax.set(xlabel="Method", ylabel="Skipped updates (%)", ylim=(0, 110))
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(root / "skip_rate.png", dpi=300)