# python simulations/plot_accuracy.py "$ROOT"

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

root = Path(sys.argv[1])
fig, ax = plt.subplots()

for path in sorted(root.glob("[A-F]/round_metrics.csv")):
    df = pd.read_csv(path).sort_values("round")
    ax.plot(df["round"], df["accuracy"], label=path.parent.name)

ax.set(xlabel="Global round", ylabel="Accuracy (%)", ylim=(0, 100))
ax.legend(title="Method")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(root / "accuracy.png", dpi=300)
fig.savefig(root / "accuracy.pdf")