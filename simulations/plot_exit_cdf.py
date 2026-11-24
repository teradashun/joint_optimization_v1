# python simulations/plot_exit_cdf.py "$ROOT"

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

root = Path(sys.argv[1])
fig, ax = plt.subplots()

for path in sorted(root.glob("[A-F]/exit_sum_cdf.csv")):
    df = pd.read_csv(path).dropna(subset=["cdf_pct"]).sort_values("exit_index_sum")
    method = path.parent.name
    label = f"{method} (no updates)" if df.empty else method
    ax.step(df["exit_index_sum"], df["cdf_pct"], where="post", label=label)

ax.set(xlabel="Sum of selected Exit indices", ylabel="Empirical CDF (%)", ylim=(0, 102))
ax.legend(title="Method")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(root / "exit_sum_cdf.png", dpi=300)