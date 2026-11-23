from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
ALPHA = "0.5"
GAMMA = 10
TOTAL = 40
CLIENT_COUNTS = [1, 5, 10, 20, 40]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
right = 4 * max(CLIENT_COUNTS) + 1

for k in CLIENT_COUNTS:
    folder = RESULTS / f"{ALPHA}_{k}|{TOTAL}_gamma={GAMMA:g}" / "B"
    df = pd.read_csv(folder / "exit_sum_cdf.csv")
    df = df.dropna(subset=["cdf_pct"]).sort_values("exit_index_sum")

    if df.empty:
        ax.plot([], [], label=f"K = {k} (no updates)")
        continue

    x = df["exit_index_sum"].tolist()
    y = df["cdf_pct"].tolist()
    ax.step(x + [right], y + [y[-1]], where="post", label=f"K = {k}")

ax.set(xlabel="Sum of selected Exit indices", ylabel="Empirical CDF (%)",
       xlim=(0, right), ylim=(0, 102))
ax.legend(title="Method B")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / f"B_clients_cdf_a{ALPHA}_g{GAMMA:g}_n{TOTAL}.png", dpi=300)