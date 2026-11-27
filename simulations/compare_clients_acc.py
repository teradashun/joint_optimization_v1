from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
ALPHA = "0.3"
GAMMA = 10
TOTAL = 40
CLIENT_COUNTS = [1, 3, 5, 10, 40]
ERROR_FREE = RESULTS / "error_free"

fig, ax = plt.subplots(figsize=(6.4, 4.2))

for k in CLIENT_COUNTS:
    folder = RESULTS / f"{ALPHA}_{k}|{TOTAL}_gamma={GAMMA:g}" / "B"
    df = pd.read_csv(folder / "round_metrics.csv").sort_values("round")
    ax.plot(df["round"], df["accuracy"], label=f"参加端末数 = {k}")

ax.set(xlabel="Global round", ylabel="Accuracy (%)", ylim=(0, 100))
ax.legend(title="Method B")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / f"B_clients_accuracy_a{ALPHA}_g{GAMMA:g}_n{TOTAL}.png", dpi=300)
fig.savefig(RESULTS / f"B_clients_accuracy_a{ALPHA}_g{GAMMA:g}_n{TOTAL}.pdf")