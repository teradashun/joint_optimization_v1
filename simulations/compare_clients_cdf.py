from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import argparse

RESULTS = Path(__file__).resolve().parents[1] / "results"
ALPHA = "0.3"
GAMMA = 10
TOTAL = 40
CLIENT_COUNTS = [1, 5, 10, 20, 40]
ERROR_FREE = RESULTS / "error_free"

parser = argparse.ArgumentParser()
parser.add_argument("--error-free", action="store_true", help="Error-freeのCDFを重ねて表示する")
args = parser.parse_args()

fig, ax = plt.subplots(figsize=(6.4, 4.2))
# right = 5 * max(CLIENT_COUNTS) + 1
right = 5

for k in CLIENT_COUNTS:
    folder = RESULTS / f"{ALPHA}_{k}|{TOTAL}_gamma={GAMMA:g}" / "B"
    df = pd.read_csv(folder / "exit_sum_cdf.csv")
    df = df.dropna(subset=["cdf_pct"]).sort_values("exit_index_sum")

    if df.empty:
        ax.plot([], [], label=f"K = {k}")
        continue

    df = df.loc[df["exit_index_sum"] <= right * k]
    x = df["exit_index_sum"] / k
    ax.step(x, df["cdf_pct"], where="post", label=f"K = {k}")


ax.set(xlabel="", ylabel="CDF (%)",xlim=(0, right), ylim=(0, 102))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / f"B_clients_cdf_a{ALPHA}_g{GAMMA:g}_n{TOTAL}.png", dpi=300)
fig.savefig(RESULTS / f"B_clients_cdf_a{ALPHA}_g{GAMMA:g}_n{TOTAL}.pdf")