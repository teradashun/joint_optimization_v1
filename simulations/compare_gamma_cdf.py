from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import argparse

RESULTS = Path(__file__).resolve().parents[1] / "results"
K = 20
CONDITION = f"0.3_{K}|40"
NUM_EXITS = 5
ERROR_FREE = RESULTS / "error_free"

parser = argparse.ArgumentParser()
parser.add_argument("--error-free", action="store_true", help="Error-freeのCDFを重ねて表示する")
args = parser.parse_args()

paths = RESULTS.glob(f"{CONDITION}_gamma=*/B/exit_sum_cdf.csv")
curves = sorted((float(p.parents[1].name.split("_gamma=")[1]), p) for p in paths)
if not curves:
    raise FileNotFoundError(f"{CONDITION} の手法Bの exit_sum_cdf.csv がありません")

fig, ax = plt.subplots(figsize=(6.4, 4.2))

for gamma, path in curves:
    df = pd.read_csv(path).dropna(subset=["cdf_pct"]).sort_values("exit_index_sum")

    # 描画用の末尾点（5*K+1）を除き、横軸を1台あたりの平均に変換
    df = df.loc[df["exit_index_sum"] <= NUM_EXITS * K]
    mean_exit = df["exit_index_sum"] / K

    label = rf"$\gamma={gamma:g}$ s"
    if df.empty:
        label += " (no updates)"
    ax.step(mean_exit, df["cdf_pct"], where="post", label=label)

if args.error_free:
    summary = pd.read_csv(ERROR_FREE / "skip_rate.csv").iloc[0]
    ref_k = int(summary["cohort"])
    n_exits = len(str(summary["exit_ids"]).split(","))

    ref = pd.read_csv(ERROR_FREE / "exit_sum_cdf.csv").dropna(subset=["cdf_pct"])
    ref = ref.loc[ref["exit_index_sum"] <= n_exits * ref_k].sort_values("exit_index_sum")

    ax.step(ref["exit_index_sum"] / ref_k, ref["cdf_pct"], where="post", color="black", linestyle="--", linewidth=2, label=f"エラーフリー")

ax.set(
    xlabel="1端末あたりの平均Exit番号",
    ylabel="CDF (%)",
    xlim=(0, NUM_EXITS),
    ylim=(0, 102),
)
ax.set_xticks(range(NUM_EXITS + 1))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / f"{CONDITION}_B_gamma_mean_exit_cdf.png", dpi=300)
fig.savefig(RESULTS / f"{CONDITION}_B_gamma_mean_exit_cdf.pdf")