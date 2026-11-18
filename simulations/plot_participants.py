import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

if __package__:
    from .plot_common import (
        experiment_directories,
        make_title,
        cohort_size,
        plot_settings,
    )
else:
    from plot_common import (
        experiment_directories,
        make_title,
        cohort_size,
        plot_settings,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", choices=("0.5", "1e10"))
    args = parser.parse_args(argv)

    plt.rcParams.update({
        "font.size": 10,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })

    for alpha, directory in experiment_directories(
        args.root, args.alpha
    ):
        _, methods = plot_settings(directory)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        reference_rounds = None
        max_cohort = 0

        for method, (label, color, style) in methods.items():
            path = directory / method / "round_metrics.csv"
            df = pd.read_csv(path)
            df = df.loc[df["round"] > 0]

            rounds = df["round"].to_numpy()
            counts = pd.to_numeric(
                df["num_participants"], errors="raise"
            ).to_numpy(dtype=float)
            cohort = cohort_size(directory, method)
            max_cohort = max(max_cohort, cohort)

            if len(df) == 0 or not np.array_equal(
                rounds, np.arange(1, len(df) + 1)
            ):
                raise ValueError(f"ラウンド番号が不正です: {path}")

            if reference_rounds is None:
                reference_rounds = rounds
            elif not np.array_equal(rounds, reference_rounds):
                raise ValueError("手法間でラウンド番号が異なります")

            if (
                not np.isfinite(counts).all()
                or ((counts < 0) | (counts > cohort)).any()
                or not np.equal(counts, np.floor(counts)).all()
            ):
                raise ValueError(f"参加台数が不正です: {path}")

            ax.plot(
                rounds,
                counts,
                label=label,
                color=color,
                linestyle=style,
                linewidth=1.6,
                marker="." if len(rounds) == 1 else None,
            )

        ax.set(
            title=make_title(directory, alpha),
            xlabel="Global round",
            ylabel="Clients participating in global update",
            xlim=(0, reference_rounds[-1] + 0.5),
            ylim=(-0.5, max_cohort + 0.5),
        )
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(ncol=2)

        output = directory / "figures"
        output.mkdir(parents=True, exist_ok=True)

        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(
                output / f"participants.{extension}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)


if __name__ == "__main__":
    main()