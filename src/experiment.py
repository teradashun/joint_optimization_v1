"""引数の読み込み・評価・記録。準備と学習の流れは FL.py の main() を参照。"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .client import test_exit
from .round_resources import gamma_boundary


def full_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def shared_state(model, prefixes):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--alpha", type=float, choices=(0.5, 1e10), required=True)
    parser.add_argument("--method", choices=("A", "B", "C", "D", "E", "F"), required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true", help="学習せずに資源割当とスキップ数を確認する")

    args = parser.parse_args(argv)

    if not math.isfinite(args.gamma) or args.gamma <= 0:
        raise ValueError("gammaは有限の正の値が必要です")

    if args.seed < 0:
        raise ValueError("seedは非負の値が必要です")

    return args


def print_resource_summary(exits, comp, channels, noises, power, size_bits, bandwidth_total, plans, selected_clients):
    """学習前に、各ラウンドの参加端末の資源割当とgamma境界を確認する。"""
    rounds = len(plans)

    for index, exit_id in enumerate(exits):
        print(f"Exit{exit_id}: 最大計算遅延={comp[:, index].max():.3f}秒")

    predicted_skips = sum(not plan[2] for plan in plans)
    print(f"更新スキップ予定: {predicted_skips}/{rounds}")

    gamma_rows = []

    print("\n===== 通信 + 計算を含む最小gamma =====")

    for t in range(rounds):
        row = {"round": t + 1}
        participants = selected_clients[t]

        for index, exit_id in enumerate(exits):
            row[f"gamma_exit{exit_id}_s"] = gamma_boundary(
                comp_one_exit=comp[participants, index],
                gains=channels[t, participants],
                noise=float(noises[t]),
                power=power,
                size_bits=float(size_bits[index]),
                total_bandwidth=bandwidth_total,
            )

        gamma_rows.append(row)

        values = ", ".join(
            f"Exit{exit_id}={row[f'gamma_exit{exit_id}_s']:.3f}秒" for exit_id in exits
        )
        print(f"Round {t + 1}: {values}")

    gamma_df = pd.DataFrame(gamma_rows)

    print("\n===== gamma境界の分布 =====")

    for exit_id in exits:
        values = gamma_df[f"gamma_exit{exit_id}_s"].to_numpy()

    shallow_max = gamma_df[f"gamma_exit{exits[0]}_s"].max()

    deep_max = gamma_df[f"gamma_exit{exits[-1]}_s"].max()


def output_directory(args, cohort, num_clients):
    alpha = "1e10" if args.alpha == 1e10 else f"{args.alpha:g}"
    condition = f"{alpha}_{cohort}|{num_clients}_gamma={args.gamma:g}"
    return args.output_root / condition / args.method


def evaluate_global(global_model, test_loader, device, exits):
    """共通テストデータ全体で、グローバルモデルの各Exitを評価する。"""
    accuracies = {}

    for exit_id in exits:
        accuracies[exit_id] = test_exit(global_model, test_loader, device, exit_id)
    return accuracies


def make_train_loaders(subsets, batch_size, seed, global_round):
    """手法間で学習乱数を揃え、各ラウンドで同じ順序のデータを使う。"""
    return [
        DataLoader(
            subset, batch_size=batch_size, shuffle=True, drop_last=True,
            generator=torch.Generator().manual_seed(seed + global_round + i),
        )
        for i, subset in enumerate(subsets)
    ]


def record_round(history, global_round, choices, updated, accuracies, exits, noise, uplink_mses):
    if updated:
        accuracy = float(np.mean([accuracies[exits[int(choice) - 1]] for choice in choices]))
    else:
        accuracy = history[-1]["accuracy"]
    history.append({
        "round": global_round,
        "accuracy": accuracy,
        "updated": bool(updated),
        "skipped": not bool(updated),
        "num_participants": len(uplink_mses),
        "exit_index_sum": int(np.sum(choices)) if updated else np.nan,
        "noise_w": float(noise),
        "uplink_mse": np.mean(list(uplink_mses.values())) if uplink_mses else np.nan,
    })


def save_round_metrics(output, histories, args, exits, cohort):
    trials = [pd.DataFrame(history) for history in histories]
    means = pd.concat(trials).groupby("round", as_index=False).mean()
    means.to_csv(output / "round_metrics.csv", index=False)

    thresholds = np.arange(len(exits) * cohort + 2)
    cdfs = []
    skip_rates = []
    for trial in trials:
        rounds = trial.loc[trial["round"] > 0]
        skip_rates.append(100 * rounds["skipped"].mean())
        values = np.sort(rounds.loc[rounds["updated"], "exit_index_sum"])
        if values.size:
            counts = np.searchsorted(values, thresholds, side="right")
            cdfs.append(100 * counts / values.size)

    pd.DataFrame({
        "exit_index_sum": thresholds,
        "cdf_pct": np.mean(cdfs, axis=0) if cdfs else np.nan,
    }).to_csv(output / "exit_sum_cdf.csv", index=False)
    pd.DataFrame([{
        "skip_rate_pct": np.mean(skip_rates),
        "iterations": len(trials),
        "iterations_with_updates": len(cdfs),
        "alpha": args.alpha,
        "gamma": args.gamma,
        "cohort": cohort,
        "exit_ids": ",".join(map(str, exits)),
    }]).to_csv(output / "skip_rate.csv", index=False)

