# train_ratio、alpha=0.5
# python -m src.FL --alpha 0.5 --method D --gamma 30

# train_ratio、alpha=1e10
# python -m src.FL --alpha 1e10 --method D --gamma 30

# random、alpha=0.5
# python -m src.FL --alpha 0.5 --method D --gamma 30

# random、alpha=1e10
# python -m src.FL --alpha 1e10 --method D --gamma 30

import copy
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .client import train
from .communication import transmit_update
from .dataset import get_datasets, split_dataset
from .models import MODEL_SPECS
from .round_resources import count_exit_flops, make_environment, plan_round
from .experiment import (
    evaluate_global, full_state, parse_args, output_directory, 
    print_resource_summary, save_round_metrics,
    record_round, shared_state
    )
from .server import federated_learning
from .utils import choose_clients, load_config, select_optimizer, set_seed


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    config = load_config(str(root))

    training        = config["training"]
    fl              = config["FL"]

    ite_num         = fl["ite_num"]
    global_rounds   = fl["global_rounds"]
    num_clients     = fl["num_clients"]
    cohort          = fl["cohort"]

    batch_size      = training["batch_size"]
    epochs          = training["epochs"]
    lr              = training["lr"]
    optimizer_name  = training["optim_name"]
    model_name      = training["model_name"]
    batch_speedup   = training["batch_speedup"]
    flops_per_cycle = training["flops_per_cycle"]
    temperature     = training["temperature"]

    spec = MODEL_SPECS[model_name]
    exits = spec["exit_ids"]

    communication = config["communication"]
    bandwidth_total = communication["total_bandwidth"]
    TXG = communication["TXG"]
    RXG = communication["RXG"]
    N_psd = communication["N_psd"]
    NF = communication["NF"]
    max_dis = communication["max_dis"]
    min_dis = communication["min_dis"]
    power = 10.0 ** ((communication["power_dbm"] - 30.0) / 10.0)

    cpu_hz = np.asarray([(1e9, 1e9, 1e9, 1e9, 1e9)[i % 5] for i in range(num_clients)])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _, test_loader = get_datasets(batch_size, "CIFAR10", val_ratio=0.0)

    if ite_num < 1:
        raise ValueError("ite_num must be positive")
    histories = []
    output = output_directory(args, cohort, num_clients)
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)

    for ite in range(ite_num):
        seed = ite + args.seed
        print(f"iteration {ite + 1}/{ite_num} (seed={seed})")

        # モデルを初期化する
        set_seed(seed)
        global_model = spec["factory"](num_classes=10).to(device)
        parameter_names = {name for name, _ in global_model.named_parameters()}
        flops = count_exit_flops(global_model)

        # 各Exitまでのモデルサイズを取得
        state = global_model.state_dict()

        size_bits = np.asarray([
            sum(
                tensor.numel() * tensor.element_size() * 8
                for name, tensor in state.items()
                if name.startswith(global_model.exit_state_prefixes[exit_id])
            )
            for exit_id in exits
        ], dtype=float)

        for index, exit_id in enumerate(exits):
            print(
                f"Exit{exit_id}: "
                f"{size_bits[index] / 1e6:.2f} Mbit, "
                f"{flops[exit_id]} FLOPs"
            )

        # 各端末の学習データとテストデータを準備する
        subsets = split_dataset(train_loader.dataset, num_clients, args.alpha)
        if any(len(subset) == 0 for subset in subsets):
            raise ValueError("学習データが空の端末があります")

        sample_counts = np.asarray([len(subset) for subset in subsets])

        client_loaders = [
                    DataLoader(
                        subset, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed + i),
                    )
                    for i, subset in enumerate(subsets)
                ]

        # 端末ごとの計算時間と、各ラウンドの通信条件を準備する
        comp = (
            sample_counts[:, None]
            * np.asarray([flops[e] for e in exits])[None, :]
            / (cpu_hz[:, None] * flops_per_cycle * batch_speedup)
        )
        _, channels, noises = make_environment(
            num_clients, global_rounds, bandwidth_total, seed,
            max_dis=max_dis, min_dis=min_dis, TXG=TXG, RXG=RXG, N_psd=N_psd, NF=NF,
        )

        # 参加端末・Exit・帯域幅を各ラウンドについて決める
        set_seed(seed)
        selected_clients = [
            choose_clients(num_clients, cohort) for _ in range(global_rounds)
        ]
        plans = [
            plan_round(
                args.method, comp[participants], channels[t, participants],
                float(noises[t]), power, size_bits, bandwidth_total, args.gamma,
            )
            for t, participants in enumerate(selected_clients)
        ]
        print_resource_summary(
            exits, comp, channels, noises, power, size_bits, bandwidth_total, plans,
            selected_clients,
        )

        if args.dry_run:
            continue

        last_accuracies = evaluate_global(global_model, test_loader, device, exits)

        initial_exit = {
            "C": 34,
            "D": 15,
            "E": 10,
            "F": 5,
            "G": 1
        }.get(args.method, exits[-1])

        history = [{
            "num_participants": 0,
            "round": 0,
            "accuracy": float(last_accuracies[initial_exit]),
            "updated": False,
            "skipped": False,
            "exit_index_sum": np.nan,
            "noise_w": np.nan,
            "uplink_mse": np.nan,
        }]

        for global_round in range(global_rounds):
            plan = plans[global_round]
            choices, _, updated, _ = plan
            participants = selected_clients[global_round]
            uplink_mses = {}

            if updated:
                set_seed(seed + global_round)
                current_shared = full_state(global_model)
                local_updates = []

                # 参加端末でローカル学習し、共有層を送信する
                for position, client_idx in enumerate(participants):
                    exit_id = exits[int(choices[position]) - 1]
                    client_loaders[client_idx].generator.manual_seed(seed + global_round + client_idx)

                    local_model = copy.deepcopy(global_model)
                    optimizer = select_optimizer(optimizer_name, local_model, lr)

                    for _ in range(epochs):
                        train(optimizer, local_model, client_loaders[client_idx], device, model_name, temperature, exit_id=exit_id)
                    
                    local_shared = shared_state(
                        local_model,
                        local_model.exit_state_prefixes[exit_id],
                    )
                    
                    if args.method == "A":
                        received_shared, client_mse = local_shared, 0.0
                    else:
                        received_shared, client_mse = transmit_update(
                            local_state=local_shared,
                            reference_state=current_shared,
                            parameter_names=parameter_names,
                            channel_gain=channels[global_round, client_idx],
                            power_w=power,
                            noise_w=float(noises[global_round]),
                            seed=[seed, global_round, client_idx],
                        )

                    local_updates.append(received_shared)
                    uplink_mses[client_idx] = client_mse

                # データ数で重み付けして共有層を集約する
                federated_learning(local_updates, global_model, sample_counts[participants].tolist())

                # 全端末のテストデータで評価する
                last_accuracies = evaluate_global(global_model, test_loader, device, exits)

            record_round(
                history, global_round + 1, choices, updated, last_accuracies,
                exits, noises[global_round], uplink_mses,
            )
            status = "更新" if updated else "スキップ"
            print(
                f"Round {global_round + 1}/{global_rounds}: "
                f"{status}, Accuracy={history[-1]['accuracy']:.3f}%"
            )


        histories.append(history)

    if not args.dry_run:
        save_round_metrics(output, histories, args, exits, cohort)
        print(f"結果保存先: {output}")


if __name__ == "__main__":
    main()
