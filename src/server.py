import torch
import math
import numpy as np


def average_models(client_updates, num_samples):
    # 層ごとに集約
    if not client_updates or len(client_updates) != len(num_samples):
        raise ValueError("更新数とデータ数のリストを確認してください")

    if any(n <= 0 for n in num_samples):
        raise ValueError("各端末のデータ数は正の値が必要です")

    # 全てのキーに対してNoneを割り当てる
    """
    例：
    クライアント1のキー: conv1.weight, conv1.bias, exit1.weight
    クライアント2のキー: conv1.weight, conv1.bias, exit2.weight
    {
        "conv1.weight": None,
        "conv1.bias": None,
        "exit1.weight": None,
        "exit2.weight": None,
    }
    """
    keys = dict.fromkeys(key for update in client_updates for key in update) 

    avg_state = {}

    for key in keys:
        """
        contributors: 層ごとに、その層の学習に参加したクライアントのデータだけを抽出する
        [
            (tensor_A, 100),  # クライアント1の重みとサンプル数
            (tensor_C, 300),  # クライアント2の重みとサンプル数
        ]
        """
        contributors = [
            (update[key], n) for update, n in zip(client_updates, num_samples) if key in update
        ]
        ref = contributors[0][0]

        if not torch.is_floating_point(ref):
            avg_state[key] = ref.detach().clone()
            continue
        
        total = sum(n for _, n in contributors)
        acc = torch.zeros_like(ref, dtype=torch.float32)

        for tensor, n in contributors:
            acc += tensor.detach().float() * n
        
        avg_state[key] = (acc / total).to(ref.dtype)

    return avg_state


def federated_learning(client_updates, global_model, num_samples):
    avg_state = average_models(client_updates, num_samples)
    global_model.load_state_dict(avg_state, strict=False)
    return global_model