import os
import yaml
import random
import numpy as np
import torch
import torch.optim as optim


def load_config(dir):   #引数のパスはFedAvg
    config_path = os.path.join(dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_optimizer(optim_name, local_model, lr):
    if optim_name == "SGD":
        return optim.SGD(local_model.parameters(), lr=lr)
    
    elif optim_name == "Adam":
        return optim.Adam(local_model.parameters(), lr=lr)
    
    else:
        raise NameError(f"オプティマイザの名前 '{optim_name}' はサポートされていません")


def choose_clients(num_clients, cohort):
    return random.sample(range(num_clients), cohort)


def acc_average(acc_history: list[list]) -> list:
    acc_history = np.array(acc_history)
    final_acc = np.mean(acc_history, axis=1)
    return final_acc


def generate_client_distances(
    num_clients,
    mean=0.0,
    std=150.0,
    min_distance=35.0,
    max_distance=300.0,
):
    distances = []

    while len(distances) < num_clients:

        # Step 1: X, Yを正規分布から生成
        x = np.random.normal(mean, std)
        y = np.random.normal(mean, std)

        # Step 2: 基地局(0, 0)からの距離
        distance = np.sqrt(x**2 + y**2)

        # Step 3: 35m以上1000m以下なら採用
        if min_distance <= distance <= max_distance:
            distances.append(distance)

    return distances


def generate_channel_gain(distance_m, RXG=0.0, TXG=0.0):
    # m → km
    distance_km = distance_m / 1000.0

    # RIS.pyと同じパスロスモデル
    PL_dB = 128.1 + 37.6 * np.log10(distance_km)

    BS_Gain = 10 ** (RXG / 10)
    User_Gain = 10 ** (TXG / 10)

    G_link_lin = (
        BS_Gain
        * User_Gain
        * 10 ** (-PL_dB / 10.0)
    )

    # パスロスによる振幅減衰
    path_gain = np.sqrt(G_link_lin)

    # Rayleigh fading
    h = (
        np.random.randn()
        + 1j * np.random.randn()
    ) / np.sqrt(2)

    return path_gain * h