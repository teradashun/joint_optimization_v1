import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .models import MODEL_SPECS


def train(optimizer, model, train_loader, device, model_name, temperature, frozen_prefixes=(), exit_id=None):
    # 評価関数の定義
    criterion = nn.CrossEntropyLoss()

    # model 学習モードに設定
    model.train()

    if frozen_prefixes:
        for name, module in model.named_modules():
            if name and (name + ".").startswith(frozen_prefixes):
                module.eval()

    for i, (images, labels) in enumerate(train_loader):
        if model_name in ["DNN"]:
            # viewで1次元配列に変更
            images, labels = images.view(-1, 28*28).to(device), labels.to(device)
        
        else:
            images, labels = images.to(device), labels.to(device)
        
        # 勾配をリセット
        optimizer.zero_grad(set_to_none=True)

        if model_name in MODEL_SPECS:
            outputs = model(images, exit_id=exit_id)
            base_arr = np.arange(1, len(outputs) + 1)
            arr = base_arr / np.sum(base_arr)

            loss = sum(weight * criterion(output, labels) for weight, output in zip(arr, outputs))

            if len(outputs) > 1:
                with torch.no_grad():
                    teacher_logits = torch.stack(outputs).mean(dim=0)
                    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

                # 各Exitの予測を教師分布に近づける
                kd_loss = sum(-(teacher_probs * F.log_softmax(output / temperature, dim=1)).sum(dim=1).mean()
                    for output in outputs) / len(outputs)

                loss = loss + temperature**2 * kd_loss

        
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 誤差逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()


def test(model, test_loader, device):

    # model 評価モードに設定
    model.eval()

    with torch.no_grad():
        correct_preds = 0
        total_preds = 0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if isinstance(outputs, (tuple, list)):
                outputs = outputs[-1]
            
            _, preds = torch.max(outputs, 1)
            correct_preds += preds.eq(labels).sum().item()
            total_preds += outputs.size(0)

        acc = 100*correct_preds/total_preds
    
    return acc


def early_test(model, test_loader, device, threshold, model_name):
    model.eval()
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if model_name in (
                "eenet20",
                "eenet34",
                "eenet34_1525",
                "eenet34_1525_half",
                "eenet34_1525_quarter",
            ):
                outputs = model.forward_early(inputs, threshold)
                preds = outputs.argmax(dim=1)

                correct_preds += preds.eq(labels).sum().item()
                total_preds += labels.size(0)

        acc = 100*correct_preds/total_preds
    
    return acc


def test_exit(model, test_loader, device, exit_layer):
    """指定したExitまでだけ計算して精度を評価する。"""
    allowed_exits = getattr(model, "exit_ids", (20, 28, 34))
    
    if exit_layer not in allowed_exits:
        raise ValueError(f"未対応のExit: {exit_layer}")

    model.eval()
    correct_preds = 0
    total_preds = 0

    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model.forward_exit(inputs, exit_layer)
            preds = outputs.argmax(dim=1)

            correct_preds += preds.eq(labels).sum().item()
            total_preds += labels.size(0)

    if total_preds == 0:
        raise ValueError("評価データがありません")

    return 100.0 * correct_preds / total_preds


def fine_tune_personal(
    optimizer,
    model,
    train_loader,
    device,
    model_name,
    shared_prefixes,
    epochs=1,
):
    original_flags = {name: p.requires_grad for name, p in model.named_parameters()}

    try:
        # 共有部分の重みを固定
        for name, p in model.named_parameters():
            if name.startswith(shared_prefixes):
                p.requires_grad_(False)

        for _ in range(epochs):
            train(optimizer, model, train_loader, device, model_name, frozen_prefixes=shared_prefixes)

    finally:
        # 次回の通常学習に向けて元に戻す
        for name, p in model.named_parameters():
            p.requires_grad_(original_flags[name])