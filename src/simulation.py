# pip install torch torchvision
import copy
import statistics
import time

import torch
from torch import nn
from torchvision.models import resnet34

NUM_SAMPLES = 256
BATCH_SIZES = (1, 8, 32, 64)
CPU_THREADS = 4
REPEATS = 3

torch.set_num_threads(CPU_THREADS)
torch.manual_seed(0)

# データ生成とモデル初期化は計測外。すべての条件で同じデータを使う。
images = torch.randn(NUM_SAMPLES, 3, 32, 32)
labels = torch.randint(0, 10, (NUM_SAMPLES,))

base_model = resnet34(weights=None, num_classes=10)
base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
base_model.maxpool = nn.Identity()
initial_state = copy.deepcopy(base_model.state_dict())
loss_fn = nn.CrossEntropyLoss()


def measure(batch_size):
    model = copy.deepcopy(base_model).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batches = list(zip(images.split(batch_size), labels.split(batch_size)))

    def train_step(x, y):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    # バッチサイズごとにウォームアップし、計測前に初期状態へ戻す。
    for _ in range(3):
        train_step(*batches[0])

    times = []
    for _ in range(REPEATS):
        model.load_state_dict(initial_state)
        start = time.perf_counter()
        for x, y in batches:
            train_step(x, y)
        times.append(time.perf_counter() - start)

    return statistics.median(times)


results = {batch_size: measure(batch_size) for batch_size in BATCH_SIZES}
baseline = results[1]

print(f"CPU threads={CPU_THREADS}, samples={NUM_SAMPLES}")
print(f"{'Batch':>5} {'Total [s]':>12} {'ms/sample':>12} {'Speedup':>9}")
for batch_size, elapsed in results.items():
    print(
        f"{batch_size:>5} {elapsed:>12.3f} "
        f"{1000 * elapsed / NUM_SAMPLES:>12.3f} {baseline / elapsed:>8.2f}x"
    )