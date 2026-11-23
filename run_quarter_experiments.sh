#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

read -rp "gamma [秒]: " GAMMA

SEED=100
RUN_ROOT="results"

for alpha in 0.5; do
    for method in A B C D E F; do
        echo "alpha=$alpha method=$method seed=$SEED"

        python3 -u -m src.FL \
            --alpha "$alpha" \
            --method "$method" \
            --gamma "$GAMMA" \
            --seed "$SEED" \
            --output-root "$RUN_ROOT"
    done
done

echo "全6パターンが完了しました。"
