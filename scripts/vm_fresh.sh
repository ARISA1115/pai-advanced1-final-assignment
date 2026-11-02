#!/bin/bash
# 新しい種(800000〜)で、画像で順番を進める方式を評価する
cd "$(dirname "$0")/.." && . .venv/bin/activate
for k in 0 1 2 3 4 5 6 7; do
  OMP_NUM_THREADS=1 python scripts/vm_eval.py --policy checkpoints/vm_dagger3.pt --episodes 200 --seed0 800000 --shard $k/8 --strict-planner --visual-progress --max-steps 800 --out results/vm_eval_fresh_part$k.json > results/vm_eval_fresh_part$k.txt 2>&1 &
done
wait
python scripts/vm_eval_summary.py vm_eval_fresh vm_eval_fresh "dagger3 + 画像で順番を進める(新しい種 800000〜)"
