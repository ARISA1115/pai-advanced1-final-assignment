#!/bin/bash
# 止まったら少し動かす処理を加えて評価する（種 800000〜、dagger3 と同じ設定に --unstick を追加）
cd "$(dirname "$0")/.." && . .venv/bin/activate
echo "開始 $(date +%T)"
for k in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=1 python scripts/vm_eval.py --policy checkpoints/vm_dagger3.pt --episodes 200 --seed0 800000 --shard $k/8 --strict-planner --visual-progress --unstick --max-steps 800 --out results/vm_sel_d3_unstick_part$k.json > results/vm_sel_d3_unstick_part$k.txt 2>&1 & done
wait
python scripts/vm_eval_summary.py vm_sel_d3_unstick vm_sel_d3_unstick "dagger3 + 止まったら動かす（種 800000〜）"
echo "完了 $(date +%T)"
