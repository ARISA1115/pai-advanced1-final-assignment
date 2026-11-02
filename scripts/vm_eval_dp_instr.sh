#!/bin/bash
# 拡散モデル改良版（epoch 4 で打ち切り）の評価と、日本語の指示からの評価（dagger3）
cd "$(dirname "$0")/.." && . .venv/bin/activate
echo "開始 $(date +%T)"
for k in 0 1 2; do OMP_NUM_THREADS=1 python scripts/vm_instruction.py --policy checkpoints/vm_dagger3.pt --shard $k/3 --out results/vm_instruction_d3_part$k.json > results/vm_instruction_d3_part$k.txt 2>&1 & done
for k in 0 1 2 3 4 5; do OMP_NUM_THREADS=1 python scripts/vm_eval.py --policy checkpoints/vm3_dpg_e4.pt --episodes 200 --seed0 800000 --shard $k/6 --strict-planner --visual-progress --unstick --max-steps 800 --out results/vm3g_eval_fresh_part$k.json > results/vm3g_eval_fresh_part$k.txt 2>&1 & done
wait
python scripts/vm_eval_summary.py vm3g_eval_fresh vm3g_eval_fresh "拡散モデル（握る・開くは分類、epoch 4 で打ち切り）+ 止まったら動かす（種 800000〜）"
echo "全部完了 $(date +%T)"
