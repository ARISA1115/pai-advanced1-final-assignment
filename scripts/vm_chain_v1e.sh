#!/bin/bash
# DAgger の追加: dagger5 の方策で動かし（種 1200000〜、全タスク）、失敗しやすい状態に正解を付けて再学習 → 種 800000〜 で評価
cd "$(dirname "$0")/.." && . .venv/bin/activate
echo "開始 $(date +%T)"
for k in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=1 python scripts/vm_collect.py --shard $k/8 --episodes 480 --seed0 1200000 --policy checkpoints/vm_dagger5.pt --out data/vm/dagger6_$k.npz > results/vm_dagger6_collect_$k.txt 2>&1 & done
wait
echo "収集の完了 $(date +%T)"
[ "$(ls data/vm/dagger6_*.npz | wc -l)" -eq 8 ] || { echo "失敗: シャードが足りない"; exit 1; }
python scripts/vm_train.py --data "data/vm/dagger3_*.npz" "data/vm/dagger4_*.npz" "data/vm/dagger5_*.npz" "data/vm/expert5_*.npz" "data/vm/dagger6_*.npz" --init checkpoints/vm_dagger5.pt --epochs ${EPOCHS:-8} --arch base --lr 2e-4 --out checkpoints/vm_dagger6.pt || { echo "失敗: 学習"; exit 1; }
echo "学習完了 $(date +%T)"
for k in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=1 python scripts/vm_eval.py --policy checkpoints/vm_dagger6.pt --episodes 200 --seed0 800000 --shard $k/8 --strict-planner --visual-progress --unstick --max-steps 800 --out results/vm_sel_d6_unstick_part$k.json > results/vm_sel_d6_unstick_part$k.txt 2>&1 & done
wait
python scripts/vm_eval_summary.py vm_sel_d6_unstick vm_sel_d6_unstick "dagger6 + 止まったら動かす（種 800000〜）"
echo "全部完了 $(date +%T)"
