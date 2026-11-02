#!/bin/bash
# DAgger の 4 回目: dagger3 の方策で動かし（種 900000〜）、正解を付けて、全データで再学習 → 新しい種で評価
cd "$(dirname "$0")/.." && . .venv/bin/activate
echo "開始: $(date +%T)"
for k in 0 1 2 3 4 5 6 7; do python scripts/vm_collect.py --shard $k/8 --episodes 320 --seed0 900000 --policy checkpoints/vm_dagger3.pt --out data/vm/dagger4_$k.npz > results/vm_dagger4_collect_$k.txt 2>&1 & done
wait
echo "収集完了: $(date +%T)"
python scripts/vm_train.py --data "data/vm/expert_*.npz" "data/vm/expert2_*.npz" "data/vm/dagger2_*.npz" "data/vm/dagger3_*.npz" "data/vm/dagger4_*.npz" --init checkpoints/vm_dagger3.pt --epochs ${EPOCHS:-12} --arch base --lr 3e-4 --out checkpoints/vm_dagger4.pt || { echo "失敗: 学習"; exit 1; }
echo "学習完了: $(date +%T)"
for k in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=1 python scripts/vm_eval.py --policy checkpoints/vm_dagger4.pt --episodes 200 --seed0 800000 --shard $k/8 --strict-planner --visual-progress --max-steps 800 --out results/vm_eval_d4_fresh_part$k.json > results/vm_eval_d4_fresh_part$k.txt 2>&1 & done
wait
python scripts/vm_eval_summary.py vm_eval_d4_fresh vm_eval_d4_fresh "dagger4 + 画像で順番を進める(新しい種 800000〜)"
echo "全部完了: $(date +%T)"
