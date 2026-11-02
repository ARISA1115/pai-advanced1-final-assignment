#!/bin/bash
# 夜間の連続作業: (A) 診断の評価（順番をオラクルで進める）と (B) v2 の DAgger 再学習 → 評価 を、同時に走らせる。
# スリープを防ぐため、caffeinate の下で実行する:  nohup caffeinate -is bash scripts/vm_night.sh > results/vm_night_log.txt 2>&1 &
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=1
date '+開始: %T'
rm -f results/vm_eval_dagger3_oracle_part*.json results/vm2_eval_dagger1_part*.json
# (A) 診断
( for k in 0 1 2 3 4 5 6 7; do python scripts/vm_eval.py --policy checkpoints/vm_dagger3.pt --episodes 200 --shard $k/8 --oracle-planner --out results/vm_eval_dagger3_oracle_part$k.json > results/vm_eval_dagger3_oracle_part$k.txt 2>&1 & done; wait
  python scripts/vm_eval_summary.py vm_eval_dagger3_oracle vm_eval_dagger3_oracle "診断: v1（DAgger 2 回目）+ 順番をオラクルで進める"; date '+診断の完了: %T' ) &
# (B) v2 の DAgger 再学習 → 評価
( OMP_NUM_THREADS=4 python scripts/vm2_train.py --data "data/vm2/expert_*" "data/vm2/dagger1_*" --init checkpoints/vm2_bc.pt --epochs 10 --lr 2e-4 --freeze-layer1 --out checkpoints/vm2_dagger1.pt || { echo "失敗: v2 の再学習"; exit 1; }
  date '+v2 の学習完了: %T'
  for k in 0 1 2 3 4 5 6 7; do python scripts/vm_eval.py --policy checkpoints/vm2_dagger1.pt --episodes 200 --shard $k/8 --out results/vm2_eval_dagger1_part$k.json > results/vm2_eval_dagger1_part$k.txt 2>&1 & done; wait
  python scripts/vm_eval_summary.py vm2_eval_dagger1 vm2_eval_dagger "視覚運動方策 v2（DAgger 後）"; date '+v2 の評価完了: %T' ) &
wait
date '+すべて完了: %T'
