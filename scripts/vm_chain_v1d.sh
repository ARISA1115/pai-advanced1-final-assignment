#!/bin/bash
# 最初に運ぶキューブが色 1・2 のエピソードの成功率が低い（色 0: 95%、色 1: 68%、色 2: 33%）ので、そのエピソードを集めて再学習する
cd "$(dirname "$0")/.." && . .venv/bin/activate
echo "開始 $(date +%T)"
for k in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=1 python scripts/vm_collect.py --shard $k/8 --episodes 1400 --seed0 1000000 --first-target 1,2 --policy checkpoints/vm_dagger3.pt --out data/vm/dagger5_$k.npz > results/vm_dagger5_collect_$k.txt 2>&1 & done
wait
echo "方策で動かした収集の完了 $(date +%T)"
for k in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=1 python scripts/vm_collect.py --shard $k/8 --episodes 1400 --seed0 1100000 --first-target 1,2 --out data/vm/expert5_$k.npz > results/vm_expert5_collect_$k.txt 2>&1 & done
wait
echo "エキスパートの収集の完了 $(date +%T)"
[ "$(ls data/vm/dagger5_*.npz data/vm/expert5_*.npz | wc -l)" -eq 16 ] || { echo "失敗: シャードが足りない"; exit 1; }
DP=$(pgrep -f vm3_train.py | head -1)
[ -n "$DP" ] && kill -STOP $DP && echo "拡散モデルの学習を一時停止した（pid $DP）$(date +%T)"
python scripts/vm_train.py --data "data/vm/expert_*.npz" "data/vm/dagger3_*.npz" "data/vm/dagger4_*.npz" "data/vm/dagger5_*.npz" "data/vm/expert5_*.npz" --init checkpoints/vm_dagger3.pt --epochs ${EPOCHS:-10} --arch base --lr 3e-4 --out checkpoints/vm_dagger5.pt
R=$?
[ -n "$DP" ] && kill -CONT $DP && echo "拡散モデルの学習を再開した $(date +%T)"
[ $R -eq 0 ] || { echo "失敗: 学習"; exit 1; }
echo "学習完了 $(date +%T)"
for k in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=1 python scripts/vm_eval.py --policy checkpoints/vm_dagger5.pt --episodes 200 --seed0 800000 --shard $k/8 --strict-planner --visual-progress --unstick --max-steps 800 --out results/vm_sel_d5_unstick_part$k.json > results/vm_sel_d5_unstick_part$k.txt 2>&1 & done
wait
python scripts/vm_eval_summary.py vm_sel_d5_unstick vm_sel_d5_unstick "dagger5 + 止まったら動かす（種 800000〜）"
echo "全部完了 $(date +%T)"
