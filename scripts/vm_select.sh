#!/bin/bash
# v1 の成績を上げる設定の選択（種 800000〜、200 エピソード）。最終の数字は、別の種（810000〜）で測る。
cd "$(dirname "$0")/.." && . .venv/bin/activate
ENS="checkpoints/vm_dagger2.pt+checkpoints/vm_dagger3.pt+checkpoints/vm_dagger4.pt"
run() {  # $1 名前 $2 方策 $3.. 追加の引数
  name=$1; pol=$2; shift 2
  for k in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=1 python scripts/vm_eval.py --policy "$pol" --episodes 200 --seed0 800000 --shard $k/8 --strict-planner --visual-progress --max-steps 800 "$@" --out results/${name}_part$k.json > results/${name}_part$k.txt 2>&1 & done
  wait; python scripts/vm_eval_summary.py $name $name "$name"
}
echo "開始 $(date +%T)"
run vm_sel_ens "$ENS"
run vm_sel_d3_check checkpoints/vm_dagger3.pt --visual-check
run vm_sel_ens_check "$ENS" --visual-check
echo "全部完了 $(date +%T)"
