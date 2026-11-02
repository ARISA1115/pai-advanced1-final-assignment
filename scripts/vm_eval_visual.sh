#!/bin/bash
# 画像による「置けたか」の確認と、やり直しの効果の評価。上限 800 ステップで、確認なし（strict のみ）と、確認あり（strict + visual）を、同じ種で比べる。
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=1
POL=${POL:-checkpoints/vm_dagger3.pt}; TAG=${TAG:-d3}; SEED0=${SEED0:-10000}; EPS=${EPS:-200}
for variant in visual novisual; do
  EXTRA="--strict-planner"; [ "$variant" = visual ] && EXTRA="--strict-planner --visual-check"
  for k in 0 1 2 3 4 5 6 7; do python scripts/vm_eval.py --policy $POL --episodes $EPS --seed0 $SEED0 --shard $k/8 --max-steps 800 $EXTRA --out results/vm_eval_${TAG}_${variant}_${SEED0}_part$k.json > results/vm_eval_${TAG}_${variant}_${SEED0}_part$k.txt 2>&1 & done
  wait
  python scripts/vm_eval_summary.py vm_eval_${TAG}_${variant}_${SEED0} vm_eval_${TAG}_${variant}_${SEED0} "${TAG} ${variant}（上限 800 ステップ、種 ${SEED0}〜）"
done
date '+完了: %T'
