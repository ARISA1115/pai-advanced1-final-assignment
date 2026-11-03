#!/bin/bash
# 未使用の種 830000〜 の 500 エピソードで、新旧の追跡を比べる
cd "$(dirname "$0")/.." && . .venv/bin/activate
echo "開始 $(date +%T)"
for k in 0 1 2 3; do
  OMP_NUM_THREADS=1 python scripts/vision_compare.py --area-frac 0.6 --shard $k/4 --out results/vision_cmp_old_part$k.json > results/vision_cmp_old_part$k.txt 2>&1 &
  OMP_NUM_THREADS=1 python scripts/vision_compare.py --area-frac 0.85 --shard $k/4 --out results/vision_cmp_new_part$k.json > results/vision_cmp_new_part$k.txt 2>&1 &
done
wait
python scripts/vm_eval_summary.py vision_cmp_old vision_cmp_old "追跡: 従来（面積 0.6、種 830000〜）"
python scripts/vm_eval_summary.py vision_cmp_new vision_cmp_new "追跡: 改良（面積 0.85、種 830000〜）"
echo "全部完了 $(date +%T)"
