#!/bin/bash
# Diffusion Policy: DAgger4 の学習が終わるのを待って（メモリと GPU を分けるため）学習 → 評価
cd "$(dirname "$0")/.." && . .venv/bin/activate
while pgrep -f "vm_train.py" > /dev/null; do sleep 20; done
echo "学習開始: $(date +%T)"
python scripts/vm3_train.py --data "data/vm2/expert_*" "data/vm2/dagger1_*" --epochs ${EPOCHS:-20} --lr 2e-4 --views 2 --cams front,top --res 80 --out checkpoints/vm3_dp.pt || { echo "失敗: 学習"; exit 1; }
echo "学習完了: $(date +%T)"
for k in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=1 python scripts/vm_eval.py --policy checkpoints/vm3_dp.pt --episodes 200 --seed0 800000 --shard $k/8 --strict-planner --visual-progress --max-steps 800 --out results/vm3_eval_fresh_part$k.json > results/vm3_eval_fresh_part$k.txt 2>&1 & done
wait
python scripts/vm_eval_summary.py vm3_eval_fresh vm3_eval_fresh "Diffusion Policy + 画像で順番を進める(新しい種 800000〜)"
echo "全部完了: $(date +%T)"
