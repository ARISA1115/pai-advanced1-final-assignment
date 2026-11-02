#!/bin/bash
# 拡散モデルの改良版: 握る・開くを、拡散でなく分類で出す。従来の拡散モデル（epoch 10）から引き継いで学習 → 評価
cd "$(dirname "$0")/.." && . .venv/bin/activate
echo "学習開始: $(date +%T)"
python scripts/vm3_train.py --data "data/vm2/expert_*" "data/vm2/dagger1_*" --init checkpoints/vm3_dp_e10.pt --grip-head --epochs ${EPOCHS:-8} --lr 2e-4 --views 2 --cams front,top --res 80 --out checkpoints/vm3_dpg.pt || { echo "失敗: 学習"; exit 1; }
echo "学習完了: $(date +%T)"
while pgrep -f "vm_eval.py" > /dev/null; do sleep 20; done          # CPU を、ほかの評価と分け合わない
for k in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=1 python scripts/vm_eval.py --policy checkpoints/vm3_dpg.pt --episodes 200 --seed0 800000 --shard $k/8 --strict-planner --visual-progress --max-steps 800 --out results/vm3g_eval_fresh_part$k.json > results/vm3g_eval_fresh_part$k.txt 2>&1 & done
wait
python scripts/vm_eval_summary.py vm3g_eval_fresh vm3g_eval_fresh "拡散モデル（握る・開くは分類）+ 画像で順番を進める（種 800000〜）"
echo "全部完了: $(date +%T)"
