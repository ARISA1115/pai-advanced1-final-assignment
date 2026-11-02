#!/bin/bash
# 最終: dagger5 + 順番を画像で進める + 止まったら動かす。種 810000〜（設定の選択に使っていない）。日本語の指示の評価と、動画の生成も行う
cd "$(dirname "$0")/.." && . .venv/bin/activate
P=checkpoints/vm_dagger5.pt
echo "開始 $(date +%T)"
for k in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=1 python scripts/vm_eval.py --policy $P --episodes 200 --seed0 810000 --shard $k/8 --strict-planner --visual-progress --unstick --max-steps 800 --out results/vm_final_part$k.json > results/vm_final_part$k.txt 2>&1 & done
for k in 0 1 2; do OMP_NUM_THREADS=1 python scripts/vm_instruction.py --policy $P --shard $k/3 --out results/vm_instruction_d5_part$k.json > results/vm_instruction_d5_part$k.txt 2>&1 & done
wait
python scripts/vm_eval_summary.py vm_final vm_final "最終: dagger5 + 順番を画像で進める + 止まったら動かす（種 810000〜）"
python - <<'P'
import json, glob
R = []
for f in sorted(glob.glob("results/vm_instruction_d5_part*.json")): R += json.load(open(f))
json.dump(sorted(R, key=lambda r: r["id"]), open("results/vm_instruction_eval.json", "w"), ensure_ascii=False, indent=1)
print("指示の評価:", len(R), "件")
P
echo "評価の完了 $(date +%T)"
python scripts/vm_video.py --policy $P --base 提出物/PAI最終課題_144133.mp4 --instr results/vm_instruction_eval.json --out 提出物/PAI最終課題_144133_2回目.mp4
echo "全部完了 $(date +%T)"
