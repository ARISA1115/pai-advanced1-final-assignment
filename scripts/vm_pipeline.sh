#!/bin/bash
# 視覚運動方策（画像を入力にした方策）の、学習 → 評価 → DAgger → 再学習 → 再評価。
# 前提: scripts/vm_collect.py で、エキスパートのデータ data/vm/expert_{0..7}.npz を集め終えていること。
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
N=8
EVAL_EPISODES=${EVAL_EPISODES:-200}

evaluate() {   # $1: 方策 $2: 名前
  for k in $(seq 0 $((N-1))); do
    python scripts/vm_eval.py --policy "$1" --episodes $EVAL_EPISODES --shard $k/$N --out results/vm_eval_$2_part$k.json > results/vm_eval_$2_part$k.txt 2>&1 &
  done
  wait
  python - "$2" <<'E'
import json, sys, glob
name = sys.argv[1]
parts = [json.load(open(f)) for f in sorted(glob.glob(f"results/vm_eval_{name}_part*.json"))]
n = sum(p["episodes"] for p in parts)
ok = sum(p["success"] * p["episodes"] for p in parts)
placed = sum(p["placed"] for p in parts); active = sum(p["active"] for p in parts)
res = {"name": name, "episodes": n, "success": ok / n, "cube_success": placed / active,
       "failed": sum((p["failed"] for p in parts), [])}
json.dump(res, open(f"results/vm_eval_{name}.json", "w"), ensure_ascii=False, indent=1)
print(f"評価 {name}: {n} エピソード、3 個とも成功 {ok/n:.3f}、個数ベース {placed/active:.3f}")
E
}

echo "== 1. BC の学習"
python scripts/vm_train.py --data "data/vm/expert_*.npz" --epochs ${BC_EPOCHS:-6} --out checkpoints/vm_bc.pt
echo "== 2. BC の評価"
evaluate checkpoints/vm_bc.pt bc

echo "== 3. DAgger の収集（学習した方策で動かし、エキスパートが正解を付ける）"
for k in $(seq 0 $((N-1))); do
  python scripts/vm_collect.py --shard $k/$N --episodes 320 --seed0 120000 --policy checkpoints/vm_bc.pt --out data/vm/dagger1_$k.npz > results/vm_dagger1_collect_$k.txt 2>&1 &
done
wait
echo "== 4. 再学習"
python scripts/vm_train.py --data "data/vm/expert_*.npz" "data/vm/dagger1_*.npz" --init checkpoints/vm_bc.pt --epochs ${DG_EPOCHS:-3} --out checkpoints/vm_dagger1.pt
echo "== 5. 再評価"
evaluate checkpoints/vm_dagger1.pt dagger1
echo "== 完了"
