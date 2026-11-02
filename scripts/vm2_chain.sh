#!/bin/bash
# 収集の完了待ち → 学習 → 評価。収集が途中で落ちたら、その場で失敗として終わる。
cd "$(dirname "$0")/.."
source .venv/bin/activate
while [ "$(ls data/vm2/expert_*_src.npy 2>/dev/null | wc -l)" -lt 8 ]; do
  if [ "$(pgrep -f vm_collect.py | wc -l)" -eq 0 ]; then echo "失敗: 収集のプロセスが無いのに、シャードが 8 個そろっていません"; exit 1; fi
  sleep 20
done
date '+収集完了: %T'
python scripts/vm2_train.py --data "data/vm2/expert_*" --epochs ${EPOCHS:-12} --freeze-layer1 --out checkpoints/vm2_bc.pt || { echo "失敗: 学習"; exit 1; }
date '+学習完了: %T'
for k in 0 1 2 3 4 5 6 7; do python scripts/vm_eval.py --policy checkpoints/vm2_bc.pt --episodes 200 --shard $k/8 --out results/vm2_eval_bc_part$k.json > results/vm2_eval_bc_part$k.txt 2>&1 & done
wait
python - <<'PY'
import json, glob
from collections import Counter
P=[json.load(open(f)) for f in sorted(glob.glob("results/vm2_eval_bc_part*.json"))]
n=sum(p["episodes"] for p in P); ok=sum(p["success"]*p["episodes"] for p in P); pl=sum(p["placed"] for p in P); ac=sum(p["active"] for p in P)
F=sum((p["failed"] for p in P),[])
print(f"視覚運動方策 v2（ResNet18 + 2 視点 + 系列、BC）: {n} エピソード、3 個とも成功 {ok/n:.3f}（{int(ok)}/{n}）、個数ベース {pl/ac:.3f}")
print("失敗の内訳（動かす数, 置けた数）:", sorted(Counter((f["active"],f["placed"]) for f in F).items()))
json.dump({"name":"v2_bc","episodes":n,"success":ok/n,"cube_success":pl/ac,"failed":F},open("results/vm2_eval_bc.json","w"),ensure_ascii=False,indent=1)
PY
date '+評価完了: %T'
