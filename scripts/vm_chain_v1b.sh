#!/bin/bash
# v1（従来型 CNN）の DAgger の 2 回目: 収集の完了待ち → 全データで再学習 → 評価。収集が落ちたら、その場で失敗として終わる。
cd "$(dirname "$0")/.."
source .venv/bin/activate
while [ "$(ls data/vm/dagger3_*.npz 2>/dev/null | wc -l)" -lt 8 ]; do
  if [ "$(pgrep -f 'vm_collect.py.*dagger3' | wc -l)" -eq 0 ]; then echo "失敗: 収集のプロセスが無いのに、シャードが 8 個そろっていません"; exit 1; fi
  sleep 20
done
date '+収集完了: %T'
while pgrep -f "vm2_chain.sh|vm2_train.py" >/dev/null; do sleep 30; done   # v2 の学習・評価が終わるまで待つ（メモリと GPU を、取り合わないため）
date '+v2 の完了を確認: %T'
python scripts/vm_train.py --data "data/vm/expert_*.npz" "data/vm/expert2_*.npz" "data/vm/dagger2_*.npz" "data/vm/dagger3_*.npz" --init checkpoints/vm_dagger2.pt --epochs ${EPOCHS:-15} --arch base --lr 3e-4 --out checkpoints/vm_dagger3.pt || { echo "失敗: 学習"; exit 1; }
date '+学習完了: %T'
for k in 0 1 2 3 4 5 6 7; do python scripts/vm_eval.py --policy checkpoints/vm_dagger3.pt --episodes 200 --shard $k/8 --out results/vm_eval_dagger3_part$k.json > results/vm_eval_dagger3_part$k.txt 2>&1 & done
wait
python - <<'PY'
import json, glob
from collections import Counter
P=[json.load(open(f)) for f in sorted(glob.glob("results/vm_eval_dagger3_part*.json"))]
n=sum(p["episodes"] for p in P); ok=sum(p["success"]*p["episodes"] for p in P); pl=sum(p["placed"] for p in P); ac=sum(p["active"] for p in P)
F=sum((p["failed"] for p in P),[])
print(f"視覚運動方策 v1（DAgger 2 回目）: {n} エピソード、3 個とも成功 {ok/n:.3f}（{int(ok)}/{n}）、個数ベース {pl/ac:.3f}")
print("失敗の内訳（動かす数, 置けた数）:", sorted(Counter((f["active"],f["placed"]) for f in F).items()))
json.dump({"name":"dagger3","episodes":n,"success":ok/n,"cube_success":pl/ac,"failed":F},open("results/vm_eval_dagger3.json","w"),ensure_ascii=False,indent=1)
PY
date '+評価完了: %T'
