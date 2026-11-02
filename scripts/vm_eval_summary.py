"""分割して評価した結果（results/<prefix>_part*.json）を、まとめて表示・保存する。"""
import glob
import json
import sys
from collections import Counter

prefix, name, label = sys.argv[1], sys.argv[2], sys.argv[3]
P = [json.load(open(f)) for f in sorted(glob.glob(f"results/{prefix}_part*.json"))]
n = sum(p["episodes"] for p in P); ok = sum(p["success"] * p["episodes"] for p in P)
pl = sum(p["placed"] for p in P); ac = sum(p["active"] for p in P)
F = sum((p["failed"] for p in P), [])
print(f"{label}: {n} エピソード、3 個とも成功 {ok/n:.3f}（{int(round(ok))}/{n}）、個数ベース {pl/ac:.3f}")
print("失敗の内訳（動かす数, 置けた数）:", sorted(Counter((f["active"], f["placed"]) for f in F).items()))
json.dump({"name": name, "episodes": n, "success": ok / n, "cube_success": pl / ac, "failed": F},
          open(f"results/{name}.json", "w"), ensure_ascii=False, indent=1)
