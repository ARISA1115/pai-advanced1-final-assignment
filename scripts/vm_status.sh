#!/bin/bash
# 進捗の要約（約 10 分ごとの定時報告の元になる）。実行中の作業と、直近の結果を、1 画面にまとめる。
cd "$(dirname "$0")/.."
date '+%F %T'
echo "[実行中のプロセス] 評価 $(pgrep -f vm_eval.py | wc -l) ／ 学習 $(pgrep -f 'vm_train.py|vm2_train.py' | wc -l) ／ 収集 $(pgrep -f vm_collect.py | wc -l)"
echo "[v2 の DAgger 再学習] $(grep -E '^epoch' results/vm_night_log.txt 2>/dev/null | tail -1 | cut -c1-80)"
for f in vm_eval_d3_strict_a vm2_eval_dagger vm_eval_d3_strict_b; do
  if [ -s results/$f.json ]; then .venv/bin/python - "$f" <<'PY'
import json, sys
d = json.load(open(f"results/{sys.argv[1]}.json")); print(f"[結果] {sys.argv[1]}: 3 個とも成功 {d['success']:.3f}（{round(d['success']*d['episodes'])}/{d['episodes']}）、個数ベース {d['cube_success']:.3f}")
PY
  fi
done
echo "[提出済みの main] $(git ls-remote origin main | cut -c1-8)（22729d6b なら、変更なし）"
