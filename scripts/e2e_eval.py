"""自然言語の指示から、ロボットの動作まで通した評価。

指示の解釈（LLM、results/instruction_eval.json）と、方策の実行（シミュレータ）を、つなげて数える。
 - 実行できる指示: 解釈が正解で、かつ、方策が全ての指定を達成したら成功
 - 実行できない指示: 解釈が「実行不可」と判断し、ロボットを動かさなければ成功
"""
import argparse
import json
from pathlib import Path

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from sorting.env import SortEnv
from sorting.instruction import InstructionError, to_task
from sorting.policy import load_policy

MAX_STEPS = 800


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default=str(ROOT / "results" / "instruction_eval.json"))
    ap.add_argument("--policy", default=str(ROOT / "checkpoints" / "policy_random.pt"))
    ap.add_argument("--seeds-per", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "e2e_eval.json"))
    args = ap.parse_args()
    rows = json.load(open(args.parsed))["rows"]
    pol = load_policy(args.policy)
    env = SortEnv(0, task="random")
    out, by = [], {}
    for r in rows:
        truth_invalid = r["category"] == "invalid"
        if truth_invalid:
            ok_rate = 1.0 if r["safe"] else 0.0       # ロボットが動かなかったか（LLM が拒否、または検査で拒否）
            executed = 0
        elif not r["correct"]:
            ok_rate, executed = 0.0, 0                # 解釈の誤り（実行しても、頼まれたことにならない）
        else:
            goal, active = to_task(r["output"])
            oks = []
            for s in range(args.seeds_per):
                env.reset(60000 + r["id"] * 10 + s, goal=goal, active=active)
                ok = False
                for _ in range(MAX_STEPS):
                    _, ok = env.step(pol.act(env.obs()))
                    if ok:
                        break
                oks.append(ok)
            ok_rate, executed = float(np.mean(oks)), args.seeds_per
        out.append({"id": r["id"], "category": r["category"], "text": r["text"], "parse_correct": r["correct"],
                    "e2e_success": ok_rate, "executed_episodes": executed})
        c = by.setdefault(r["category"], []); c.append(ok_rate)
    summary = {"policy": str(Path(args.policy).resolve().relative_to(ROOT)), "seeds_per_instruction": args.seeds_per,
               "e2e_success_overall": float(np.mean([o["e2e_success"] for o in out])),
               "parse_accuracy": float(np.mean([o["parse_correct"] for o in out])),
               "by_category": {k: float(np.mean(v)) for k, v in by.items()}, "rows": out}
    json.dump(summary, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("指示の解釈の正解率", round(summary["parse_accuracy"], 3), " 指示から動作までの成功率", round(summary["e2e_success_overall"], 3))
    print(summary["by_category"])


if __name__ == "__main__":
    main()
