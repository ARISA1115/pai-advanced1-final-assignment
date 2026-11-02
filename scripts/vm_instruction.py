"""日本語の指示 → 画像だけで動く方策（視覚運動方策）。

 指示 → 目標（LLM の解釈。保存済みの出力、results/instruction_eval.json を使う）→ 検査 → 画像・ロボット自身の状態・タスクで動かす。
 キューブの位置を推定する部品も、状態ベクトルも、動作には使わない。
"""
import argparse
import json

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from sorting.env import SortEnv
from sorting.instruction import InstructionError, describe, to_task
from sorting.vision import FrontLocalizer, make_progress_checker
from sorting.visuomotor import FrontCam, Planner, Unstick, load_visuo, proprio_vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--parsed", default=str(ROOT / "results" / "instruction_eval.json"))
    ap.add_argument("--seeds-per", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    k, n = map(int, a.shard.split("/"))
    rows = json.load(open(a.parsed))["rows"][k::n]
    pol = load_visuo(a.policy)
    env = SortEnv(0, task="random"); env.fixed_order = True
    cam = FrontCam(env, pol.res, pol.cams); loc = FrontLocalizer(env)
    planner = Planner(env, strict=True, progress_checker=make_progress_checker(env, loc))   # 評価（vm_eval.py の最終設定）と同じ
    out = []
    for r in rows:
        if r["category"] == "invalid":                      # 実行できない指示: 検査で拒否できたら、動かさない
            out.append({"id": r["id"], "category": r["category"], "text": r["text"], "executed": 0, "success": 1.0 if r.get("safe") else 0.0}); continue
        if not r["correct"]:                                # 解釈の誤り（動かしても、頼まれたことにならない）
            out.append({"id": r["id"], "category": r["category"], "text": r["text"], "executed": 0, "success": 0.0}); continue
        goal, active = to_task(r["output"])
        oks = []
        for s in range(a.seeds_per):
            env.reset(70000 + r["id"] * 10 + s, goal=goal, active=active); planner.reset(); un = Unstick(env, seed=70000 + r["id"] * 10 + s)
            for _ in range(a.max_steps):
                tgt = planner.update()
                _, ok = env.step(un.filter(pol.act(cam.frame(), proprio_vec(env, tgt))))
                if ok:
                    break
            oks.append(env.success())
        out.append({"id": r["id"], "category": r["category"], "text": r["text"], "executed": a.seeds_per, "success": float(np.mean(oks))})
        print(f"  {r['id']:2d} {r['text'][:30]} → {np.mean(oks):.2f}", flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
