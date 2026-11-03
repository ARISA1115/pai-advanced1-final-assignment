"""追跡の設定の比較: 一部が隠れた領域で位置を更新するか（area_frac）を変え、同じ種で、画像から推定して動かす成功率を比べる。"""
import argparse
import json

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from sorting.env import SortEnv
from sorting.policy import load_policy
from sorting.vision import FrontLocalizer, Tracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area-frac", type=float, required=True); ap.add_argument("--seed0", type=int, default=830000)
    ap.add_argument("--episodes", type=int, default=500); ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    k, n = map(int, a.shard.split("/"))
    pol = load_policy(str(ROOT / "checkpoints" / "policy_random.pt")); env = SortEnv(0, task="random")
    tr = Tracker(env, FrontLocalizer(env), area_frac=a.area_frac)
    seeds = list(range(a.seed0, a.seed0 + a.episodes))[k::n]
    ok_n, placed, active, failed = 0, 0, 0, []
    for s in seeds:
        env.reset(s); tr.reset(); ok = False
        for _ in range(800):
            _, ok = env.step(pol.act(env.obs(tr.update())))
            if ok:
                break
        ok_n += int(ok); placed += env.n_placed(); active += env.n_active()
        if not ok:
            failed.append({"seed": int(s), "placed": int(env.n_placed()), "active": int(env.n_active())})
        print(f"  seed {s} {'成功' if ok else '失敗'}（{env.n_placed()}/{env.n_active()}）", flush=True)
    json.dump({"episodes": len(seeds), "success": ok_n / len(seeds), "placed": placed, "active": active, "failed": failed}, open(a.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
