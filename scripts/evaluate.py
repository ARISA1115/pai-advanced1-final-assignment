"""成功率の評価。エキスパート / 学習した方策 / 何もしない方策 を、同じ乱数の種で比べる。"""
import argparse
import json
import time

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from sorting.env import SortEnv
from sorting.expert import expert_action
from sorting.policy import load_policy

MAX_STEPS = 800


def run(mode, seeds, policy=None, task="random"):
    env = SortEnv(0, task=task)
    full, placed, active = 0, 0, 0
    steps, failed = [], []
    for s in seeds:
        env.reset(s)
        ok = False
        for k in range(MAX_STEPS):
            if mode == "expert":
                a = expert_action(env)
            elif mode == "bc":
                a = policy.act(env.obs())
            else:
                a = np.array([0.0, 0.0, 0.0, -1.0])
            _, ok = env.step(a)
            if ok:
                break
        full += int(ok); placed += env.n_placed(); active += env.n_active()
        if ok:
            steps.append(k + 1)
        else:
            failed.append({"seed": int(s), "placed": int(env.n_placed())})
    n = len(seeds)
    return {"mode": mode, "episodes": n, "full_success_rate": full / n,
            "cube_success_rate": placed / active, "mean_steps_success": float(np.mean(steps)) if steps else None,
            "failed": failed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--policy", default=str(ROOT / "checkpoints" / "policy.pt"))
    ap.add_argument("--modes", nargs="+", default=["expert", "bc", "null"])
    ap.add_argument("--task", default="random", choices=["random", "color"])
    ap.add_argument("--out", default=str(ROOT / "results" / "eval.json"))
    args = ap.parse_args()
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    policy = load_policy(args.policy) if "bc" in args.modes else None
    res = []
    for m in args.modes:
        t = time.time()
        r = run(m, seeds, policy, args.task)
        print(r, f"({time.time()-t:.0f}s)", flush=True)
        res.append(r)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
