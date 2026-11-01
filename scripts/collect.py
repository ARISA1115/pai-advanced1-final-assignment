"""エキスパートのデモを集める。実行時の行動にはノイズを加えるが、記録するのはノイズなしの正解の行動（DART）。"""
import argparse
import time

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from sorting.env import SortEnv, MAX_DELTA
from sorting.expert import expert_action

MAX_STEPS = 800
NOISE_LEVELS = [0.0, 0.005, 0.010]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--noise", nargs="+", type=float, default=NOISE_LEVELS)
    ap.add_argument("--task", default="random", choices=["random", "color"])
    ap.add_argument("--out", default=str(ROOT / "data" / "demos_random.npz"))
    args = ap.parse_args()

    env = SortEnv(args.seed0, task=args.task)
    X, Y, EP = [], [], []
    kept, tried, t0 = 0, 0, time.time()
    seed = args.seed0
    while kept < args.episodes:
        rng = np.random.default_rng(seed + 777)
        sigma = args.noise[seed % len(args.noise)]
        env.reset(seed)
        obs, acts, ok = [], [], False
        for _ in range(MAX_STEPS):
            a = expert_action(env)
            obs.append(env.obs()); acts.append(a.copy())
            a_exec = a.copy()
            a_exec[:3] += rng.normal(0, sigma, 3)
            _, ok = env.step(a_exec)
            if ok:
                break
        tried += 1
        if ok:
            X += obs; Y += acts; EP += [kept] * len(obs)
            kept += 1
        seed += 1
        if kept % 50 == 0 and ok:
            print(f"{kept}/{args.episodes} episodes, {len(X)} samples, {time.time()-t0:.0f}s", flush=True)
    out = args.out
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, obs=np.array(X, np.float32), act=np.array(Y, np.float32), ep=np.array(EP, np.int32))
    print(f"saved {len(X)} samples from {kept} episodes (tried {tried}); demo success rate {kept/tried:.3f}")


if __name__ == "__main__":
    main()
