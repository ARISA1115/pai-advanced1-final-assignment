"""画像とキューブの真の位置の組を集める。エキスパート（ノイズ付き）と、学習した方策の実行の両方から。"""
import argparse
import time

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from sorting.env import SortEnv
from sorting.expert import expert_action
from sorting.policy import load_policy
from sorting.vision import Cameras, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--policy", default=str(ROOT / "checkpoints" / "policy_random.pt"))
    ap.add_argument("--out", default=str(ROOT / "data" / "vision_train.npz"))
    args = ap.parse_args()
    env = SortEnv(0, task="random")
    cams = Cameras(env)
    pol = load_policy(args.policy)
    X, Y = [], []
    t0 = time.time()
    for j in range(args.episodes):
        seed = args.seed0 + j
        rng = np.random.default_rng(seed)
        env.reset(seed)
        use_policy = j % 2 == 1
        sigma = [0.0, 0.006, 0.012][j % 3]
        for k in range(600):
            if k % args.every == 0:
                X.append(cams.frame()); Y.append(truth(env).reshape(-1))
            if use_policy:
                a = pol.act(env.obs())
            else:
                a = expert_action(env); a[:3] += rng.normal(0, sigma, 3)
            _, ok = env.step(a)
            if ok:
                break
        if (j + 1) % 50 == 0:
            print(f"{j+1}/{args.episodes} episodes, {len(X)} frames, {time.time()-t0:.0f}s", flush=True)
    np.savez(args.out, x=np.array(X, np.uint8), y=np.array(Y, np.float32))
    print("saved", len(X), "frames")


if __name__ == "__main__":
    main()
