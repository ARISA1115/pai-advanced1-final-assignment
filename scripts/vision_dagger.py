"""画像を入れた DAgger。

方策を、画像から推定した位置（誤差あり）で観測を作って動かし、
その観測に対して、シミュレータの真値を見るエキスパートが正解の行動を付ける。
推定誤差のある観測の分布に、方策を慣らす。

  収集: python scripts/vision_dagger.py collect --shard k/n --policy <方策> --seed0 70000 --out data/vdagger_k.npz
  学習: python scripts/vision_dagger.py train --data data/demos_random.npz data/dagger_random_1.npz ... data/vdagger_*.npz --out checkpoints/policy_vision.pt
"""
import argparse
import glob
import time

import numpy as np
import torch

import _path  # noqa: F401
from _path import ROOT
from sorting.env import SortEnv
from sorting.expert import expert_action
from sorting.policy import load_policy
from sorting.vision import Cameras, load_cubenet
from train import fit, load_demos

MAX_STEPS = 600


def collect(args):
    k, n = map(int, args.shard.split("/"))
    env = SortEnv(0, task="random"); cams = Cameras(env)
    policy, net = load_policy(args.policy), load_cubenet(args.cubenet)
    seeds = list(range(args.seed0, args.seed0 + args.episodes))[k::n]
    X, Y, EP, ok_n, t0 = [], [], [], 0, time.time()
    for j, s in enumerate(seeds):
        env.reset(s)
        for _ in range(MAX_STEPS):
            est = net.estimate(cams.frame())
            obs = env.obs(est)
            X.append(obs); Y.append(expert_action(env)); EP.append(j)      # 入力は推定値の観測、正解は真値のエキスパート
            _, ok = env.step(policy.act(obs))
            if ok:
                break
        ok_n += int(env.success())
        if (j + 1) % 10 == 0:
            print(f"[{k}/{n}] {j+1}/{len(seeds)} 成功率 {ok_n/(j+1):.2f} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(args.out, obs=np.array(X, np.float32), act=np.array(Y, np.float32), ep=np.array(EP, np.int32))
    print(f"[{k}/{n}] saved {len(X)} samples, rollout success {ok_n/len(seeds):.3f}")


def train(args):
    paths = []
    for pat in args.data:
        paths += sorted(glob.glob(pat))
    obs, act, ep = load_demos(paths)
    print("data files", len(paths), "samples", len(obs))
    model, hist = fit(obs, act, ep, epochs=args.epochs, seed=0, verbose=True)
    torch.save({"state_dict": model.state_dict()}, args.out)
    print("saved", args.out.split("/")[-1])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--shard", default="0/1"); c.add_argument("--episodes", type=int, default=300)
    c.add_argument("--seed0", type=int, default=70000)
    c.add_argument("--policy", default=str(ROOT / "checkpoints" / "policy_random.pt"))
    c.add_argument("--cubenet", default=str(ROOT / "checkpoints" / "cubenet.pt"))
    c.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", required=True); t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--out", default=str(ROOT / "checkpoints" / "policy_vision.pt"))
    args = ap.parse_args()
    collect(args) if args.cmd == "collect" else train(args)


if __name__ == "__main__":
    main()
