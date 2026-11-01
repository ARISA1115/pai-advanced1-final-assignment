"""デモから、方策（MLP）を学習する。"""
import argparse
import json
import os
import time

import numpy as np
import torch

import _path  # noqa: F401
from _path import ROOT
from sorting.policy import MLPPolicy, to_target

torch.set_num_threads(4)


def load_demos(paths):
    obs, act, ep, off = [], [], [], 0
    for p in paths:
        z = np.load(p)
        obs.append(z["obs"]); act.append(z["act"]); ep.append(z["ep"] + off)
        off = int(ep[-1].max()) + 1
    return np.concatenate(obs), np.concatenate(act), np.concatenate(ep)


def fit(obs, act, ep, epochs=100, seed=0, verbose=False):
    """最後の 10% のエピソードを検証に使い、学習した方策と損失の履歴を返す。"""
    torch.manual_seed(seed); np.random.seed(seed)
    y = to_target(act)
    val = ep >= int(ep.max() * 0.9) if ep.max() >= 9 else np.zeros(len(ep), bool)
    tr = ~val
    if verbose:
        print(f"samples train={tr.sum()} val={val.sum()}")
    model = MLPPolicy()
    model.mean.copy_(torch.as_tensor(obs[tr].mean(0)))
    model.std.copy_(torch.as_tensor(obs[tr].std(0) + 1e-3))
    Xt, Yt = torch.as_tensor(obs[tr]), torch.as_tensor(y[tr])
    Xv, Yv = torch.as_tensor(obs[val]), torch.as_tensor(y[val])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist, t0 = [], time.time()
    for e in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            loss = torch.nn.functional.mse_loss(model(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = torch.nn.functional.mse_loss(model(Xv), Yv).item() if len(Xv) else float("nan")
        hist.append({"epoch": e + 1, "train": tot / len(Xt), "val": vl})
        if verbose and ((e + 1) % 10 == 0 or e == 0):
            print(f"epoch {e+1:3d} train {tot/len(Xt):.5f} val {vl:.5f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[str(ROOT / "data" / "demos_random.npz")])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "policy.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    obs, act, ep = load_demos(args.data)
    model, hist = fit(obs, act, ep, args.epochs, args.seed, verbose=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.out)
    os.makedirs(ROOT / "results", exist_ok=True)
    with open(ROOT / "results" / "train_history.json", "w") as f:
        json.dump(hist, f)
    print("saved", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
