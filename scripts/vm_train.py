"""視覚運動方策を学習する（模倣学習。画像には、ランダムな平行移動の増強をかける）。"""
import argparse
import glob
import json
import time

import numpy as np
import torch

import _path  # noqa: F401
from _path import ROOT
from sorting.env import MAX_DELTA
from sorting.visuomotor import make_visuo

torch.set_num_threads(4)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--init", default=None)
    ap.add_argument("--arch", default="base", choices=["base", "film"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = []
    for pat in a.data:
        paths += sorted(glob.glob(pat))
    # メモリを、一時的に 2 倍にしないため、先に大きさを調べ、配列を一度だけ確保して、シャードを 1 つずつ書き込む
    shapes = []
    for p in paths:
        z = np.load(p); shapes.append((len(z["y"]), z["img"].shape[1:], z["vec"].shape[1:], z["y"].shape[1:])); del z
    n_all = sum(sh[0] for sh in shapes)
    I = np.empty((n_all,) + shapes[0][1], np.uint8); V = np.empty((n_all,) + shapes[0][2], np.float32); Y = np.empty((n_all,) + shapes[0][3], np.float32)
    o = 0
    for p, sh in zip(paths, shapes):
        z = np.load(p); I[o:o + sh[0]] = z["img"]; V[o:o + sh[0]] = z["vec"]; Y[o:o + sh[0]] = z["y"]; o += sh[0]; del z
    Y = Y.copy(); Y[:, :3] /= MAX_DELTA
    n = len(I); nv = max(1000, n // 20)
    perm = np.random.default_rng(0).permutation(n); va, tr = perm[:nv], perm[nv:]
    print(f"files {len(paths)} samples {n} train {len(tr)} val {len(va)}", flush=True)
    It, Vt, Yt = torch.as_tensor(I), torch.as_tensor(V), torch.as_tensor(Y)    # データは CPU に置き、バッチごとに GPU へ送る（約 40 万サンプルでも、メモリが足りるように）
    model = make_visuo(a.arch)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True)["state_dict"])
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(tr) + 127) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    hist, t0 = [], time.time()
    for e in range(a.epochs):
        model.train(); p = torch.as_tensor(np.random.permutation(tr)); tot = 0.0
        for i in range(0, len(p), 128):
            idx = p[i:i + 128]
            img = It[idx].to(DEV)
            dx, dy = np.random.randint(-4, 5, 2)                 # 画像のランダムな平行移動（増強）
            img = torch.roll(img, shifts=(int(dy), int(dx)), dims=(2, 3))
            loss = torch.nn.functional.mse_loss(model(img, Vt[idx].to(DEV)), Yt[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([torch.nn.functional.mse_loss(model(It[va[i:i+256]].to(DEV), Vt[va[i:i+256]].to(DEV)), Yt[va[i:i+256]].to(DEV)).item() for i in range(0, len(va), 256)]))
        hist.append({"epoch": e + 1, "train": tot / len(p), "val": vl})
        print(f"epoch {e+1} train {tot/len(p):.4f} val {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "arch": a.arch}, a.out)
    json.dump(hist, open(a.out.replace(".pt", "_history.json"), "w"))
    print("saved", a.out.split("/")[-1])


if __name__ == "__main__":
    main()
