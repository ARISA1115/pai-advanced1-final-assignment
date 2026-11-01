"""キューブの位置を推定する CNN を学習する。検証は、別のエピソードから集めた画像で行う。"""
import argparse
from pathlib import Path
import json
import time

import numpy as np
import torch

import _path  # noqa: F401
from _path import ROOT
from sorting.vision import CubeNet, CubeNetSS

torch.set_num_threads(6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data" / "vision_train.npz"))
    ap.add_argument("--val", default=str(ROOT / "data" / "vision_val.npz"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "cubenet.pt"))
    ap.add_argument("--arch", default="cnn", choices=["cnn", "ss"])
    ap.add_argument("--subsample", type=int, default=1, help="学習データを、この間隔で間引く（時間の節約）")
    ap.add_argument("--summary", default=str(ROOT / "results" / "cubenet_train.json"))
    args = ap.parse_args()
    torch.manual_seed(0)
    tr, va = np.load(args.train), np.load(args.val)
    Xt, Yt, Xv, Yv = tr["x"][::args.subsample], tr["y"][::args.subsample], va["x"], va["y"]
    mean, std = Yt.mean(0), Yt.std(0) + 1e-6
    model = CubeNetSS() if args.arch == "ss" else CubeNet()
    model.mean.copy_(torch.as_tensor(mean)); model.std.copy_(torch.as_tensor(std))
    Xt_t, Yt_t = torch.as_tensor(Xt), torch.as_tensor((Yt - mean) / std)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.epochs * ((len(Xt) + 127) // 128))
    hist, t0 = [], time.time()

    def val_err():
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(torch.as_tensor(Xv[i:i + 256])) for i in range(0, len(Xv), 256)]) * model.std + model.mean
        err = (pred.numpy() - Yv).reshape(-1, 3, 3)
        return np.abs(err)

    for e in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt_t))
        tot = 0.0
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            x = Xt_t[idx]
            loss = torch.nn.functional.smooth_l1_loss(model(x), Yt_t[idx], beta=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        ae = val_err()
        hist.append({"epoch": e + 1, "train_loss": tot / len(perm), "val_mae_mm": (ae.mean((0, 1)) * 1000).tolist()})
        print(f"epoch {e+1:2d} loss {tot/len(perm):.4f} 検証の平均絶対誤差 [mm] x,y,z = {np.round(ae.mean((0,1))*1000,1)} ({time.time()-t0:.0f}s)", flush=True)
    ae = val_err()
    summary = {"val_mae_mm": (ae.mean((0, 1)) * 1000).tolist(), "val_median_mm": (np.median(ae, axis=(0, 1)) * 1000).tolist(),
               "val_p95_mm": (np.percentile(ae.reshape(-1, 3), 95, axis=0) * 1000).tolist(),
               "train_frames": int(len(Xt)), "val_frames": int(len(Xv)), "history": hist}
    torch.save({"state_dict": model.state_dict(), "arch": args.arch}, args.out)
    json.dump(summary, open(args.summary, "w"), indent=1)
    print("saved", Path(args.out).name, "median [mm]", np.round(summary["val_median_mm"], 1), "p95 [mm]", np.round(summary["val_p95_mm"], 1))


if __name__ == "__main__":
    main()
