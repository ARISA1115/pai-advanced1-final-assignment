"""Diffusion Policy の学習（事前学習した ResNet18 + 拡散モデルで、動作の系列を生成する）。

データは、vm_collect.py の --npy 出力（_img / _vec / _y / _ep / _src）。
 - 系列の正解: エキスパートのデモ（src = 0）は、同じエピソードの、続く CHUNK ステップ分の動作。
   学習した方策で動かしたデータ（DAgger、src = 1）は、系列の正解が取れないので、1 ステップ目だけを使う。
 - 検証: エピソード単位で、5% を取り分ける。
"""
import argparse
import glob
import json
import time

import numpy as np
import torch

import _path  # noqa: F401
from _path import ROOT
from sorting.env import MAX_DELTA
from sorting.visuomotor import CHUNK, VisuoDiffusion

torch.set_num_threads(4)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


class CatImages:
    """画像（約 12 GB）を、メモリに全部載せず、ディスクから必要な分だけ読む（連結もしない）。"""

    def __init__(self, arrays):
        self.arrays = arrays; self.off = np.cumsum([0] + [len(a) for a in arrays])

    def __len__(self):
        return int(self.off[-1])

    def __getitem__(self, idx):
        idx = np.asarray(idx); out = np.empty((len(idx),) + self.arrays[0].shape[1:], np.uint8)
        sh = np.searchsorted(self.off, idx, side="right") - 1
        for k in np.unique(sh):
            m = sh == k; out[m] = self.arrays[k][idx[m] - self.off[k]]
        return torch.from_numpy(out)


def load_shards(prefixes):
    I, V, Y, EP, SRC = [], [], [], [], []
    for n, pre in enumerate(prefixes):
        I.append(np.load(pre + "_img.npy", mmap_mode="r")); V.append(np.load(pre + "_vec.npy")); Y.append(np.load(pre + "_y.npy"))
        EP.append(np.load(pre + "_ep.npy").astype(np.int64) + n * 100000); SRC.append(np.load(pre + "_src.npy"))
    return CatImages(I), np.concatenate(V), np.concatenate(Y), np.concatenate(EP), np.concatenate(SRC)


def chunk_labels(Y, EP, SRC, k):
    n = len(Y); Yc = np.zeros((n, k, 4), np.float32); M = np.zeros((n, k), np.float32)
    for j in range(k):
        idx = np.minimum(np.arange(n) + j, n - 1)
        ok = (EP[idx] == EP) & (SRC == 0) & (np.arange(n) + j < n)
        ok[:] = ok | (j == 0)
        Yc[:, j] = np.where(ok[:, None], Y[idx], Y); M[:, j] = ok
    return Yc, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="接頭辞（例: data/vm2/expert_*）。_img.npy を除いたもの")
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--freeze-layer1", action="store_true"); ap.add_argument("--init", default=None)
    ap.add_argument("--pretrained", default=str(ROOT / "data" / "pretrained" / "resnet18-f37072fd.pth"))
    ap.add_argument("--cams", default="front,top"); ap.add_argument("--res", type=int, default=80)
    ap.add_argument("--grip-head", action="store_true", help="握る・開くを、拡散でなく分類で出す")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prefixes = []
    for pat in a.data:
        prefixes += sorted(p[:-len("_img.npy")] for p in glob.glob(pat + "_img.npy"))
    I, V, Y, EP, SRC = load_shards(prefixes)
    Y = Y.copy(); Y[:, :3] /= MAX_DELTA
    Yc, M = chunk_labels(Y, EP, SRC, CHUNK)
    eps = np.unique(EP); rng = np.random.default_rng(0)
    val_eps = set(rng.choice(eps, max(1, len(eps) // 20), replace=False).tolist())
    isval = np.array([e in val_eps for e in EP])
    tr, va = np.nonzero(~isval)[0], np.nonzero(isval)[0]
    print(f"シャード {len(prefixes)}、サンプル {len(I)}（学習 {len(tr)}、検証 {len(va)}）、データの出所: エキスパート {(SRC==0).sum()} / 方策 {(SRC==1).sum()}", flush=True)
    It, Vt = I, torch.as_tensor(V)
    Yt, Mt = torch.as_tensor(Yc), torch.as_tensor(M)
    model = VisuoDiffusion(n_views=a.views, grip_head=a.grip_head)
    if a.init:
        sd = torch.load(a.init, map_location="cpu", weights_only=True)["state_dict"]; own = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print(f"初期値: {len(keep)}/{len(own)} 個のパラメータを引き継いだ（形が違うものは、新しく初期化）", flush=True)
    else:
        model.trunk.load_pretrained(a.pretrained)
    if a.freeze_layer1:
        for m in (model.trunk.conv1, model.trunk.bn1, model.trunk.layer1):
            for p in m.parameters():
                p.requires_grad = False
    model.to(DEV)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(tr) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)

    def batch(idx):
        i = torch.as_tensor(idx)
        return It[np.asarray(idx)].to(DEV), Vt[i].to(DEV), Yt[i].to(DEV), Mt[i].to(DEV)

    def loss_fn(pred, y, m):
        return (((pred - y) ** 2).mean(-1) * m).sum() / m.sum()

    hist, t0 = [], time.time()
    for e in range(a.epochs):
        model.train(); perm = np.random.permutation(tr); tot, cnt = 0.0, 0
        for s in range(0, len(perm), a.batch):
            img, vec, y, m = batch(perm[s:s + a.batch])
            dx, dy = np.random.randint(-3, 4, 2)                      # 画像のランダムな平行移動（増強）
            img = torch.roll(img, shifts=(int(dy), int(dx)), dims=(2, 3))
            loss = model.loss(img, vec, y, m)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(perm[s:s + a.batch]); cnt += len(perm[s:s + a.batch])
        model.eval(); vl, vn = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(va), 256):
                img, vec, y, m = batch(va[s:s + 256]); vl += model.loss(img, vec, y, m).item() * len(va[s:s + 256]); vn += len(va[s:s + 256])
        hist.append({"epoch": e + 1, "train": tot / cnt, "val": vl / vn})
        print(f"epoch {e+1}/{a.epochs} train {tot/cnt:.4f} val {vl/vn:.4f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "arch": ("diffusion_g" if a.grip_head else "diffusion"), "in_ch": 3 * a.views,
                    "cams": a.cams.split(","), "res": a.res}, a.out)       # エポックごとに保存（途中で止まっても、使える）
    json.dump(hist, open(a.out.replace(".pt", "_history.json"), "w"))
    print("saved", a.out.split("/")[-1])


if __name__ == "__main__":
    main()
