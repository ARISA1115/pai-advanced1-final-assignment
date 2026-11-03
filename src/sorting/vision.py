"""カメラ画像から、キューブの位置（x, y, z）を推定する CNN。

真上（top）と横（side）の 2 視点を重ねた 6 チャンネルの画像（96×96）を入力し、
3 個のキューブの位置（9 個の数）を回帰する。アームがキューブを隠す場面でも学習データに含める。
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch
from torch import nn

RES = 96
N_CUBES = 3


class Cameras:
    def __init__(self, env, res: int = RES):
        self.env = env
        self.r = mujoco.Renderer(env.m, res, res)

    def frame(self) -> np.ndarray:
        """(6, res, res) の uint8。"""
        views = []
        for cam in ("top", "side"):
            self.r.update_scene(self.env.d, camera=cam)
            views.append(self.r.render())
        return np.concatenate(views, axis=2).transpose(2, 0, 1).copy()


def truth(env) -> np.ndarray:
    return np.stack([env.cube_pos(i) for i in range(N_CUBES)]).astype(np.float32)


class CubeNet(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o, k, s):
            return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2), nn.BatchNorm2d(o), nn.ReLU())

        self.conv = nn.Sequential(blk(6, 32, 5, 2), blk(32, 64, 3, 2), blk(64, 96, 3, 2), blk(96, 128, 3, 2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 6 * 6, 256), nn.ReLU(), nn.Linear(256, 9))
        self.register_buffer("mean", torch.zeros(9))
        self.register_buffer("std", torch.ones(9))

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: uint8 (B, 6, H, W)
        return self.head(self.conv(x.float() / 255.0))

    @torch.no_grad()
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        y = self.forward(torch.as_tensor(frame).unsqueeze(0))[0] * self.std + self.mean
        return y.numpy().reshape(3, 3).astype(np.float64)


class CubeNetSS(nn.Module):
    """空間ソフトアーグマックス版。特徴マップのヒートマップから、画像上の位置をサブピクセルで取り出す。"""

    def __init__(self, keypoints: int = 24):
        super().__init__()

        def blk(i, o, k, s):
            return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2), nn.BatchNorm2d(o), nn.ReLU())

        self.conv = nn.Sequential(blk(6, 32, 5, 2), blk(32, 64, 3, 1), blk(64, 64, 3, 2), blk(64, 64, 3, 1))
        self.kp = nn.Conv2d(64, keypoints, 1)
        self.head = nn.Sequential(nn.Linear(keypoints * 2, 256), nn.ReLU(), nn.Linear(256, 9))
        self.register_buffer("mean", torch.zeros(9))
        self.register_buffer("std", torch.ones(9))
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.kp(self.conv(x.float() / 255.0))                       # (B, K, 24, 24)
        b, k, hh, ww = h.shape
        w = torch.softmax(h.reshape(b, k, -1) * self.temperature, dim=-1).reshape(b, k, hh, ww)
        ys = torch.linspace(-1, 1, hh, device=x.device); xs = torch.linspace(-1, 1, ww, device=x.device)
        ex = (w.sum(2) * xs).sum(-1); ey = (w.sum(3) * ys).sum(-1)
        return self.head(torch.cat([ex, ey], dim=1))

    @torch.no_grad()
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        y = self.forward(torch.as_tensor(frame).unsqueeze(0))[0] * self.std + self.mean
        return y.numpy().reshape(3, 3).astype(np.float64)


def load_cubenet(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = CubeNetSS() if ck.get("arch") == "ss" else CubeNet()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


# ---------------------------------------------------------------------------
# 幾何に基づく推定器（前方斜め上のカメラ + 色の領域の重心 + 校正）と、追跡
# ---------------------------------------------------------------------------
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from scipy import ndimage  # noqa: E402

CALIB = Path(__file__).resolve().parents[2] / "checkpoints" / "front_calib.json"
# 色の判定（比の条件。影で暗くなっても使える）。RGB は 0〜255
COLOR_RULES = {
    0: lambda r, g, b: (r > 60) & (r > 1.7 * g) & (r > 1.7 * b),
    1: lambda r, g, b: (g > 45) & (g > 1.3 * r) & (g > 1.25 * b),
    2: lambda r, g, b: (b > 70) & (b > 1.5 * r) & (b > 1.1 * g),
}
HELD_DZ = -0.0024      # 持ち上げ中のキューブの中心の、手先に対する高さ [m]（実測の平均）
PAD_XY_ = np.array([[0.62, -0.20], [0.62, 0.00], [0.62, 0.20]])


class FrontLocalizer:
    """前方斜め上のカメラで、3 色のキューブの位置（机の面の x, y）を推定する。

    色ごとに領域を取り、置き場所（既知の位置）の領域を除いた最大のものの重心を、カメラの幾何で机の上へ戻す。
    見え方による偏りは、校正（アフィン補正）で取り除く。
    """

    def __init__(self, env, res: int = 256, cam: str = "front", plane_z: float = 0.03):
        self.env, self.res, self.plane_z = env, res, plane_z
        self.r = mujoco.Renderer(env.m, res, res)
        self.cam = cam
        cid = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        mujoco.mj_forward(env.m, env.d)
        self.cpos = env.d.cam_xpos[cid].copy()
        self.R = env.d.cam_xmat[cid].reshape(3, 3).copy()      # 列: カメラの x（右）、y（上）、z（後ろ）
        self.f = (res / 2) / np.tan(np.deg2rad(env.m.cam_fovy[cid]) / 2)
        self.c = (res - 1) / 2
        self.A = np.eye(2); self.b = np.zeros(2)
        if CALIB.exists():
            d = json.load(open(CALIB)); self.A = np.array(d["A"]); self.b = np.array(d["b"])
        self.pad_px = np.array([self.project(np.array([x, y, 0.0005])) for x, y in PAD_XY_])

    def project(self, X: np.ndarray) -> np.ndarray:
        Xc = self.R.T @ (X - self.cpos)
        return np.array([self.c + self.f * Xc[0] / -Xc[2], self.c - self.f * Xc[1] / -Xc[2]])

    def backproject(self, u: float, v: float, z: float) -> np.ndarray:
        d = self.R @ np.array([(u - self.c) / self.f, -(v - self.c) / self.f, -1.0])
        t = (z - self.cpos[2]) / d[2]
        return self.cpos + t * d

    def blobs(self):
        """色ごとに、置き場所を除いた最大の領域の (重心 u, v, 面積)。見つからなければ None。"""
        self.r.update_scene(self.env.d, camera=self.cam)
        img = self.r.render().astype(np.float32)
        r_, g_, b_ = img[..., 0], img[..., 1], img[..., 2]
        out = []
        for k, rule in COLOR_RULES.items():
            lab, n = ndimage.label(rule(r_, g_, b_))
            best, best_area = None, 0
            for j in range(1, n + 1):
                ys, xs = np.nonzero(lab == j)
                if len(xs) < 5:
                    continue
                cu, cv = xs.mean(), ys.mean()
                if np.linalg.norm(np.array([cu, cv]) - self.pad_px[k]) < 0.035 * self.f:   # 置き場所の領域
                    continue
                if len(xs) > best_area:
                    best, best_area = (cu, cv, len(xs)), len(xs)
            out.append(best)
        return out

    def pad_area(self, k: int) -> int:
        """色 k の、置き場所の領域（キューブが乗っていれば、その分を含む）の面積。見えなければ 0。"""
        self.r.update_scene(self.env.d, camera=self.cam)
        img = self.r.render().astype(np.float32)
        lab, n = ndimage.label(COLOR_RULES[k](img[..., 0], img[..., 1], img[..., 2]))
        best = 0
        for j in range(1, n + 1):
            ys, xs = np.nonzero(lab == j)
            if len(xs) >= 5 and np.linalg.norm(np.array([xs.mean(), ys.mean()]) - self.pad_px[k]) < 0.035 * self.f:
                best = max(best, len(xs))
        return best

    def to_xy(self, u: float, v: float, z: float) -> np.ndarray:
        """画素の位置を、高さ z の面の (x, y) に戻し、校正（アフィン補正）をかける。"""
        return self.A @ self.backproject(u, v, z)[:2] + self.b

    def raw(self) -> np.ndarray:
        """校正前の推定（3, 2）。見つからない色は NaN。"""
        out = np.full((3, 2), np.nan)
        for k, bl in enumerate(self.blobs()):
            if bl is not None:
                out[k] = self.backproject(bl[0], bl[1], self.plane_z)[:2]
        return out

    def detect(self) -> np.ndarray:
        """校正後の推定（3, 2）。見つからない色は NaN。"""
        raw = self.raw()
        return raw @ self.A.T + self.b


class Tracker:
    """画像と、ロボット自身の手先・グリッパーから、キューブの位置を追跡する（キューブの真の位置は使わない）。

    - 最初に、前方カメラの色の重心で、3 個の位置を推定する。
    - 毎ステップ、カメラで再検出する。机の上のキューブは、見えていれば位置を更新する（落としても拾える）。
    - つかんだら、手先との相対位置で追跡し、見えていれば、カメラで相対位置を修正する（指の中での滑り）。
    - 離したら、その場に置かれたとみなす。
    """

    def __init__(self, env, localizer: FrontLocalizer, fallback=None, redetect: bool = True, area_frac: float = 0.85):
        self.env, self.loc, self.fallback, self.redetect = env, localizer, fallback, redetect
        self.area_frac = area_frac      # 領域の面積が、最初の面積のこの割合より小さければ（一部が隠れている）、位置を更新しない
        self.est = np.tile(np.array([0.4, 0.0, 0.02]), (3, 1)); self.holding = None; self.off = None
        self.area0 = np.full(3, np.nan); self.missing = 0
        self.lost = set()       # 落としたキューブ（カメラで探し直す）

    def reset(self):
        bl = self.loc.blobs()
        self.missing = sum(b is None for b in bl)
        for i in range(3):
            if bl[i] is None:
                xy = self.fallback()[i][:2] if self.fallback is not None else np.array([0.4, 0.0])
                self.area0[i] = np.nan
            else:
                xy = self.loc.to_xy(bl[i][0], bl[i][1], self.loc.plane_z); self.area0[i] = bl[i][2]
            self.est[i] = [xy[0], xy[1], 0.02]
        self.holding, self.off = None, None; self.lost = set()
        return self.est.copy()

    def update(self):
        ee = self.env.tcp(); og = self.env.grip_open()
        # --- 持っている / 離した / 落とした の判定（手先とグリッパーだけで判定）
        if self.holding is None:
            t = self.env.target_index(self.est)
            if t is not None and 0.25 < og < 0.65 and np.linalg.norm(ee - self.est[t]) < 0.04:
                self.holding = t; self.off = self.est[t] - ee
        else:
            h = self.holding
            if og > 0.80:                                   # 離した: その場に置かれた
                self.est[h][:2] = (ee + self.off)[:2]; self.est[h][2] = 0.02
                self.holding = None
            elif og < 0.2:                                  # 落とした（何もつかめていない）: 位置はカメラで探す
                self.est[h][2] = 0.02
                self.lost.add(h); self.holding = None
            else:
                p = ee + self.off
                p[2] = max(0.02, ee[2] + HELD_DZ)
                self.est[h] = p
        # --- カメラで再検出（落としたキューブと、手先から遠いキューブだけ。近いと、ハンドに隠れて重心がずれる）
        if self.redetect:
            bl = self.loc.blobs()
            for i in range(3):
                b = bl[i]
                if b is None or np.isnan(self.area0[i]) or b[2] < self.area_frac * self.area0[i] or i == self.holding:
                    continue
                far = np.linalg.norm(ee[:2] - self.est[i][:2]) > 0.08
                if (i in self.lost or far) and self.est[i][2] < 0.05:
                    self.est[i][:2] = self.loc.to_xy(b[0], b[1], self.loc.plane_z)
        return self.est.copy()


def make_visual_verifier(env, loc: "FrontLocalizer", tol: float = 0.04):
    """画像から、「動かすキューブのうち、置き場所に置けていないもの」の番号の一覧を返す関数を作る。
    置き場所から tol [m] 以上離れた所に、そのキューブの色の領域が、はっきり見えたときだけ「置けていない」とする。
    見えない（腕に隠れている、自分の色の置き場所に重なっている）場合は、置けているものとして扱う。"""
    def verify():
        bl = loc.blobs()
        missing = []
        for i in range(3):
            if not env.active[i] or bl[i] is None:
                continue
            xy = loc.to_xy(bl[i][0], bl[i][1], loc.plane_z)
            if np.linalg.norm(xy - env.goal_xy(i)) > tol:
                missing.append(i)
        return missing
    return verify


class ProgressChecker:
    """画像から、「そのキューブ（番号）が、置き場所にある」かを答える。順番を進める判定に使う（動作の制御には使わない）。

    ・その色の領域が見えるとき: 置き場所から tol [m] 以内なら True。
    ・自分の色の置き場所に置いたとき（領域が置き場所と重なる）: 置き場所の領域の面積が、空のとき（reset 時に記録）より、15% 以上大きければ True。
      腕に隠れると、面積は減るだけなので、誤って True にはなりにくい。
    """

    def __init__(self, env, loc: "FrontLocalizer", tol: float = 0.04, grow: float = 1.15):
        self.env, self.loc, self.tol, self.grow = env, loc, tol, grow
        self.ref = [0, 0, 0]

    def reset(self):
        self.ref = [self.loc.pad_area(k) for k in range(3)]       # 空の置き場所の面積

    def __call__(self, i: int) -> bool:
        bl = self.loc.blobs()
        if bl[i] is not None:
            xy = self.loc.to_xy(bl[i][0], bl[i][1], self.loc.plane_z)
            return bool(np.linalg.norm(xy - self.env.goal_xy(i)) < self.tol)
        if self.env.goal[i] == i:
            a = self.loc.pad_area(i)
            return bool(self.ref[i] > 0 and a > self.grow * self.ref[i])
        return False


def make_progress_checker(env, loc: "FrontLocalizer", tol: float = 0.04):
    return ProgressChecker(env, loc, tol)
