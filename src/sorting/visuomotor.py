"""視覚運動方策: カメラ画像から、直接、手先の動作を出す（キューブの位置を、別に推定しない）。

入力: 前方カメラの画像（96×96）、ロボット自身の手先・グリッパー、今のタスク（運ぶキューブの色と、置き場所）。
出力: 手先の変位と把持（状態ベクトルの方策と、同じ 4 次元）。
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch
from torch import nn

from .env import MAX_DELTA

RES = 96
VEC_DIM = 4 + 3 + 3


class FrontCam:
    def __init__(self, env, res: int = RES, cams: tuple = ("front",)):
        """cams: 使うカメラの名前（"front" 前方斜め上、"top" 真上、"side" 横）。画像は、チャンネルの向きに重ねる。"""
        self.env, self.cams = env, tuple(cams)
        self.r = mujoco.Renderer(env.m, res, res)

    def frame(self) -> np.ndarray:
        out = []
        for c in self.cams:
            self.r.update_scene(self.env.d, camera=c)
            out.append(self.r.render().transpose(2, 0, 1))
        return np.concatenate(out, axis=0).copy()              # (3 × カメラ数, H, W) uint8


def proprio_vec(env, target: int | None) -> np.ndarray:
    """手先 3 + グリッパー 1 + 運ぶキューブの色（one-hot 3）+ 置き場所（one-hot 3）。キューブの位置は、含まない。"""
    ee = (env.tcp() - np.array([0.45, 0.0, 0.2])) / np.array([0.2, 0.3, 0.2])
    cube = np.zeros(3); pad = np.zeros(3)
    if target is not None:
        cube[target] = 1.0; pad[env.goal[target]] = 1.0
    return np.concatenate([ee, [env.grip_open()], cube, pad]).astype(np.float32)


class Planner:
    """運ぶキューブの順番を決める。ロボット自身の量（グリッパーの動き、手先の位置）だけで、次へ進める。

    strict=False（初版）: つかんだあとに、グリッパーが開いたら、次へ進む。
    strict=True（改良）: さらに、そのとき手先が、置き場所の真上の低い位置にあるときだけ進める。
        途中で落とした、早く開いた、などの場合は、同じキューブを、やり直す。
    使う情報は、手先の位置・グリッパーの開き（ロボット自身の量）と、置き場所の位置（タスクで与えられる既知の配置）だけ。
    """

    def __init__(self, env, strict: bool = False, verifier=None, retries: int = 2, progress_checker=None):
        """verifier: 引数なしで呼ぶと、画像から「置き場所に置けていないキューブの番号の一覧」を返す関数（任意）。
        progress_checker: 引数（キューブの番号）で呼ぶと、画像から「そのキューブが、置き場所にある」かを返す関数（任意）。
        グリッパーが開いている間（置いたあと）、これが True を返したら、すぐに次のキューブへ進める。"""
        self.env, self.strict, self.verifier, self.max_retries = env, strict, verifier, retries
        self.progress_checker = progress_checker
        self.reset()

    def reset(self):
        self.order = [i for i in range(3) if self.env.active[i]]
        self.k = 0; self.closed_steps = 0; self.idle = 0; self.retries_left = self.max_retries; self.check_window = 0
        if getattr(self, 'progress_checker', None) is not None and hasattr(self.progress_checker, 'reset'):
            self.progress_checker.reset()               # 空の置き場所の面積を記録する（キューブが、まだ置き場所にない時点）

    @property
    def target(self):
        return self.order[self.k] if self.k < len(self.order) else None

    def _released_over_goal(self) -> bool:
        ee = self.env.tcp(); gxy = self.env.goal_xy(self.target)
        return bool(np.linalg.norm(ee[:2] - gxy) < 0.05 and ee[2] < 0.10)

    def update(self):
        og = self.env.grip_open()
        if 0.25 < og < 0.65:
            self.closed_steps += 1                       # 何かをつかんでいる
        elif og > 0.85 and self.closed_steps >= 3:       # つかんだあとに、開いた = 置いた（候補）
            if self.closed_steps >= 8 and ((not self.strict) or self._released_over_goal()):
                self.k += 1
            elif self.progress_checker is not None:
                self.check_window = 30                   # 規則では進めなかった。画像で確認する時間（30 ステップ）を設ける
            self.closed_steps = 0
        elif og <= 0.25:
            self.closed_steps = 0                        # 何もつかめなかった
        if self.check_window > 0 and self.target is not None:
            self.check_window -= 1
            if self.check_window % 3 == 0 and self.progress_checker(self.target):   # 画像で、今の目標が置き場所にあると分かったら、次へ
                self.k += 1; self.check_window = 0
        if self.target is None and self.verifier is not None and self.retries_left > 0:
            self.idle += 1                               # 順番が終わった。手が離れて、画像で確認できるまで、少し待つ
            if self.idle >= 20:
                missing = self.verifier()
                if missing:                              # 置き場所に置けていないキューブがある → やり直す
                    self.order, self.k, self.retries_left = list(missing), 0, self.retries_left - 1
                self.idle = 0
        return self.target


class VisuoPolicy(nn.Module):
    def __init__(self, keypoints: int = 32, in_ch: int = 3):
        super().__init__()

        def blk(i, o, k, s):
            return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2), nn.GroupNorm(8, o), nn.ReLU())

        self.conv = nn.Sequential(blk(in_ch, 32, 5, 2), blk(32, 64, 3, 2), blk(64, 64, 3, 1), blk(64, 64, 3, 1))
        self.kp = nn.Conv2d(64, keypoints, 1)
        self.temperature = nn.Parameter(torch.ones(1))
        self.head = nn.Sequential(nn.Linear(keypoints * 2 + VEC_DIM, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))

    def forward(self, img: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        h = self.kp(self.conv(img.float() / 255.0))
        b, k, hh, ww = h.shape
        w = torch.softmax(h.reshape(b, k, -1) * self.temperature, dim=-1).reshape(b, k, hh, ww)
        ys = torch.linspace(-1, 1, hh, device=img.device); xs = torch.linspace(-1, 1, ww, device=img.device)
        ex = (w.sum(2) * xs).sum(-1); ey = (w.sum(3) * ys).sum(-1)
        return self.head(torch.cat([ex, ey, vec], dim=1))

    @torch.no_grad()
    def act(self, img: np.ndarray, vec: np.ndarray) -> np.ndarray:
        y = self.forward(torch.as_tensor(img).unsqueeze(0), torch.as_tensor(vec).unsqueeze(0))[0].numpy()
        return np.array([y[0] * MAX_DELTA, y[1] * MAX_DELTA, y[2] * MAX_DELTA, 1.0 if y[3] > 0 else -1.0])


class VisuoPolicyFiLM(nn.Module):
    """タスク（運ぶキューブの色と、置き場所）を、画像の特徴に直接かける（FiLM）構造。
    画像の中の「どのキューブか」を、タスクで選べるようにする。"""

    def __init__(self, keypoints: int = 32, width: int = 64, in_ch: int = 3):
        super().__init__()

        def blk(i, o, k, s):
            return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2), nn.GroupNorm(8, o), nn.ReLU())

        self.c1 = blk(in_ch, 32, 5, 2); self.c2 = blk(32, width, 3, 2)
        self.c3 = blk(width, width, 3, 1); self.c4 = blk(width, width, 3, 1)
        self.film2 = nn.Linear(VEC_DIM, 2 * width); self.film3 = nn.Linear(VEC_DIM, 2 * width)
        self.kp = nn.Conv2d(width, keypoints, 1)
        self.temperature = nn.Parameter(torch.ones(1))
        self.head = nn.Sequential(nn.Linear(keypoints * 2 + VEC_DIM, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))

    @staticmethod
    def _film(h, layer, vec):
        g, b = layer(vec).chunk(2, dim=1)
        return h * (1 + g[:, :, None, None]) + b[:, :, None, None]

    def forward(self, img: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        h = self.c2(self.c1(img.float() / 255.0))
        h = self._film(h, self.film2, vec)
        h = self._film(self.c3(h), self.film3, vec)
        h = self.c4(h)
        h = self.kp(h)
        b, k, hh, ww = h.shape
        w = torch.softmax(h.reshape(b, k, -1) * self.temperature, dim=-1).reshape(b, k, hh, ww)
        ys = torch.linspace(-1, 1, hh, device=img.device); xs = torch.linspace(-1, 1, ww, device=img.device)
        ex = (w.sum(2) * xs).sum(-1); ey = (w.sum(3) * ys).sum(-1)
        return self.head(torch.cat([ex, ey, vec], dim=1))

    @torch.no_grad()
    def act(self, img: np.ndarray, vec: np.ndarray) -> np.ndarray:
        y = self.forward(torch.as_tensor(img).unsqueeze(0), torch.as_tensor(vec).unsqueeze(0))[0].numpy()
        return np.array([y[0] * MAX_DELTA, y[1] * MAX_DELTA, y[2] * MAX_DELTA, 1.0 if y[3] > 0 else -1.0])


def make_visuo(arch: str, in_ch: int = 3):
    if arch == "resnet":
        return VisuoResNet(n_views=in_ch // 3)
    return VisuoPolicyFiLM(in_ch=in_ch) if arch == "film" else VisuoPolicy(in_ch=in_ch)


class Unstick:
    """方策の出力が、同じ場所で止まったら（決定的な方策の、固定点）、少し動かしてから、方策に戻す。

    使う情報は、手先の位置だけ（キューブの真の位置は使わない）。
    window ステップの間、手先の動きが span [m] 未満なら、ランダムな水平方向 + 上向きへ、steps ステップ動かす。握る・開くは、方策の出力のまま。
    """

    def __init__(self, env, window: int = 30, span: float = 0.008, steps: int = 15, seed: int = 0):
        self.env, self.window, self.span, self.steps = env, window, span, steps
        self.rng = np.random.default_rng(seed); self.reset()

    def reset(self):
        self.hist, self.left, self.dir, self.count = [], 0, None, 0

    def filter(self, a: np.ndarray) -> np.ndarray:
        self.hist.append(self.env.tcp().copy())
        if self.left > 0:
            self.left -= 1
            return np.array([*self.dir, a[3]])
        if len(self.hist) >= self.window:
            h = np.array(self.hist[-self.window:])
            if np.ptp(h, axis=0).max() < self.span:
                th = self.rng.uniform(0, 2 * np.pi)
                self.dir = np.array([np.cos(th) * MAX_DELTA * 0.6, np.sin(th) * MAX_DELTA * 0.6, MAX_DELTA * 0.5])
                self.left = self.steps; self.hist = []; self.count += 1
                return np.array([*self.dir, a[3]])
        return a


class EnsemblePolicy:
    """複数の方策の出力（握る・開くは、符号を付ける前の値）を平均する。入力の画像の形は、全員同じであること。"""

    def __init__(self, models):
        self.models = models; self.cams = models[0].cams; self.res = models[0].res

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, img: np.ndarray, vec: np.ndarray) -> np.ndarray:
        x, v = torch.as_tensor(img).unsqueeze(0), torch.as_tensor(vec).unsqueeze(0)
        y = torch.stack([m.forward(x, v)[0] for m in self.models]).mean(0).numpy()
        return np.array([y[0] * MAX_DELTA, y[1] * MAX_DELTA, y[2] * MAX_DELTA, 1.0 if y[3] > 0 else -1.0])


def load_visuo(path: str):
    if "+" in path:                                      # 「a.pt+b.pt+c.pt」は、方策の平均
        return EnsemblePolicy([load_visuo(p) for p in path.split("+")])
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("arch") in ("diffusion", "diffusion_g"):
        m = VisuoDiffusion(n_views=ck["in_ch"] // 3, grip_head=ck["arch"] == "diffusion_g")
    else:
        m = make_visuo(ck.get("arch", "base"), ck.get("in_ch", 3))
    m.load_state_dict(ck["state_dict"]); m.eval()
    m.cams = tuple(ck.get("cams", ("front",))); m.res = int(ck.get("res", RES))
    return m


# ---------------------------------------------------------------------------
# 事前学習した ResNet18（ImageNet）を使う視覚運動方策。複数の視点、動作の系列（チャンク）の出力
# ---------------------------------------------------------------------------
class _Basic(nn.Module):
    """torchvision の BasicBlock と、同じパラメータ名（事前学習の重みを、そのまま読み込むため）。"""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + (x if self.downsample is None else self.downsample(x)))


class ResNet18Trunk(nn.Module):
    """ResNet18 の、layer2 まで（ストライド 8、128 チャンネル）。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False); self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(_Basic(64, 64, 1), _Basic(64, 64, 1))
        self.layer2 = nn.Sequential(_Basic(64, 128, 2), _Basic(128, 128, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):                                   # x: float 0〜1 (B, 3, H, W)
        x = (x - self.mean) / self.std
        x = torch.relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, 3, 2, 1)
        return self.layer2(self.layer1(x))

    def load_pretrained(self, path: str):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        keep = {k: v for k, v in sd.items() if k.split(".")[0] in ("conv1", "bn1", "layer1", "layer2")}
        missing, unexpected = self.load_state_dict(keep, strict=False)
        assert not unexpected and all(k in ("mean", "std") for k in missing), (missing, unexpected)


CHUNK = 8


class VisuoResNet(nn.Module):
    """画像（複数の視点）→ ResNet18 → キーポイント → ロボットの状態・タスクと結合 → 動作の系列（CHUNK ステップ分）。"""

    def __init__(self, n_views: int = 2, keypoints: int = 32, chunk: int = CHUNK):
        super().__init__()
        self.n_views, self.chunk = n_views, chunk
        self.trunk = ResNet18Trunk()
        self.kp = nn.Conv2d(128, keypoints, 1)
        self.temperature = nn.Parameter(torch.ones(1))
        self.head = nn.Sequential(nn.Linear(n_views * keypoints * 2 + VEC_DIM, 512), nn.ReLU(),
                                  nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, chunk * 4))

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.trunk.modules():                       # 事前学習した BatchNorm の統計は、固定する
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, img: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:    # img: uint8 (B, 3 × 視点, H, W)
        b = img.shape[0]
        x = img.float().div(255.0).reshape(b * self.n_views, 3, *img.shape[2:])
        h = self.kp(self.trunk(x))
        bk, k, hh, ww = h.shape
        w = torch.softmax(h.reshape(bk, k, -1) * self.temperature, dim=-1).reshape(bk, k, hh, ww)
        ys = torch.linspace(-1, 1, hh, device=img.device); xs = torch.linspace(-1, 1, ww, device=img.device)
        ex = (w.sum(2) * xs).sum(-1); ey = (w.sum(3) * ys).sum(-1)
        feat = torch.cat([ex, ey], dim=1).reshape(b, -1)
        return self.head(torch.cat([feat, vec], dim=1)).reshape(b, self.chunk, 4)

    @torch.no_grad()
    def act(self, img: np.ndarray, vec: np.ndarray) -> np.ndarray:
        y = self.forward(torch.as_tensor(img).unsqueeze(0), torch.as_tensor(vec).unsqueeze(0))[0, 0].numpy()
        return np.array([y[0] * MAX_DELTA, y[1] * MAX_DELTA, y[2] * MAX_DELTA, 1.0 if y[3] > 0 else -1.0])


# ---------------------------------------------------------------------------
# Diffusion Policy（拡散モデルで、動作の系列を生成する方策）
#   観測（複数視点の画像のキーポイント + ロボットの状態・タスク）を条件に、ノイズから CHUNK ステップ分の動作を生成する。
#   学習: 動作の系列にノイズを加え、加えたノイズを当てる（DDPM、T = DIFF_T）。生成: DDIM で DIFF_SAMPLE ステップ。
#   実行: 生成した系列のうち、先頭の EXEC ステップを実行して、作り直す（receding horizon）。
#   デノイザーは、残差 MLP（時刻・観測を、各ブロックに足し込む）。
# ---------------------------------------------------------------------------
DIFF_T = 50
DIFF_SAMPLE = 8
EXEC = 4


def _alpha_bar(t: int = DIFF_T) -> torch.Tensor:
    s = 0.008; x = torch.linspace(0, t, t + 1)
    f = torch.cos((x / t + s) / (1 + s) * np.pi / 2) ** 2
    ab = (f / f[0])[1:]
    return ab.clamp(1e-4, 0.9999)


class _ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n = nn.LayerNorm(d); self.f = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

    def forward(self, x, c):
        return x + self.f(self.n(x + c))


class VisuoDiffusion(VisuoResNet):
    def __init__(self, n_views: int = 2, keypoints: int = 32, chunk: int = CHUNK, width: int = 512, blocks: int = 4,
                 grip_head: bool = False):
        super().__init__(n_views, keypoints, chunk)
        self.head = None
        # grip_head: 握る・開くを、拡散で生成せず、観測から直接分類する（拡散は、手先の変位 3 次元だけ）
        self.grip_head = grip_head; self.adim = 3 if grip_head else 4
        if grip_head:
            self.grip = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, chunk))
        od = n_views * keypoints * 2 + VEC_DIM
        self.obs_mlp = nn.Sequential(nn.Linear(od, width), nn.GELU(), nn.Linear(width, width))
        self.t_emb = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, width))
        self.inp = nn.Linear(chunk * self.adim, width)
        self.blocks = nn.ModuleList([_ResBlock(width) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, chunk * self.adim))
        self.register_buffer("ab", _alpha_bar())
        self.queue: list = []

    def reset(self):
        self.queue = []

    def _obs(self, img: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        b = img.shape[0]
        x = img.float().div(255.0).reshape(b * self.n_views, 3, *img.shape[2:])
        h = self.kp(self.trunk(x))
        bk, k, hh, ww = h.shape
        w = torch.softmax(h.reshape(bk, k, -1) * self.temperature, dim=-1).reshape(bk, k, hh, ww)
        ys = torch.linspace(-1, 1, hh, device=img.device); xs = torch.linspace(-1, 1, ww, device=img.device)
        ex = (w.sum(2) * xs).sum(-1); ey = (w.sum(3) * ys).sum(-1)
        return self.obs_mlp(torch.cat([torch.cat([ex, ey], dim=1).reshape(b, -1), vec], dim=1))

    def _eps(self, a_noisy: torch.Tensor, t: torch.Tensor, obs_c: torch.Tensor) -> torch.Tensor:
        f = torch.arange(16, device=t.device)
        ang = t.float()[:, None] / DIFF_T * 1000.0 * torch.exp(-f * 0.5)[None]
        c = self.t_emb(torch.cat([ang.sin(), ang.cos()], 1)) + obs_c
        x = self.inp(a_noisy.reshape(len(a_noisy), -1))
        for blk in self.blocks:
            x = blk(x, c)
        return self.out(x).reshape(-1, self.chunk, self.adim)

    def loss(self, img, vec, y, m):
        """y: (B, chunk, 4)  [-1, 1]、m: (B, chunk) 有効な位置。加えたノイズの予測誤差（有効な位置だけ）。"""
        b = y.shape[0]
        obs_c = self._obs(img, vec)
        g = y[..., 3]; y = y[..., :self.adim]
        t = torch.randint(0, DIFF_T, (b,), device=y.device)
        e = torch.randn_like(y)
        ab = self.ab[t][:, None, None]
        pred = self._eps(ab.sqrt() * y + (1 - ab).sqrt() * e, t, obs_c)
        loss = (((pred - e) ** 2).mean(-1) * m).sum() / m.sum()
        if self.grip_head:
            bce = nn.functional.binary_cross_entropy_with_logits(self.grip(obs_c), (g > 0).float(), reduction="none")
            loss = loss + (bce * m).sum() / m.sum()
        return loss

    @torch.no_grad()
    def sample(self, img: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        obs_c = self._obs(img, vec); b = img.shape[0]
        a = torch.randn(b, self.chunk, self.adim, device=img.device)
        ts = torch.linspace(DIFF_T - 1, 0, DIFF_SAMPLE).long()
        for j, t in enumerate(ts):
            tt = torch.full((b,), int(t), device=img.device)
            e = self._eps(a, tt, obs_c)
            ab = self.ab[t]; abn = self.ab[ts[j + 1]] if j + 1 < len(ts) else torch.tensor(1.0)
            x0 = ((a - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1)
            a = abn.sqrt() * x0 + (1 - abn).sqrt() * e
        a = a.clamp(-1, 1)
        if self.grip_head:
            a = torch.cat([a, torch.where(self.grip(obs_c) > 0, 1.0, -1.0)[..., None]], -1)
        return a

    @torch.no_grad()
    def act(self, img: np.ndarray, vec: np.ndarray) -> np.ndarray:
        if not self.queue:
            seq = self.sample(torch.as_tensor(img).unsqueeze(0), torch.as_tensor(vec).unsqueeze(0))[0].numpy()
            self.queue = list(seq[:EXEC])
        y = self.queue.pop(0)
        return np.array([y[0] * MAX_DELTA, y[1] * MAX_DELTA, y[2] * MAX_DELTA, 1.0 if y[3] > 0 else -1.0])
