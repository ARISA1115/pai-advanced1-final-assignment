"""録画と字幕の描画。"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class Recorder:
    """シミュレーション時間に合わせて、一定の fps でフレームを集める。"""

    def __init__(self, env, width=1280, height=720, fps=30):
        self.env, self.fps = env, fps
        self.r = mujoco.Renderer(env.m, height, width)
        self.cam = "main"
        self.frames: list[np.ndarray] = []
        self.next_t = 0.0
        self.caption_fn = None
        self.scene_hook = None      # レンダリング用のシーンに、目印（球など）を足す関数

    def reset_clock(self):
        self.next_t = self.env.d.time

    def maybe_frame(self):
        d = self.env.d
        if d.time + 1e-9 >= self.next_t:
            self.r.update_scene(d, camera=self.cam)
            if self.scene_hook is not None:
                self.scene_hook(self.r.scene)
            img = self.r.render()
            if self.caption_fn is not None:
                img = self.caption_fn(img)
            self.frames.append(img)
            self.next_t += 1.0 / self.fps


def draw_caption(img: np.ndarray, lines: list[str], size: int = 30) -> np.ndarray:
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    f = font(size)
    h = int(size * 1.5) * len(lines) + 20
    dr.rectangle([0, im.height - h, im.width, im.height], fill=(0, 0, 0, 150))
    y = im.height - h + 10
    for ln in lines:
        dr.text((24, y), ln, font=f, fill=(255, 255, 255, 255))
        y += int(size * 1.5)
    return np.array(Image.alpha_composite(im, ov).convert("RGB"))


def _wrap(text: str, f, max_w: int) -> list[str]:
    """画面の幅に収まるように折り返す。英数字・数字・記号のかたまり（99.9% など）は、途中で切らない。"""
    import re
    tokens = re.findall(r"[A-Za-z0-9.%+\-_/:=<>]+|\s+|.", text)
    out, cur = [], ""
    for t in tokens:
        if f.getlength(cur + t) > max_w and cur.strip():
            out.append(cur.rstrip()); cur = ("  " if text.startswith(" ") else "") + t.lstrip()
        else:
            cur += t
    out.append(cur.rstrip())
    return out


def card(width: int, height: int, title: str, body: list[str], title_size=54, body_size=32,
         bg=(20, 28, 44)) -> np.ndarray:
    im = Image.new("RGB", (width, height), bg)
    dr = ImageDraw.Draw(im)
    dr.text((80, 70), title, font=font(title_size), fill=(255, 255, 255))
    y = 70 + int(title_size * 1.7)
    f = font(body_size)
    for ln in body:
        for sub in (_wrap(ln, f, width - 160) if ln else [""]):
            dr.text((80, y), sub, font=f, fill=(210, 220, 235))
            y += int(body_size * 1.45)
    return np.array(im)


def add_sphere(scene, pos, rgba, radius=0.018):
    """描画用のシーンに、目印の球を 1 つ足す（物理には影響しない）。"""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0]), np.asarray(pos, dtype=np.float64),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def banner(img: np.ndarray, lines: list[str], size: int = 30, top: bool = True, color=(0, 0, 0, 170)) -> np.ndarray:
    """画面の上（または下）に、半透明の帯と文字を重ねる。長い行は、画面の幅で折り返す。"""
    im = Image.fromarray(img).convert("RGBA")
    f = font(size)
    wrapped = []
    for ln in lines:
        wrapped += _wrap(ln, f, im.width - 48)
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    h = int(size * 1.5) * len(wrapped) + 20
    y0 = 0 if top else im.height - h
    dr.rectangle([0, y0, im.width, y0 + h], fill=color)
    y = y0 + 10
    for ln in wrapped:
        dr.text((24, y), ln, font=f, fill=(255, 255, 255, 255))
        y += int(size * 1.5)
    return np.array(Image.alpha_composite(im, ov).convert("RGB"))
