"""状態を見て行動を決める、幾何ベースのエキスパート（状態を持たない）。

方策と同じ観測・同じ行動（手先の変位 3 次元 + 把持 1 次元）を出す。
DAgger で、方策が到達した状態に対しても、正解の行動を付け直せる。
"""
from __future__ import annotations

import numpy as np

from .env import MAX_DELTA, CUBE_HALF, SortEnv

HOVER_Z = 0.16     # 移動中の手先の高さ
GRASP_Z = 0.025    # つかむときの手先の高さ
PLACE_Z = 0.035    # 置くときの手先の高さ


def _toward(ee: np.ndarray, goal: np.ndarray) -> np.ndarray:
    d = goal - ee
    n = np.linalg.norm(d)
    if n > MAX_DELTA:
        d = d / n * MAX_DELTA
    return d


def expert_action(env: SortEnv) -> np.ndarray:
    ee = env.tcp()
    i = env.target_index()
    if i is None:
        return np.array([0.0, 0.0, MAX_DELTA, -1.0])    # 全部置けたので、上がって待つ
    c = env.cube_pos(i)
    pad = env.goal_xy(i)
    og = env.grip_open()
    h = ee[2]
    dist_ee_cube = np.linalg.norm(c - ee)
    carrying = dist_ee_cube < 0.035 and 0.25 < og < 0.65

    if carrying:
        dxy_pad = np.linalg.norm(c[:2] - pad)
        goal_xy = pad - (c[:2] - ee[:2])   # つかんだ位置のずれを補正して、キューブ自体を置き場所へ合わせる
        if dxy_pad > 0.012:
            # 上がってから横へ。低いうちは、真上に上がる
            if h < 0.12:
                return np.array([0.0, 0.0, MAX_DELTA, 1.0])
            goal = np.array([goal_xy[0], goal_xy[1], HOVER_Z + 0.02])
            return np.append(_toward(ee, goal), 1.0)
        # 置き場所の真上：下げて、接地したら開く
        if c[2] > CUBE_HALF + 0.006:
            goal = np.array([goal_xy[0], goal_xy[1], PLACE_Z])
            return np.append(_toward(ee, goal), 1.0)
        return np.array([0.0, 0.0, 0.0, -1.0])

    dxy = np.linalg.norm(c[:2] - ee[:2])
    near = dxy < 0.012 and h < 0.032
    if near:
        if og < 0.2:                        # 何もつかめずに閉じた
            return np.array([0.0, 0.0, MAX_DELTA, -1.0])
        return np.array([0.0, 0.0, 0.0, 1.0])  # 閉じる（閉じ終わるまで待つ）
    if dxy > (0.02 if h < 0.10 else 0.01):
        if h < 0.10 and dxy > 0.02:
            return np.array([0.0, 0.0, MAX_DELTA, -1.0])     # 低いうちは、先に上がる
        goal = np.array([c[0], c[1], HOVER_Z])
        return np.append(_toward(ee, goal), -1.0)
    goal = np.array([c[0], c[1], GRASP_Z])
    return np.append(_toward(ee, goal), -1.0)
