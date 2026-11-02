"""MuJoCo + Franka Panda の仕分けタスク環境。観測・行動・成功判定を持つ。"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[2] / "assets" / "franka_emika_panda" / "sorting_scene.xml"

N_CUBES = 3
CUBE_HALF = 0.02
PAD_XY = np.array([[0.62, -0.20], [0.62, 0.00], [0.62, 0.20]])
HOME_Q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
SUBSTEPS = 25          # 0.002 s * 25 = 0.05 s -> 20 Hz
MAX_DELTA = 0.02       # 1 制御あたりの手先の変位の上限 [m]
WS_LOW = np.array([0.25, -0.45, 0.012])
WS_HIGH = np.array([0.75, 0.45, 0.60])
TCP_OFFSET = np.array([0.0, 0.0, 0.1034])   # hand 座標系での指先の中心
OBS_DIM = 17
PLACE_XY_TOL = 0.03
PLACE_Z_MAX = 0.045


class SortEnv:
    def __init__(self, seed: int = 0, task: str = "random"):
        """task: "color" = 色と同じ置き場所（段階 0）、"random" = 目標をエピソードごとに変える"""
        self.task = task
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        for b in range(self.m.nbody):
            name = self.m.body(b).name
            if name and not name.startswith("cube_"):
                self.m.body_gravcomp[b] = 1.0   # ロボット自身の重力を補償（位置制御の垂れ下がり防止）
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self.hand = self.m.body("hand").id
        self.cube_body = [self.m.body(f"cube_{i}").id for i in range(N_CUBES)]
        self.cube_qadr = [self.m.joint(f"cube_{i}_joint").qposadr[0] for i in range(N_CUBES)]
        self.arm_qadr = [self.m.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]
        self.arm_dofadr = [self.m.joint(f"joint{i}").dofadr[0] for i in range(1, 8)]
        self.f_qadr = [self.m.joint("finger_joint1").qposadr[0], self.m.joint("finger_joint2").qposadr[0]]
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.ctrl_grip = 255.0
        self.steps = 0
        self.goal = np.arange(N_CUBES)          # キューブ i の目標の置き場所
        self.active = np.ones(N_CUBES, bool)    # 動かすキューブ
        self.fixed_order = False                # True なら、置いていないキューブを、番号の小さい順に取る（視覚運動方策の実験用）
        self.reset(seed)

    # ---- 幾何 -------------------------------------------------------
    def tcp(self) -> np.ndarray:
        R = self.d.xmat[self.hand].reshape(3, 3)
        return self.d.xpos[self.hand] + R @ TCP_OFFSET

    def cube_pos(self, i: int) -> np.ndarray:
        return self.d.xpos[self.cube_body[i]].copy()

    def grip_open(self) -> float:
        return float((self.d.qpos[self.f_qadr[0]] + self.d.qpos[self.f_qadr[1]]) / 0.08)

    def goal_xy(self, i: int) -> np.ndarray:
        return PAD_XY[self.goal[i]]

    def placed(self, i: int, pos: np.ndarray | None = None) -> bool:
        """動かさないキューブは、最初から「置けている」扱い。pos を渡すと、その位置（推定値）で判定する。"""
        if not self.active[i]:
            return True
        p = self.cube_pos(i) if pos is None else pos
        return bool(np.linalg.norm(p[:2] - self.goal_xy(i)) < PLACE_XY_TOL and p[2] < PLACE_Z_MAX)

    def n_active(self) -> int:
        return int(self.active.sum())

    def n_placed(self) -> int:
        """動かすキューブのうち、置けた数。"""
        return sum(self.placed(i) for i in range(N_CUBES) if self.active[i])

    def success(self) -> bool:
        return self.n_placed() == self.n_active()

    def set_task(self, goal, active):
        self.goal = np.asarray(goal, dtype=int).copy()
        self.active = np.asarray(active, dtype=bool).copy()

    # ---- 乱数配置 ----------------------------------------------------
    def reset(self, seed: int | None = None, goal=None, active=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.arm_qadr] = HOME_Q
        self.d.qpos[self.f_qadr] = 0.04
        self.d.ctrl[:7] = HOME_Q
        self.d.ctrl[7] = 255.0
        self.ctrl_grip = 255.0
        self.q_cmd = HOME_Q.copy()
        pts: list[np.ndarray] = []
        while len(pts) < N_CUBES:
            p = np.array([self.rng.uniform(0.30, 0.48), self.rng.uniform(-0.30, 0.30)])
            if all(np.linalg.norm(p - q) > 0.07 for q in pts):
                pts.append(p)
        for i, p in enumerate(pts):
            a = self.cube_qadr[i]
            self.d.qpos[a:a + 3] = [p[0], p[1], CUBE_HALF]
            self.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        if self.task == "random":
            k = int(self.rng.choice([1, 2, 3], p=[0.2, 0.3, 0.5]))
            idx = self.rng.permutation(N_CUBES)
            self.active = np.zeros(N_CUBES, bool); self.active[idx[:k]] = True
            self.goal = self.rng.permutation(N_CUBES)
        else:
            self.goal = np.arange(N_CUBES); self.active = np.ones(N_CUBES, bool)
        mujoco.mj_forward(self.m, self.d)
        for _ in range(40):
            self._sim(0.0, hold=True)
        if goal is not None:                      # 外から目標を与える（自然言語の指示など）
            self.set_task(goal, np.ones(N_CUBES, bool) if active is None else active)
        self.steps = 0
        self.home_R = self.d.xmat[self.hand].reshape(3, 3).copy()
        return self.obs()

    # ---- 観測 --------------------------------------------------------
    def target_index(self, est: np.ndarray | None = None) -> int | None:
        """est: 画像から推定したキューブの位置 (3, 3)。None ならシミュレータの真値。"""
        ee = self.tcp()
        best, bd = None, 1e9
        for i in range(N_CUBES):
            pi = None if est is None else est[i]
            if self.placed(i, pi):
                continue
            if self.fixed_order:
                return i
            dist = np.linalg.norm((self.cube_pos(i) if est is None else est[i]) - ee)
            if dist < bd:
                best, bd = i, dist
        return best

    def obs(self, est: np.ndarray | None = None) -> np.ndarray:
        ee = self.tcp()
        i = self.target_index(est)
        if est is None:
            remaining = self.n_active() - self.n_placed()
        else:
            remaining = sum(1 for k in range(N_CUBES) if self.active[k] and not self.placed(k, est[k]))
        if i is None:
            i = 0
        c = self.cube_pos(i) if est is None else est[i]
        gxy = self.goal_xy(i)
        pad = np.array([gxy[0], gxy[1], CUBE_HALF])
        onehot = np.zeros(3)
        onehot[self.goal[i]] = 1.0          # 目標の置き場所（色ではない）
        lifted = 1.0 if c[2] > CUBE_HALF + 0.02 else 0.0
        return np.concatenate([
            ee, [self.grip_open()], c - ee, c - pad, onehot,
            [lifted, remaining / N_CUBES, ee[2], CUBE_HALF],
        ]).astype(np.float32)

    # ---- 制御 --------------------------------------------------------
    def _ik_step(self, target: np.ndarray) -> np.ndarray:
        """手先位置を target へ近づける関節角（姿勢は初期姿勢を保つ）。"""
        pt = self.tcp()
        mujoco.mj_jac(self.m, self.d, self.jacp, self.jacr, pt, self.hand)
        J = np.vstack([self.jacp[:, self.arm_dofadr], self.jacr[:, self.arm_dofadr]])
        R = self.d.xmat[self.hand].reshape(3, 3)
        Re = self.home_R @ R.T
        rot_err = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        err = np.concatenate([target - pt, 1.0 * rot_err])
        lam = 0.05
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), err)
        dq = np.clip(dq, -0.15, 0.15)
        q_act = self.d.qpos[self.arm_qadr]
        # 指令角を保持して積算する（現在角から作り直すと、重力の垂れ下がりが累積して沈む）
        q = np.clip(self.q_cmd + dq, q_act - 0.3, q_act + 0.3)
        lo = self.m.jnt_range[[self.m.joint(f"joint{i}").id for i in range(1, 8)], 0]
        hi = self.m.jnt_range[[self.m.joint(f"joint{i}").id for i in range(1, 8)], 1]
        return np.clip(q, lo, hi)

    def _sim(self, g: float, hold: bool = False):
        for _ in range(SUBSTEPS if not hold else 1):
            mujoco.mj_step(self.m, self.d)

    def step(self, action: np.ndarray, substep_cb=None):
        a = np.asarray(action, dtype=np.float64)
        delta = np.clip(a[:3], -MAX_DELTA, MAX_DELTA)
        target = np.clip(self.tcp() + delta, WS_LOW, WS_HIGH)
        self.q_cmd = self._ik_step(target)
        self.d.ctrl[:7] = self.q_cmd
        self.ctrl_grip = 255.0 if a[3] < 0 else 0.0
        self.d.ctrl[7] = self.ctrl_grip
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)
            if substep_cb is not None:
                substep_cb()
        self.steps += 1
        return self.obs(), self.success()
