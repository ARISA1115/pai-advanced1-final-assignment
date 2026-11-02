"""診断: 運ぶ順番を真の情報で与え（動作だけを見る）、最初のキューブに対して、どの段階まで進むかを数える。
段階: 近づく（手とキューブの水平距離 < 3 cm）→ 握る（近いときに、手の開きが 0.65 未満）→ 持ち上げる（キューブが 3 cm 以上浮く）→ 置く。"""
import sys, json
sys.path.insert(0, "src")
import numpy as np, torch
torch.set_num_threads(2)
from sorting.env import SortEnv
from sorting.visuomotor import FrontCam, load_visuo, proprio_vec

def run(path, seeds, steps=300):
    env = SortEnv(0, task="random"); env.fixed_order = True
    pol = load_visuo(path); cam = FrontCam(env, pol.res, pol.cams)
    st = dict(near=0, grasp=0, lift=0, placed=0); mind = []
    for s in seeds:
        env.reset(s); i0 = env.target_index()
        if i0 is None: continue
        z0 = env.cube_pos(i0)[2]; near = grasp = lift = False; md = 9
        for _ in range(steps):
            t = env.target_index()
            env.step(pol.act(cam.frame(), proprio_vec(env, t)))
            c = env.cube_pos(i0); d = np.linalg.norm(env.tcp()[:2] - c[:2]); md = min(md, d)
            if d < 0.03: near = True
            if near and d < 0.03 and env.grip_open() < 0.65: grasp = True
            if c[2] > z0 + 0.03: lift = True
            if env.placed(i0): break
        st["near"] += near; st["grasp"] += grasp; st["lift"] += lift; st["placed"] += bool(env.placed(i0)); mind.append(md)
    n = len(seeds)
    return {k: f"{v}/{n}" for k, v in st.items()} | {"最小水平距離の中央値[cm]": round(100 * float(np.median(mind)), 1)}

if __name__ == "__main__":
    seeds = list(range(10000, 10040))
    for p in sys.argv[1:]:
        print(p, run(p, seeds), flush=True)
