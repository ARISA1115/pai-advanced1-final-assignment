"""視覚運動方策のデータを集める。エキスパート（ノイズ付き）または学習した方策（DAgger）で動かし、
画像・ロボット自身の状態・タスクを入力、特権情報（キューブの真の位置）を見るエキスパートの動作を正解として記録する。"""
import argparse
import time

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from sorting.env import SortEnv
from sorting.expert import expert_action
from sorting.visuomotor import FrontCam, Planner, load_visuo, proprio_vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=100000)
    ap.add_argument("--policy", default=None, help="指定すると、その方策で動かす（DAgger）。なければ、ノイズ付きのエキスパート")
    ap.add_argument("--max-steps", type=int, default=450)
    ap.add_argument("--cams", default="front", help="使うカメラ（カンマ区切り。例: front,top）")
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--npy", action="store_true", help=".npz でなく、--out を接頭辞にして、_img.npy / _vec.npy / _y.npy に保存")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    k, n = map(int, a.shard.split("/"))
    env = SortEnv(0, task="random"); env.fixed_order = True
    cams = tuple(a.cams.split(","))
    cam = FrontCam(env, a.res, cams); planner = Planner(env)
    pol = load_visuo(a.policy) if a.policy else None
    seeds = list(range(a.seed0, a.seed0 + a.episodes))[k::n]
    I, V, Y, EP, SRC = [], [], [], [], []
    ok_n, t0 = 0, time.time()
    for j, s in enumerate(seeds):
        env.reset(s); planner.reset()
        rng = np.random.default_rng(s)
        sigma = [0.0, 0.004, 0.008][j % 3]
        for step in range(a.max_steps):
            tgt = planner.update()
            label = expert_action(env)                      # 特権情報（真の位置）を見る
            img = cam.frame(); vec = proprio_vec(env, tgt if tgt is not None else None)
            I.append(img); V.append(vec); Y.append(label.astype(np.float32)); EP.append(j); SRC.append(0 if pol is None else 1)
            if pol is None:
                act = label.copy(); act[:3] += rng.normal(0, sigma, 3)
            else:
                act = pol.act(img, vec)
            _, ok = env.step(act)
            if ok:
                break
        ok_n += int(env.success())
        if (j + 1) % 10 == 0:
            print(f"[{k}/{n}] {j+1}/{len(seeds)} 成功 {ok_n/(j+1):.2f} サンプル {len(I)} ({time.time()-t0:.0f}s)", flush=True)
    if a.npy:
        np.save(a.out + "_img.npy", np.array(I, np.uint8)); np.save(a.out + "_vec.npy", np.array(V, np.float32)); np.save(a.out + "_y.npy", np.array(Y, np.float32)); np.save(a.out + "_ep.npy", np.array(EP, np.int32)); np.save(a.out + "_src.npy", np.array(SRC, np.int8))
    else:
        np.savez(a.out, img=np.array(I, np.uint8), vec=np.array(V, np.float32), y=np.array(Y, np.float32))
    print(f"[{k}/{n}] saved {len(I)} samples, rollout success {ok_n/len(seeds):.3f}")


if __name__ == "__main__":
    main()
