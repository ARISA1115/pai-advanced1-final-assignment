"""視覚運動方策の評価。画像と、ロボット自身の状態、タスク（運ぶ順番は、グリッパーの動きから、自動で決める）だけで動かす。"""
import argparse
import json

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from sorting.env import SortEnv
from sorting.visuomotor import Unstick, FrontCam, Planner, load_visuo, proprio_vec
from sorting.vision import FrontLocalizer, make_progress_checker, make_visual_verifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True); ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=10000); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--visual-progress", action="store_true", help="置いた直後に、画像で置き場所にあると分かったら、すぐに次へ進める")
    ap.add_argument("--visual-check", action="store_true", help="順番が終わったあと、画像で置けたかを確認し、置けていなければ、やり直す")
    ap.add_argument("--unstick", action="store_true", help="手先が同じ場所で止まったら、少し動かしてから、方策に戻す")
    ap.add_argument("--strict-planner", action="store_true", help="改良した順番の進行（置き場所の真上で開いたときだけ進める）")
    ap.add_argument("--oracle-planner", action="store_true", help="診断用: 運ぶ順番を、真の情報（置けたかどうか）で進める。順番の進行が、成績の限界かを調べる"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    k, n = map(int, a.shard.split("/"))
    env = SortEnv(0, task="random"); env.fixed_order = True
    pol = load_visuo(a.policy)
    cam = FrontCam(env, pol.res, pol.cams); loc = FrontLocalizer(env) if (a.visual_check or a.visual_progress) else None
    planner = Planner(env, strict=a.strict_planner, verifier=(make_visual_verifier(env, loc) if a.visual_check else None),
                      progress_checker=(make_progress_checker(env, loc) if a.visual_progress else None))
    seeds = list(range(a.seed0, a.seed0 + a.episodes))[k::n]
    ok_n, placed, active, failed = 0, 0, 0, []
    for s in seeds:
        env.reset(s); planner.reset(); getattr(pol, 'reset', lambda: None)(); un = Unstick(env, seed=s) if a.unstick else None
        for _ in range(a.max_steps):
            tgt = planner.update()
            if a.oracle_planner:
                tgt = env.target_index()
            act = pol.act(cam.frame(), proprio_vec(env, tgt))
            _, ok = env.step(un.filter(act) if un else act)
            if ok:
                break
        ok_n += int(env.success()); placed += env.n_placed(); active += env.n_active()
        if not env.success():
            failed.append({"seed": int(s), "placed": int(env.n_placed()), "active": int(env.n_active())})
        print(f"  seed {s} {'成功' if env.success() else '失敗'}（{env.n_placed()}/{env.n_active()}）", flush=True)
    json.dump({"episodes": len(seeds), "success": ok_n / len(seeds), "placed": placed, "active": active, "failed": failed}, open(a.out, "w"), ensure_ascii=False)
    print(f"[{k}/{n}] 3 個とも成功 {ok_n/len(seeds):.3f}、個数ベース {placed/active:.3f}")


if __name__ == "__main__":
    main()
