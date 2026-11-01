"""画像から推定した位置で観測を作り、閉ループで動かして、真値のときと比べる。"""
import argparse
import json

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from sorting.env import SortEnv
from sorting.policy import load_policy
from sorting.vision import Cameras, FrontLocalizer, Tracker, load_cubenet

MAX_STEPS = 800


def run(policy, net, seeds, use_vision, estimator="cnn"):
    env = SortEnv(0, task="random")
    cams = Cameras(env)
    tracker = Tracker(env, FrontLocalizer(env)) if estimator == "color" else None
    ok_n, placed, active, failed = 0, 0, 0, []
    errs = []
    for s in seeds:
        env.reset(s)
        ok = False
        if use_vision and tracker is not None:
            tracker.reset()
        for _ in range(MAX_STEPS):
            if use_vision:
                est = tracker.update() if tracker is not None else net.estimate(cams.frame())
                errs.append(np.abs(est - np.stack([env.cube_pos(i) for i in range(3)])))
                obs = env.obs(est)
            else:
                obs = env.obs()
            _, ok = env.step(policy.act(obs))
            if ok:
                break
        ok_n += int(ok); placed += env.n_placed(); active += env.n_active()
        print(f"  seed {s} {'成功' if ok else '失敗'}（{env.n_placed()}/{env.n_active()}）{'画像' if use_vision else '真値'}", flush=True)
        if not ok:
            failed.append({"seed": int(s), "placed": int(env.n_placed()), "active": int(env.n_active())})
    return ok_n / len(seeds), placed / active, failed, (np.concatenate(errs) if errs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(ROOT / "checkpoints" / "policy_random.pt"))
    ap.add_argument("--cubenet", default=str(ROOT / "checkpoints" / "cubenet.pt"))
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--out", default=str(ROOT / "results" / "vision_eval.json"))
    ap.add_argument("--estimator", default="cnn", choices=["cnn", "color"])
    ap.add_argument("--shard", default="0/1", help="k/n: 種を n 個に分けた k 番目だけ実行（並列実行用）")
    ap.add_argument("--merge", nargs="*", help="分割した結果を 1 つにまとめる（実行はしない）")
    args = ap.parse_args()
    if args.merge:
        parts = [json.load(open(f)) for f in args.merge]
        n = sum(p["episodes"] for p in parts)
        w = lambda k: sum(p[k] * p["episodes"] for p in parts) / n
        errs = np.concatenate([np.load(f.replace(".json", "_errs.npy")) for f in args.merge])
        med = np.median(errs, axis=0) * 1000; p95 = np.percentile(errs, 95, axis=0) * 1000
        res = {"episodes": n, "state_success": w("state_success"), "state_cube_success": w("state_cube_success"),
               "state_failed": sum((p["state_failed"] for p in parts), []),
               "vision_success": w("vision_success"), "vision_cube_success": w("vision_cube_success"),
               "vision_failed": sum((p["vision_failed"] for p in parts), []),
               "median_mm_xy": float(np.mean(med[:2])), "median_mm": med.tolist(), "p95_mm": p95.tolist(),
               "note": "推定誤差は、閉ループ中の全フレーム（アームが隠す場面を含む）での、キューブ 3 個分の絶対誤差"}
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"画像から: {res['vision_success']:.3f}（真値 {res['state_success']:.3f}）、個数ベース {res['vision_cube_success']:.3f}、誤差の中央値 [mm] {np.round(med,1)}")
        return
    k, nsh = map(int, args.shard.split("/"))
    policy = load_policy(args.policy)
    net = load_cubenet(args.cubenet) if args.estimator == "cnn" else None
    seeds = list(range(args.seed0, args.seed0 + args.episodes))[k::nsh]
    s_state, c_state, f_state, _ = run(policy, net, seeds, False, args.estimator)
    print(f"[{k}/{nsh}] 真値の観測: {s_state:.3f} {c_state:.3f}", flush=True)
    s_vis, c_vis, f_vis, errs = run(policy, net, seeds, True, args.estimator)
    res = {"episodes": len(seeds), "state_success": s_state, "state_cube_success": c_state, "state_failed": f_state,
           "vision_success": s_vis, "vision_cube_success": c_vis, "vision_failed": f_vis}
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    np.save(args.out.replace(".json", "_errs.npy"), errs.reshape(-1, 3))
    print(f"[{k}/{nsh}] 画像から: 3 個とも成功 {s_vis:.3f}、個数ベース {c_vis:.3f}", flush=True)


if __name__ == "__main__":
    main()
