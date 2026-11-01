"""DAgger: 方策で動かして、到達した状態にエキスパートの正解を付け直し、デモに足して学習し直す。

- 集める種: 20000 以降（学習のデモ 0〜505、最終評価 10000〜10299 と重ならない）
- どの反復を採用するかは、別の検証用の種（30000〜30099）で選ぶ。最終評価の種は選択に使わない。
"""
import argparse
import json
import shutil
import time

import numpy as np
import torch

import _path  # noqa: F401
from _path import ROOT
from evaluate import run
from sorting.env import SortEnv
from sorting.expert import expert_action
from sorting.policy import load_policy
from train import fit, load_demos

MAX_STEPS = 800
VAL_SEEDS = list(range(30000, 30100))
TEST_SEEDS = list(range(10000, 10300))


def rollout(policy, seeds, path, task):
    env = SortEnv(0, task=task)
    X, Y, EP = [], [], []
    ok_n = 0
    for j, s in enumerate(seeds):
        env.reset(s)
        for _ in range(MAX_STEPS):
            X.append(env.obs()); Y.append(expert_action(env)); EP.append(j)
            _, ok = env.step(policy.act(env.obs()))
            if ok:
                break
        ok_n += int(env.success())
    np.savez_compressed(path, obs=np.array(X, np.float32), act=np.array(Y, np.float32), ep=np.array(EP, np.int32))
    return ok_n / len(seeds), len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--task", default="random", choices=["random", "color"])
    ap.add_argument("--demos", default=str(ROOT / "data" / "demos_random.npz"))
    ap.add_argument("--base", default="policy_random_bc.pt")
    ap.add_argument("--prefix", default="random")
    ap.add_argument("--only-test", action="store_true", help="学習は飛ばし、保存済みの候補を、最終評価だけやり直す")
    ap.add_argument("--test-episodes", type=int, default=1000)
    args = ap.parse_args()

    ck = ROOT / "checkpoints"
    if args.only_test:
        log = json.load(open(ROOT / "results" / f"dagger_{args.prefix}.json"))
        for c in log["candidates"]:
            r = run("bc", list(range(10000, 10000 + args.test_episodes)), load_policy(str(ck / c["ckpt"])), args.task)
            c["test_full"] = r["full_success_rate"]; c["test_cube"] = r["cube_success_rate"]; c["test_failed"] = r["failed"]
            print(c["name"], "test", c["test_full"], c["test_cube"], flush=True)
        json.dump(log, open(ROOT / "results" / f"dagger_{args.prefix}.json", "w"), ensure_ascii=False, indent=2)
        return
    base = ck / args.base
    demo_paths = [args.demos]
    log = {"candidates": []}
    policy = load_policy(str(base))
    v = run("bc", VAL_SEEDS, policy, args.task)
    log["candidates"].append({"name": "BC（DAgger なし）", "ckpt": args.base,
                              "val_full": v["full_success_rate"], "val_cube": v["cube_success_rate"]})
    print("BC val", v["full_success_rate"], flush=True)
    seed0 = 20000
    for it in range(1, args.iters + 1):
        seeds = list(range(seed0, seed0 + args.episodes)); seed0 += args.episodes
        path = ROOT / "data" / f"dagger_{args.prefix}_{it}.npz"
        t = time.time()
        rate, n = rollout(policy, seeds, path, args.task)
        print(f"iter {it}: rollout success {rate:.3f}, {n} labeled samples ({time.time()-t:.0f}s)", flush=True)
        demo_paths.append(str(path))
        obs, act, ep = load_demos(demo_paths)
        model, _ = fit(obs, act, ep, epochs=100, seed=0)
        out = ck / f"policy_{args.prefix}_dagger{it}.pt"
        torch.save({"state_dict": model.state_dict()}, out)
        policy = load_policy(str(out))
        v = run("bc", VAL_SEEDS, policy, args.task)
        log["candidates"].append({"name": f"DAgger {it} 回", "ckpt": out.name, "rollout_success": rate,
                                  "labeled_samples": n, "val_full": v["full_success_rate"],
                                  "val_cube": v["cube_success_rate"]})
        print(f"iter {it}: val full {v['full_success_rate']:.3f}", flush=True)

    best = max(log["candidates"], key=lambda c: (c["val_full"], c["val_cube"], -c["ckpt"].count("dagger")))
    log["selected"] = best["ckpt"]
    shutil.copy(ck / best["ckpt"], ck / f"policy_{args.prefix}.pt")
    print("selected", best["ckpt"], flush=True)
    # 最終評価（選択に使っていない種）
    for c in log["candidates"]:
        r = run("bc", list(range(10000, 10000 + args.test_episodes)), load_policy(str(ck / c["ckpt"])), args.task)
        c["test_full"] = r["full_success_rate"]; c["test_cube"] = r["cube_success_rate"]
        c["test_failed"] = r["failed"]
        print(c["name"], "test", c["test_full"], c["test_cube"], flush=True)
    with open(ROOT / "results" / f"dagger_{args.prefix}.json", "w") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
