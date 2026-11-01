"""比較実験: (1) デモの本数、(2) ノイズ注入あり/なし。評価は学習と分けた乱数の種で行う。"""
import json
import time

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from evaluate import run
from sorting.policy import MLPPolicy
from train import fit, load_demos

EPISODES = 100
SEEDS = list(range(10000, 10000 + EPISODES))


def evaluate(model):
    r = run("bc", SEEDS, model)
    return {"full_success_rate": r["full_success_rate"], "cube_success_rate": r["cube_success_rate"]}


def main():
    obs, act, ep = load_demos([str(ROOT / "data" / "demos.npz")])
    out = {"episodes": EPISODES, "n_demos": [], "noise": []}
    for n in (10, 25, 50, 100, 400):
        m = ep < n
        # 検証用の分割を使うため、n 本のうち最後の 10% は検証に回る（n < 10 のときは全て学習）
        model, _ = fit(obs[m], act[m], ep[m], epochs=100, seed=0)
        r = evaluate(model)
        r["demos"] = n
        out["n_demos"].append(r)
        print("demos", n, r, flush=True)
    clean = ROOT / "data" / "demos_clean.npz"
    if clean.exists():
        o2, a2, e2 = load_demos([str(clean)])
        model, _ = fit(o2, a2, e2, epochs=100, seed=0)
        r = evaluate(model); r["label"] = "ノイズなしのデモ 400 本"
        out["noise"].append(r)
        print("clean demos", r, flush=True)
        r2 = {"label": "ノイズありのデモ 400 本（本番）"}
        model, _ = fit(obs, act, ep, epochs=100, seed=0)
        r2.update(evaluate(model))
        out["noise"].append(r2)
        print("noisy demos", r2, flush=True)
    with open(ROOT / "results" / "ablation.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
