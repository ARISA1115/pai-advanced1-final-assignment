"""前方カメラの色の重心による位置推定を校正する（アフィン補正）。キューブの真の位置との差を最小二乗で当てる。"""
import json

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from sorting.env import SortEnv
from sorting.vision import CALIB, FrontLocalizer


def main():
    env = SortEnv(0, task="random")
    loc = FrontLocalizer(env)
    loc.A, loc.b = np.eye(2), np.zeros(2)
    P, T, missing, n = [], [], 0, 0
    for s in range(80000, 80400):
        env.reset(s)
        raw = loc.raw()
        for i in range(3):
            n += 1
            if np.isnan(raw[i]).any():
                missing += 1; continue
            P.append(raw[i]); T.append(env.cube_pos(i)[:2])
    P, T = np.array(P), np.array(T)
    X = np.hstack([P, np.ones((len(P), 1))])
    W, *_ = np.linalg.lstsq(X, T, rcond=None)           # (3, 2)
    A, b = W[:2].T, W[2]
    err_before = np.linalg.norm(P - T, axis=1) * 1000
    err_after = np.linalg.norm(P @ A.T + b - T, axis=1) * 1000
    res = {"A": A.tolist(), "b": b.tolist(), "frames": 400, "cubes": n, "not_found": missing,
           "median_mm_before": float(np.median(err_before)), "median_mm_after": float(np.median(err_after)),
           "p95_mm_after": float(np.percentile(err_after, 95)), "max_mm_after": float(err_after.max())}
    CALIB.parent.mkdir(exist_ok=True)
    json.dump(res, open(CALIB, "w"), indent=1)
    json.dump({k: v for k, v in res.items() if k not in ("A", "b")}, open(ROOT / "results" / "front_calib_eval.json", "w"), indent=1)
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items() if k not in ("A", "b")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
