"""説明つきの MP4 を作る。指示文 → 解釈（JSON）→ 実行、実行できない指示の拒否、画像からの位置推定も見せる。"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import _path  # noqa: F401
from _path import ROOT
from sorting.env import SortEnv
from sorting.instruction import InstructionError, describe, to_task
from sorting.policy import load_policy
from sorting.render import Recorder, add_sphere, banner, card
from sorting.vision import FrontLocalizer, Tracker

W, H, FPS = 1280, 720, 30
MAX_STEPS = 800
RGBA = {0: (0.9, 0.15, 0.15, 0.5), 1: (0.15, 0.7, 0.2, 0.5), 2: (0.2, 0.35, 0.95, 0.5)}


def hold(frames, img, seconds):
    frames.extend([img] * int(seconds * FPS))


def jload(name):
    p = ROOT / "results" / name
    return json.load(open(p)) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(ROOT / "checkpoints" / "policy_random.pt"))
    ap.add_argument("--parsed", default=str(ROOT / "results" / "instruction_eval.json"))
    ap.add_argument("--scenes", default=str(ROOT / "scripts" / "video_scenes.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "demo.mp4"))
    args = ap.parse_args()

    policy = load_policy(args.policy)
    rows = {r["id"]: r for r in json.load(open(args.parsed))["rows"]}
    scenes = json.load(open(args.scenes))
    env = SortEnv(0, task="random")
    rec = Recorder(env, W, H, FPS)
    tracker = Tracker(env, FrontLocalizer(env))
    frames: list[np.ndarray] = []

    ev = jload("eval_random.json"); dg = jload("dagger_random.json"); ie = jload("instruction_eval.json"); cnn = jload("vision_eval_before_dagger.json")
    e2e = jload("e2e_eval.json"); vis = jload("vision_eval.json"); ct = jload("cubenet_train.json")

    hold(frames, card(W, H, "日本語で指示できる、学習した AI のロボットアーム", [
        "Physical AI 応用編1 最終課題",
        "",
        "「赤と青を入れ替えて」のような指示を、ロボットアームが実行する",
        "学習した AI と、画像認識を組み合わせている:",
        "  ① 指示の解釈: ローカルの LLM（Qwen2.5-7B）が、指示を目標に変換",
        "  ② 動作: 模倣学習 + DAgger で学習した方策（ニューラルネット）",
        "  ③ 認識: カメラ画像からキューブの位置を推定（色の領域の重心 + 追跡）。学習した CNN は精度が足りず、置き換えた",
        "シミュレータ: MuJoCo + Franka Emika Panda ／ 等倍速（倍速なし）",
    ], title_size=50, body_size=29), 9)
    hold(frames, card(W, H, "仕組み", [
        "日本語の指示 → LLM → JSON → 検査 → 目標（どのキューブを、どの置き場所へ）",
        "   ※ 存在しない色や、同じ置き場所への衝突は、検査で拒否してロボットを動かさない",
        "",
        "カメラ画像 → 色の重心（カメラの幾何で校正）+ 追跡 → キューブの位置 ┐",
        "手先・グリッパー（ロボット自身の関節）  ┴→ 観測 → 方策 → 手先の変位 + 把持 → 20 Hz で制御",
        "",
        "方策の学習: スクリプトの熟練者のデモ 600 本 + DAgger（方策が到達した状態に正解を付け直す）2 回",
    ], title_size=50, body_size=27), 13)

    for n, sc in enumerate(scenes, 1):
        mode = sc.get("mode", "state")
        text = sc["text"] if "text" in sc else rows[sc["id"]]["text"]
        parsed = sc.get("parsed") or rows[sc["id"]]["output"]
        env.reset(sc["seed"])
        rec.reset_clock()
        try:
            goal, active = to_task(parsed)
            refused = None
        except InstructionError as e:
            goal = active = None; refused = str(e)
        head = [f"指示: 「{text}」"]
        if refused is None:
            head.append("解釈: " + describe(goal, active) + ("／ 位置はカメラ画像から推定（白い球が推定位置）" if mode == "vision" else ""))
            env.set_task(goal, active)
            if mode == "vision":
                tracker.reset()
        else:
            head.append("解釈: 実行できない指示 → ロボットは動かさない（" + refused.replace("指示を実行できない: ", "") + "）")

        def cap(img, head=head):
            lines = head + ([f"置けた数: {env.n_placed()}/{env.n_active()}"] if refused is None else [])
            return banner(img, lines, size=28, top=True)
        rec.caption_fn = cap
        est_state = {"est": None}

        def hook(scene):
            if mode == "vision" and est_state["est"] is not None:
                for i in range(3):
                    add_sphere(scene, est_state["est"][i] + np.array([0, 0, 0.035]), (1, 1, 1, 0.9), 0.011)
        rec.scene_hook = hook if mode == "vision" else None

        if refused is not None:
            for _ in range(int(4.5 * FPS)):
                rec.maybe_frame(); rec.next_t = env.d.time        # 静止した場面
                mujoco.mj_step(env.m, env.d); env.d.time += 1.0 / FPS
            frames.extend(rec.frames); rec.frames = []
            continue
        for _ in range(MAX_STEPS):
            if mode == "vision":
                est = tracker.update(); est_state["est"] = est
                obs = env.obs(est)
            else:
                obs = env.obs()
            _, ok = env.step(policy.act(obs), substep_cb=rec.maybe_frame)
            if ok:
                break
        for _ in range(int(1.5 * FPS)):
            for _ in range(2):
                mujoco.mj_step(env.m, env.d)
            rec.maybe_frame()
        frames.extend(rec.frames); rec.frames = []
        print(f"場面 {n}: 「{text}」 モード={mode} 結果 {env.n_placed()}/{env.n_active()} {'成功' if env.success() else '未達'}")

    lines = []
    if ev:
        e = {r["mode"]: r for r in ev}
        lines += [f"動作（目標を変えた課題、{e['bc']['episodes']} エピソード）: 3 個とも成功 エキスパート {e['expert']['full_success_rate']*100:.1f}% ／ 学習した方策 {e['bc']['full_success_rate']*100:.1f}%"]
    if dg:
        c = dg["candidates"]
        lines += [f"  DAgger なし {c[0]['test_full']*100:.1f}% → DAgger 後 {max(x['test_full'] for x in c[1:])*100:.1f}%（失敗件数の差を検定）"]
    if ie:
        lines += [f"指示の解釈（LLM、{ie['n']} 件の正解つき指示）: 正解率 {ie['accuracy']*100:.0f}%"]
    if e2e:
        lines += [f"指示から動作まで通した成功率: {e2e['e2e_success_overall']*100:.0f}%"]
    if vis:
        lines += [f"画像から位置を推定して動かした成功率（{vis['episodes']} エピソード）: {vis['vision_success']*100:.1f}%（真値の観測では {vis['state_success']*100:.1f}%）。推定誤差の中央値は x {vis['median_mm'][0]:.1f} mm、y {vis['median_mm'][1]:.1f} mm"]
        if cnn:
            lines += [f"  学習した CNN だけで推定した場合は {cnn['vision_success']*100:.0f}% にとどまり、色の領域の重心 + 追跡に置き換えて改善した"]
    lines += ["", "限界: シミュレータ内、キューブ 3 個・同形。実機では、物体認識と接触の不確かさが加わる"]
    hold(frames, card(W, H, "結果", lines, title_size=54, body_size=28), 16)

    with imageio.get_writer(args.out, fps=FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=1, quality=7) as w:
        for f in frames:
            w.append_data(f)
    print(f"frames={len(frames)} duration={len(frames)/FPS:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
