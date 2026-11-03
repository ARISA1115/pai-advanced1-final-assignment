"""提出動画に、画像だけで動く方策の場面を足す。

場面は、scripts/vm_video_scenes.json の、成功した例（動画の中でも、選んだことを明記する）。全体の成功率は、結果のカードで示す。追加の場面は 2 倍速（字幕に明記）。
"""
import argparse
import json

import imageio.v2 as imageio
import mujoco
import numpy as np

import _path  # noqa: F401
from _path import ROOT
from sorting.env import SortEnv
from sorting.instruction import describe, to_task
from sorting.render import Recorder, banner, card
from sorting.vision import FrontLocalizer, make_progress_checker
from sorting.visuomotor import FrontCam, Planner, Unstick, load_visuo, proprio_vec

W, H, FPS = 1280, 720, 30
TITLE = "日本語の指示で動くピック＆プレース"
MAX_SECONDS = 30            # 1 場面の上限（実時間）。超えたら打ち切り、そのことを字幕で示す。追加の場面は 2 倍速で見せる


def hold(frames, img, seconds):
    frames.extend([img] * int(seconds * FPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True); ap.add_argument("--base", required=True, help="1 回目の動画")
    ap.add_argument("--final", default=str(ROOT / "results" / "vm_final.json"))
    ap.add_argument("--instr", default=None, help="日本語の指示からの評価の結果")
    ap.add_argument("--scenes", type=int, default=3); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {r["id"]: r for r in json.load(open(ROOT / "results" / "instruction_eval.json"))["rows"]}
    scenes = json.load(open(ROOT / "scripts" / "vm_video_scenes.json"))[:a.scenes]   # 成功した例を選んだ場面
    env = SortEnv(0, task="random"); env.fixed_order = True
    pol = load_visuo(a.policy); cam = FrontCam(env, pol.res, pol.cams); loc = FrontLocalizer(env)
    pl = Planner(env, strict=True, progress_checker=make_progress_checker(env, loc))
    rec = Recorder(env, W, H, FPS)
    frames = []
    hold(frames, card(W, H, "追加: 画像だけで動く方策", [
        "ここまでの方策は、キューブの位置（カメラ画像から推定した値）を入力にしていた",
        "ここからの方策は、カメラ画像から、直接、手先の動きを出す（キューブの位置は、入力に含めない）",
        "入力: カメラ画像、ロボット自身の手先・グリッパー、今のタスク（運ぶ色と置き場所）",
        "学習: 模倣学習 + DAgger。運ぶ順番は、置けたかを画像で判定して進める",
        "次の場面は、成功した例を選んで見せている（全体の成功率は、次のカードの数字で、失敗もある）。この部分は 2 倍速",
    ], title_size=50, body_size=28), 9)
    for n, sc in enumerate(scenes, 1):
        r = rows[sc["id"]]; goal, active = to_task(r["output"])
        env.reset(sc["seed"], goal=goal, active=active); pl.reset(); un = Unstick(env, seed=sc["seed"]); rec.reset_clock()
        head = [f"指示: 「{r['text']}」", "解釈: " + describe(goal, active) + " ／ 動作: 画像だけで動く方策（2 倍速）"]
        rec.caption_fn = lambda img, head=head: banner(img, head + [f"置けた数: {env.n_placed()}/{env.n_active()}"], size=28, top=True)
        rec.scene_hook = None
        for _ in range(MAX_SECONDS * 20):
            tgt = pl.update()
            _, ok = env.step(un.filter(pol.act(cam.frame(), proprio_vec(env, tgt))), substep_cb=rec.maybe_frame)
            if ok:
                break
        end = "成功" if env.success() else f"実時間 {MAX_SECONDS} 秒で打ち切り（未達）"
        head.append(f"結果: {end}")
        for _ in range(int(1.5 * FPS)):
            for _ in range(2):
                mujoco.mj_step(env.m, env.d)
            rec.maybe_frame()
        frames.extend(rec.frames[::2]); rec.frames = []
        print(f"場面 {n}: 「{r['text']}」 {env.n_placed()}/{env.n_active()} {end}", flush=True)
    lines = []
    try:
        f = json.load(open(a.final))
        lines.append(f"画像だけで動く方策: 3 個とも成功 {f['success']*100:.1f}%、個数ベース {f['cube_success']*100:.1f}%（学習・設定の選択に使っていない {f['episodes']} エピソード）")
    except FileNotFoundError:
        lines.append("（最終の結果ファイルが、まだない）")
    if a.instr:
        ins = json.load(open(a.instr)); ex = [x for x in ins if x["executed"] > 0]
        lines.append(f"日本語の指示から動作まで: 実行した {len(ex)} 件の平均 {np.mean([x['success'] for x in ex])*100:.1f}%")
    track_line = "位置を推定して動かす方式（87.4%）には、届いていない"
    try:
        co = json.load(open(ROOT / "results" / "vision_cmp_old.json")); cn = json.load(open(ROOT / "results" / "vision_cmp_new.json"))
        track_line = f"位置を推定して動かす方式も、追跡の改良で {cn['success']*100:.1f}%（同じ {cn['episodes']} エピソードで、従来は {co['success']*100:.1f}%）。画像だけの方策は、これに届いていない"
    except FileNotFoundError:
        pass
    lines += ["事前に決めた基準（3 個とも成功 50% 以上）を超えた", track_line, "拡散モデルによる方策も試したが、この時間内では、これを超えなかった"]
    hold(frames, card(W, H, "追加の結果", lines, title_size=54, body_size=28), 10)
    print(f"追加分 {len(frames)/FPS:.1f} 秒", flush=True)
    # 1 回目の動画の、冒頭のタイトルのカード（9 秒）と、結果のカード（最後の 16 秒）を、新しい内容に差し替える。場面の映像は、そのまま
    def jr(name):
        return json.load(open(ROOT / "results" / name))
    title = card(W, H, TITLE, [
        "LLM による解釈と、画像からの模倣学習",
        "Physical AI 応用編1 最終課題",
        "",
        "「赤と青を入れ替えて」のような指示を、ロボットアームが実行する",
        "学習した AI と、画像認識を組み合わせている:",
        "  ① 指示の解釈: ローカルの LLM（Qwen2.5-7B）が、指示を目標に変換",
        "  ② 動作: 模倣学習 + DAgger で学習した方策（ニューラルネット）",
        "  ③ 認識: カメラ画像からキューブの位置を推定（色の領域の重心 + 追跡）。学習した CNN は精度が足りず、置き換えた",
        "シミュレータ: MuJoCo + Franka Emika Panda ／ 等倍速（倍速なし）",
    ], title_size=50, body_size=29)
    e = {r["mode"]: r for r in jr("eval_random.json")}; ie = jr("instruction_eval.json"); e2e = jr("e2e_eval.json"); vis = jr("vision_eval.json"); cnn = jr("vision_eval_before_dagger.json")
    co, cn = jr("vision_cmp_old.json"), jr("vision_cmp_new.json")
    res = card(W, H, "結果", [
        f"動作（目標を変えた課題、{e['bc']['episodes']} エピソード）: 3 個とも成功 エキスパート {e['expert']['full_success_rate']*100:.1f}% ／ 学習した方策 {e['bc']['full_success_rate']*100:.1f}%",
        f"指示の解釈（LLM、{ie['n']} 件の正解つき指示）: 正解率 {ie['accuracy']*100:.0f}%",
        f"指示から動作まで通した成功率: {e2e['e2e_success_overall']*100:.0f}%",
        f"画像から位置を推定して動かした成功率（{vis['episodes']} エピソード）: {vis['vision_success']*100:.1f}%（真値の観測では {vis['state_success']*100:.1f}%）。上の場面は、この版の追跡",
        f"追跡を改良した版: {cn['success']*100:.1f}%（別の {cn['episodes']} エピソードの比較で、従来は {co['success']*100:.1f}%）",
        f"  学習した CNN だけで推定した場合は {cnn['vision_success']*100:.0f}% にとどまり、色の領域の重心 + 追跡に置き換えて改善した",
        "",
        "限界: シミュレータ内、キューブ 3 個・同形。実機では、物体認識と接触の不確かさが加わる",
    ], title_size=54, body_size=27)
    rd = imageio.get_reader(a.base); total = rd.count_frames(); rd = imageio.get_reader(a.base); n = 0
    with imageio.get_writer(a.out, fps=FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=1, quality=7) as w:
        for f in rd:
            w.append_data(title if n < 9 * FPS else res if n >= total - 16 * FPS else f); n += 1
        for f in frames:
            w.append_data(f)
    print(f"全体 {(n+len(frames))/FPS:.1f} 秒 -> {a.out}")


if __name__ == "__main__":
    main()
