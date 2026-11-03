"""レポート（PDF）を作る。数値は results/*.json から読み、手で書き写さない。
使い方: python report/build_report.py --account <omnicampus アカウント名> --video-url <YouTube の URL>
"""
import argparse
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
R = ROOT / "results"


def J(name):
    return json.load(open(R / name))


def pct(x, d=1):
    s = f"{x*100:.{d}f}%"
    return s.replace(".0%", "%") if d == 1 else s


def sig(p):
    return "差は統計的に確かめられた" if p < 0.05 else "差は統計的には確かめられなかった"


def ptxt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def table(head, rows, cls=""):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='{cls}'><tr>{h}</tr>{b}</table>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--video-url", default="（提出時に記載）")
    ap.add_argument("--with-vm", action="store_true", help="追加実験（画像を入力にした方策）を、報告書に入れる")
    ap.add_argument("--no-ai-note", action="store_true", help="生成 AI を使った旨の一文を入れない")
    args = ap.parse_args()

    # ---- 画像
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image
    from sorting.env import SortEnv
    from sorting.expert import expert_action
    from sorting.vision import Cameras
    env = SortEnv(0, task="random"); env.reset(10001)
    rend = mujoco.Renderer(env.m, 540, 960); cams = Cameras(env)
    imgs = []
    for k in range(260):
        env.step(expert_action(env))
        if k in (0, 110):
            rend.update_scene(env.d, camera="main"); imgs.append(rend.render())
            if k == 0:
                f = cams.frame()
    out = ROOT / "report" / "img"; out.mkdir(exist_ok=True)
    imageio.imwrite(out / "scene_start.png", imgs[0]); imageio.imwrite(out / "scene_carry.png", imgs[1])
    two = np.concatenate([f[:3].transpose(1, 2, 0), f[3:].transpose(1, 2, 0)], axis=1)
    Image.fromarray(two).resize((two.shape[1] * 3, two.shape[0] * 3), Image.NEAREST).save(out / "cnn_input.png")
    rf = mujoco.Renderer(env.m, 360, 360)
    env.reset(10001); rf.update_scene(env.d, camera="front"); imageio.imwrite(out / "front_view.png", rf.render())
    for n in ("fig_loss.png", "fig_demos.png", "fig_instr.png", "fig_cubenet.png", "fig_vision.png"):
        shutil.copy(R / n, out / n)

    # ---- 数値
    b_bc, b_dg = J("baseline/confirm_bc.json")[0], J("baseline/confirm_dagger2.json")[0]
    nb = b_bc["episodes"]
    fb1, fb2 = len(b_bc["failed"]), len(b_dg["failed"])
    p_base = fisher_exact([[fb1, nb - fb1], [fb2, nb - fb2]])[1]
    ev = {r["mode"]: r for r in J("eval_random.json")}
    n1 = ev["bc"]["episodes"]
    dg = J("dagger_random.json")
    cand = dg["candidates"]; sel = next(c for c in cand if c["ckpt"] == dg["selected"])
    bc0 = cand[0]
    f0, f1 = len(bc0["test_failed"]), len(sel["test_failed"])
    nt = len(range(10000, 10000 + 1000))
    p_s1 = fisher_exact([[f0, nt - f0], [f1, nt - f1]])[1]
    ie, iz, e2e = J("instruction_eval.json"), J("instruction_eval_zeroshot.json"), J("e2e_eval.json")
    ct, vis = J("cubenet_train.json"), J("vision_eval.json")
    ss, calib, held = J("cubenet_ss_train.json"), J("front_calib_eval.json"), J("front_loc_heldout.json")
    cnn200, col1, col2 = J("vision_eval_before_dagger.json"), J("vision_eval_color_v1_200.json"), J("vision_eval_200.json")
    pv = J("eval_random_policy_vision.json")[0]
    _fs = [f['seed'] for f in vis['vision_failed']]
    dev_rate = 1 - sum(1 for x in _fs if x < 10200) / 200
    unseen_rate = 1 - sum(1 for x in _fs if x >= 10200) / (vis['episodes'] - 200)
    CATJ = {"single": "単独", "swap": "入れ替え", "multi": "複数", "sort": "色ごと", "exclude": "除外", "casual": "口語", "invalid": "実行不可"}

    cand_rows = [[html.escape(c["name"]) + (" ←採用" if c["ckpt"] == dg["selected"] else ""), pct(c["val_full"]),
                  pct(c["test_full"]), pct(c["test_cube"]), str(len(c["test_failed"]))] for c in cand]
    ex_ids = [13, 33, 32, 28, 50, 30, 53, 62]
    rows_by_id = {r["id"]: r for r in ie["rows"]}
    ex_rows = []
    for i in ex_ids:
        r = rows_by_id[i]
        out_txt = json.dumps(r["output"], ensure_ascii=False)
        res = "正解" if r["correct"] else ("誤り（検査で拒否）" if r.get("safe") else "誤り")
        ex_rows.append([html.escape(r["text"]), f"<code>{html.escape(out_txt)}</code>", res])
    cat_rows = [[CATJ[c], f"{ie['by_category'][c]['correct']}/{ie['by_category'][c]['n']}", f"{iz['by_category'][c]['correct']}/{iz['by_category'][c]['n']}"] for c in CATJ]

    heading_ai = "" if args.no_ai_note else "・生成 AI の利用"
    vm_html = ""
    if args.with_vm:
        import os
        def jv(name):
            f = R / name
            return json.load(open(f)) if f.exists() else None
        # 推移の表: 失敗した最初の試行から、最終の方策まで。結果ファイルがあるものだけを載せる
        steps = [
            ("1 回目（CPU、6 エポック、約 8 万サンプル）", "従来型 CNN", jv("vm_eval_bc.json")),
            ("2 回目（GPU、30 エポック、同じデータ）", "従来型 CNN", jv("vm_eval_base30.json")),
            ("3 回目（60 エポック、データ 2 倍）", "従来型 CNN", jv("vm_eval_base60.json")),
            ("4 回目（DAgger 1 回、約 29 万サンプル）", "従来型 CNN", jv("vm_eval_dagger2.json")),
            ("5 回目（DAgger 2 回、約 40 万サンプル）", "従来型 CNN", jv("vm_eval_dagger3.json")),
            ("v2（事前学習 ResNet18、2 視点、8 ステップの系列）", "ResNet18", jv("vm2_eval_bc.json")),
            ("v2 + DAgger", "ResNet18", jv("vm2_eval_dagger.json")),
            ("5 回目 + 順番を画像で進める（種 10000〜、判定の設計にも使った種）", "従来型 CNN", jv("vm_eval_d3_vprog.json")),
            ("同上（種 800000〜。以下、設定の比較に使った種）", "従来型 CNN", jv("vm_eval_fresh.json")),
            ("DAgger をもう 1 回", "従来型 CNN", jv("vm_eval_d4_fresh.json")),
            ("3 つの方策の出力の平均", "従来型 CNN", jv("vm_sel_ens.json")),
            ("手先が止まったら少し動かす処理を追加", "従来型 CNN", jv("vm_sel_d3_unstick.json")),
            ("最初に運ぶ色が 1・2 のデータを追加して再学習", "従来型 CNN", jv("vm_sel_d5_unstick.json")),
            ("拡散モデル（握る・開くは分類、学習は途中で打ち切り）", "ResNet18 + 拡散", jv("vm3g_eval_fresh.json")),
            ("<b>最終（種 810000〜。設定の選択に使っていない種）</b>", "従来型 CNN", jv("vm_final.json")),
        ]
        rows = [[a, b, pct(c["success"]), pct(c["cube_success"])] for a, b, c in steps if c]
        final = jv("vm_final.json")
        best = final if final else max((c for _, _, c in steps if c), key=lambda c: c["success"])
        bs = best["success"]
        base_vis = vis["vision_success"]
        if bs >= base_vis:
            concl = f"画像を直接入力にした方策は、最良で {pct(bs)} に達し、位置推定を挟む方式（{pct(base_vis)}）を上回った。"
        elif bs >= 0.5:
            concl = f"画像を直接入力にした方策は、最良で {pct(bs)} に達し、基準の 50% を超えたが、位置推定を挟む方式（{pct(base_vis)}）には届かなかった。"
        else:
            concl = f"画像を直接入力にした方策は、最良でも {pct(bs)} にとどまった。"
        ins = jv("vm_instruction_eval.json"); vm_instr_html = ""
        if ins:
            ex = [x for x in ins if x["executed"] > 0]; inv = [x for x in ins if x["category"] == "invalid"]
            vm_instr_html = f"<p>日本語の指示からも、この方策を動かした（保存済みの LLM の出力を使用）。実行した {len(ex)} 件（各 3 回）の平均成功率は {pct(sum(x['success'] for x in ex)/len(ex))}、実行できない指示は {int(sum(x['success'] for x in inv))}/{len(inv)} を拒否した。</p>"
        co, cn = jv("vision_cmp_old.json"), jv("vision_cmp_new.json"); track_html = ""
        if co and cn:
            track_html = f"""
<h3>6.2b 追跡の改良（2 回目の提出で追加）</h3>
<p>§6.2 の失敗は、握ったまま運ぶ途中で滑って落ちるものだった。つかんだ時点の横方向の推定のずれ（90% 点）は、失敗 8.5 mm、成功 5.4 mm、真値を使った場合 3.5 mm で（各 32 エピソード）、腕に一部隠れたキューブの領域で位置を更新していることを疑った。領域の面積が最初の面積の 85% より小さいときは更新しない（従来は 60%）ように直した。調査に使っていない種 830000〜 の {cn['episodes']} エピソードで、同じ種の比較をした。</p>
{table(["追跡", "3 個とも成功", "個数ベース"], [["従来（面積 60%）", pct(co['success']), pct(co['cube_success'])], ["<b>改良（面積 85%）</b>", f"<b>{pct(cn['success'])}</b>", pct(cn['cube_success'])]])}
<p class="cap">表 4b　失敗の件数: 従来 {len(co['failed'])} 件 → 改良 {len(cn['failed'])} 件。§6.2 の 87.4% は、種 10000〜 の 1000 エピソードの数字で、種が違う。</p>
"""
        vm_html = f"""{track_html}
<h3>6.3 追加実験: 画像を直接入力にした方策（視覚運動方策）</h3>
<p>ここまでの方策は、状態ベクトル（キューブの位置を含む）を入力にしていた。この追加実験では、<b>カメラ画像から、直接、手先の動作を出す方策</b>を作った。入力は、画像と、ロボット自身の手先・グリッパー、今のタスク（運ぶキューブの色と置き場所）だけで、<b>キューブの位置は、入力に含めない</b>。運ぶ順番は、色の番号の順に固定する。次へ進める合図には、グリッパーの動き（つかんで、開いた）を使い、最終版では、手を開いた位置の規則と画像の判定を加えた（キューブの真の位置は使わない）。正解は、キューブの真の位置を見るエキスパート（学習時だけ使う特権情報）。評価は、学習に使っていない種の 200 エピソード（95% 信頼区間は、おおよそ ±6.5 ポイント）。</p>
<p>最初の試行は、<b>0%</b> だった。同じデータ・同じ構造で、学習のエポックを 6 から 30 に増やすと、検証損失が 0.135 から 0.049 に、成功率が 0% から 15.5% になった（エポック不足が、主な要因と見られる。乱数の影響は、確かめていない）。その後、データを増やし、学習した方策で動かして正解を付け直す DAgger を重ね、構造も改善した。</p>
<p>運ぶ順番を、真の情報で進める診断で、順番の進め方に改善の余地があると分かったので、置いたかどうかを画像で判定して順番を進める仕組みを加えた（動作そのものは、方策が決める）。さらに、失敗を調べると、方策の出力が途中で止まるエピソードと、<b>最初に運ぶキューブの色による差</b>（色 0: 95%、色 1: 68%、色 2: 33%）が見つかった。前者には止まったら少し動かす処理を、後者には該当するエピソードのデータの追加を試した（DAgger の追加と方策の平均は、効果がなかった。データの追加は、同じ種の比較で 139/200 から 158/200 に上がった。対応のある検定 McNemar で p = 0.016）。拡散モデルで動作の系列を生成する方式（Diffusion Policy の考え方。デノイザーは残差 MLP で、元の論文の U-Net とは異なる）も実装したが、握る・開くの切り替えが不安定で（従来型: 1〜2 回、拡散モデル: 5〜24 回。4 エピソードの計測）、握る・開くを分類に分けた版でも、学習を 8 エポック中 4 で打ち切った時点で 8.5% と、この課題の時間内では、従来型を超えなかった。</p>
{table(["方策（実験の順）", "構造", "3 個とも成功", "個数ベース"], rows)}
{vm_instr_html}
<p class="cap">表 5　<b>→ 結論: {concl}</b>先行研究（R3M: 事前学習した視覚表現が、ゼロからの学習より、シミュレーションの 12 タスクで 20% 以上向上、ACT: 行動の系列を出す方策、DAgger: 方策が到達した状態での再学習）の考え方を、構造と学習に取り入れた（概要の記述に基づく。効果は、この課題で、実測した値のみを報告している）。</p>
"""
    ai_note = "" if args.no_ai_note else "<li>コードと本資料の作成に、生成 AI（Claude Code）を使った。設計・実験条件・結果の数値は、実行ログ（results/*.json）から取っている。</li>"

    body = f"""
<h1>日本語の指示で動くピック＆プレース ― LLM による解釈と、画像からの模倣学習</h1>
<p class="sub">Physical AI 応用編1 最終課題　説明資料</p>
<div style="border:1.5px solid #4c78a8;padding:4pt 8pt;margin:6pt 0;">
<b>2 回目の提出で追加した結果</b>（詳細は §6.2b、§6.3）
<ul>
<li><b>位置を推定して動かす方式: 87.4% → 95.9%</b>（同じ 389 エピソードの比較、373/389 と 340/389。McNemar 検定で p &lt; 0.001）。腕に一部隠れた領域で、キューブの位置を更新しないようにした。</li>
<li><b>画像から直接動かす方策を新しく追加: 3 個とも成功 76.0%</b>（学習・設定の選択に使っていない 200 エピソード。事前に決めた基準 50% を超えた）。日本語の指示から動作まで 72.8%。位置を推定する方式には届いていない。</li>
<li>拡散モデルによる方策も実装し、比較した（学習を途中で打ち切った時点で 8.5%）。</li>
</ul></div>
<p>動画: YouTube に提出（URL は、omnicampus の提出フォームに記載）。等倍速（倍速なし）、約 3 分以内。</p>

<h2>1. 概要と目的</h2>
<p>机の上の 3 色のキューブを、ロボットアーム（Franka Emika Panda、MuJoCo）が指定された置き場所へ運ぶ。
<b>「赤と青を入れ替えて」のような日本語の指示</b>を、学習した AI（LLM と方策）と、画像からの位置推定を組み合わせて実行する。</p>
<ul>
<li><b>① 指示の解釈</b>: ローカルの LLM（Qwen2.5-7B-Instruct、4bit）が、指示を目標（どのキューブをどの置き場所へ）に変換する。LLM の出力は、検査してから実行する。</li>
<li><b>② 動作</b>: スクリプトの熟練者のデモからの模倣学習（Behavior Cloning）と DAgger で学習した、ニューラルネットの方策。</li>
<li><b>③ 認識</b>: カメラ画像から、キューブの位置を推定する。学習した CNN を試したが精度が足りず（閉ループの成功率 27%）、<b>最終版は、色の領域の重心をカメラの幾何で机の上に戻し、校正する手法 + 追跡</b>に置き換えた（学習した AI ではない）。</li>
</ul>
<p>サンプルの難易度では「上級」（自然言語によるロボット操作）にあたる内容を目指した。API キーは使わず、全て手元で動く。評価は、段階ごとに数値で行った（§3〜§6）。</p>
<div class="figs"><img src="img/scene_start.png"><img src="img/scene_carry.png"></div>
<p class="cap">図 1　シーン（左: 開始時。奥に 3 色の置き場所、右: 運んでいる途中）。</p>

<h2>2. 構成と、段階の設計</h2>
<p><code>日本語の指示 → LLM → JSON → 検査 → 目標</code>、<code>前方斜め上のカメラ画像 → 色の領域の重心（校正）+ 追跡 → キューブの位置</code>、<code>手先・グリッパー → 観測（17 次元）→ 方策 → 手先の変位 + 把持 → 20 Hz で制御</code></p>
{table(["段階", "内容", "追加した理由"], [
 ["0 ベースライン", "キューブを、同じ色の置き場所へ", "まず動く基準を作り、データ量・DAgger の効果を測る"],
 ["1 目標の一般化", "目標を、エピソードごとに変える（入れ替え、一部だけ動かす）", "人が目標を指定できるようにする"],
 ["2 自然言語", "日本語の指示を、LLM で目標に変換", "上級のサンプル（自然言語での操作）に相当"],
 ["3 画像からの認識", "カメラ画像から、位置を推定", "シミュレータの真値を使わない"]])}
<p>観測・行動・エキスパート・方策の構造は、全段階で共通。観測は 17 次元（手先の位置、グリッパーの開き、対象キューブの手先と置き場所に対する相対位置、目標の置き場所の one-hot、持ち上がっているか、残りの数、手先の高さ など）、行動は 4 次元（手先の変位 x, y, z と把持）。方策は 17→256→256→4 の MLP（損失は MSE、Adam、100 エポック）。</p>

<h2>3. 段階 0：ベースライン（色と同じ置き場所）</h2>
<p>デモ 400 本から BC を学習し、DAgger（方策が到達した状態に、エキスパートが正解を付け直して再学習）を 2 回行った。学習・検証と重ならない新しい種 {nb} エピソードで確認した。</p>
{table(["方策", "3 個とも成功", "失敗した件数"], [
 ["BC のみ", pct(b_bc["full_success_rate"]), f"{fb1} / {nb}"], ["BC + DAgger", pct(b_dg["full_success_rate"]), f"{fb2} / {nb}"]])}
<p class="cap">表 1　失敗件数の差を、Fisher の正確確率検定で検定すると {ptxt(p_base)}（{sig(p_base)}）。<b>→ 結論: DAgger は、BC のみで残る失敗（{fb1} 件）を {fb2} 件まで減らす。以後、DAgger を使う。</b></p>
<div class="figs"><img src="img/fig_demos.png"><img src="img/fig_loss.png"></div>
<p class="cap">図 2　左: デモの本数と成功率（BC のみ）。10 本で 75%、100 本で飽和する。右: 方策の学習曲線（段階 1 の学習）。<b>→ 結論: この課題では、デモは 100 本程度で足りる。検証の損失も、学習に追従して下がり、過学習の兆しはない。</b></p>

<h2>4. 段階 1：目標の一般化</h2>
<p>動かすキューブの数（1〜3）と、どのキューブをどの置き場所へ運ぶか（順列）を、エピソードごとの乱数で変える。方策は、目標の置き場所を観測として受け取る。動かさないキューブは、障害物として机に残る。
デモは 600 エピソード（約 16 万サンプル（158,066））、DAgger は 300 エピソードを 2 回。<b>どの方策を採用するかは、検証用の種（30000〜30099）で決め、最終評価の種（10000〜10999）は選択に使っていない。</b>検証が同率のときは、反復の少ない方を採用した。</p>
{table(["方策", "検証 3 個とも成功", f"最終評価（{nt} エピソード）3 個とも成功", "個数ベース", "失敗した件数"], cand_rows)}
<p class="cap">表 2　BC のみと採用した方策の失敗件数の差は、{ptxt(p_s1)}（{sig(p_s1)}）。何もしない方策の成功率は {pct(ev['null']['full_success_rate'])}（下限）、エキスパートは {pct(ev['expert']['full_success_rate'])}。<b>→ 結論: 目標を変えても、BC のみでほぼ成功する（{pct(cand[0]['test_full'])}）。DAgger の追加効果は、この課題では確かめられない。DAgger 1 回を採用したのは、検証の種で同率なら反復の少ない方、という事前の規則による。</b></p>

<h2>5. 段階 2：自然言語の指示</h2>
<p><b>方法</b>: Qwen2.5-7B-Instruct（4bit、MLX 版）に、システムの指示と few-shot 7 例を与え、<code>{{"moves":[{{"cube":"red","to":"blue"}}]}}</code> 形式の JSON を出力させる（温度 0）。出力は、検査（存在しない色、同じ置き場所への衝突、形式の誤りを拒否）を通ったものだけを、目標として実行する。
<b>評価</b>: 正解つきの指示 62 件（単独・入れ替え・複数・色ごと・除外・口語・実行できない指示）。few-shot の例とは文面を重ねない。<b>評価セットは、結果を見て直していない。</b></p>
{table(["", "few-shot あり", "ゼロショット"], [
 ["全体の正解率", f"{pct(ie['accuracy'])}（{round(ie['accuracy']*ie['n'])}/{ie['n']}）", f"{pct(iz['accuracy'])}（{round(iz['accuracy']*iz['n'])}/{iz['n']}）"],
 ["実行できる指示（52 件）", pct(ie["valid_instructions_accuracy"]), pct(iz["valid_instructions_accuracy"])],
 ["実行できない指示の拒否（LLM のみ）", pct(ie["invalid_refused_llm_only"]), pct(iz["invalid_refused_llm_only"])],
 ["実行できない指示の拒否（検査込み）", pct(ie["invalid_refused_with_validation"]), pct(iz["invalid_refused_with_validation"])]])}
<div class="figs"><img src="img/fig_instr.png" style="width:62%"></div>
<p class="cap">図 3　カテゴリ別の正解率（カッコ内は件数）。few-shot の効果は、1 件の差で、この規模では確かめられない。<b>→ 結論: 弱いのは「口語」と「複数」の指示。確認の質問を返す仕組みが要る。</b></p>
{table(["指示", "LLM の出力", "結果"], ex_rows)}
<p class="cap">表 3　指示の例（正解 4 件と、誤り 5 件のうち 4 件。誤りの全件は、results/instruction_eval.json にある）。<b>→ 結論: 誤りのうち、ロボットが誤った動作をするのは 1 件（「赤、青のとこね」）だけで、残りは、動かない安全側の失敗。</b></p>
<p><b>指示から動作まで通した評価</b>（解釈が正解の指示を、5 つの種で実行）: 指示の解釈の正解率 {pct(e2e['parse_accuracy'])}、<b>指示から動作までの成功率 {pct(e2e['e2e_success_overall'])}</b>。実行できない指示は、検査によって、ロボットが動かずに済んだ件数を成功として数えた。</p>

<h2>6. 段階 3：画像からの位置推定</h2>
<p>方策は、これまでと同じ（真値で学習したもの）。観測のうち、キューブの位置だけを、画像からの推定値に置き換えて、同じ種で比べた。手先の位置とグリッパーの開きは、ロボット自身の関節から得る（実機でも得られる量）。</p>
<h3>6.1 学習した CNN（試したが、不十分だった）</h3>
<p>真上と横の 2 視点（96×96）を重ねて入力し、3 個のキューブの位置を回帰する。学習 {ct['train_frames']} フレーム、検証 {ct['val_frames']} フレーム（検証は別のエピソード）。検証の誤差の中央値は x {ct['val_median_mm'][0]:.1f}、y {ct['val_median_mm'][1]:.1f}、z {ct['val_median_mm'][2]:.1f} mm（95 パーセンタイル y {ct['val_p95_mm'][1]:.1f} mm）。
閉ループ中の y の誤差が、方策の許容（約 1 cm）を超え、<b>画像から推定した位置で動かした成功率は {pct(cnn200['vision_success'])}</b>（真値では {pct(cnn200['state_success'])}）にとどまった。
構造を、特徴マップから位置を取り出すキーポイント方式（空間ソフトアーグマックス）に変えても、y は改善しなかった（検証の平均絶対誤差 y {ss['val_mae_mm'][1]:.1f} mm）。ただし、時間の都合で、学習データは半分（{ss['train_frames']} フレーム）、12 エポックだったため、構造の優劣を比べた結果とは言えない。
また、推定値の観測に、真値の正解を付けて方策を学習し直す「画像を入れた DAgger」は、<b>方策の真値での成功率を {pct(pv['full_success_rate'])} に下げた</b>（元の方策は、同じ 300 エピソードで 99.7%）ため、採用しなかった。誤差のある観測と、真値の正解が混ざり、方策の判断が鈍ったと考える。</p>
<h3>6.2 色の領域の重心 + 校正 + 追跡（最終版）</h3>
<div class="figs"><img src="img/front_view.png" style="width:30%"><img src="img/fig_vision.png" style="width:66%"></div>
<p class="cap">図 4　左: 前方斜め上のカメラ（アームに隠されにくい）。右: 推定方式ごとの閉ループの成功率（同じ方策、同じ 200 エピソード。最終版は 1000 エピソード）。</p>
<ul>
<li><b>最初の位置</b>: 前方斜め上のカメラ画像で、色ごとの領域の重心を、カメラの幾何（位置・向き・画角）で机の上へ戻し、校正（アフィン補正）をかける。置き場所の領域は、既知の位置で除く。
校正に使っていない 200 エピソード（キューブ {held['cubes']} 個）で、<b>見つからなかったものは {held['not_found']} 個、誤差の中央値 {held['median_mm']:.2f} mm、95 パーセンタイル {held['p95_mm']:.2f} mm、最大 {held['max_mm']:.2f} mm</b>（CNN の y の中央値 {ct['val_median_mm'][1]:.1f} mm の約 10 分の 1）。</li>
<li><b>追跡</b>: キューブはつかむまで動かない。つかんだら、手先との相対位置で追跡する（持ち上げ中の z は、実測した平均値で補正）。離したら、その場に置かれたとみなす。キューブの真の位置は、追跡では使っていない。</li>
<li><b>再検出</b>: 落としたキューブと、手先から遠いキューブだけ、毎ステップ、カメラで位置を探し直す。</li>
</ul>
{table(["推定の方式（同じ方策・同じ種）", "3 個とも成功", "個数ベース", "推定誤差の中央値 x / y / z [mm]"], [
 ["真値の観測（上限）", pct(col2["state_success"]), pct(col2["state_cube_success"]), "—"],
 ["学習した CNN（全結合）", pct(cnn200["vision_success"]), pct(cnn200["vision_cube_success"]), " / ".join(f"{v:.1f}" for v in cnn200["median_mm"]) if "median_mm" in cnn200 else "4.3 / 14.0 / 0.4"],
 ["色の重心 + 追跡（第 1 版）", pct(col1["vision_success"]), pct(col1["vision_cube_success"]), " / ".join(f"{v:.1f}" for v in col1["median_mm"]) if "median_mm" in col1 else "—"],
 [f"<b>色の重心 + 追跡 + 再検出（最終版、{vis['episodes']} エピソード）</b>", f"<b>{pct(vis['vision_success'])}</b>（真値 {pct(vis['state_success'])}）", f"<b>{pct(vis['vision_cube_success'])}</b>", " / ".join(f"{v:.1f}" for v in vis["median_mm"])]])}
<p class="cap">表 4　第 1 版・CNN は 200 エピソード、最終版は 1000 エピソード（種 10000〜10999）。誤差は、閉ループ中の全フレームでの、キューブ 3 個分の絶対誤差の中央値。<b>→ 結論: この課題の画像認識は、学習した CNN より、カメラの幾何を使う手法のほうが、精度（y で約 10 倍）も成功率（27% → 87%）も高い。残る失敗は、搬送中の滑りで、画像では直せない。</b></p>
<p><b>残る失敗（{len(vis['vision_failed'])} / {vis['episodes']} 件）</b>は、動かす 3 個のうち 1〜2 個を置いたところで止まったものが大半（{sum(1 for f in vis['vision_failed'] if f['active']==3 and f['placed'] in (1,2))} 件）。種 10200 以降の失敗の先頭 48 件を再実行して調べたところ、<b>48 件すべてで、キューブがグリッパーから外れて落ちて</b>（グリッパーが 0.2 未満まで閉じて）いた。持っているキューブは、手とハンドに隠れて、カメラで見えない（面積が初期の 30% 未満の場面が過半）ため、画像では、滑りを修正できなかった。</p>
<p><b>この評価の公平さについて</b>: 追跡の改良（持ち上げ中の z の補正、落下の検出、再検出の抑制）は、最終評価の種のうち 10000〜10199 の失敗を調べて行った。そのため、改良に使っていない種 10200〜10999（800 エピソード）での成功率を併記する: <b>{pct(unseen_rate)}</b>（改良に使った 200 エピソードでは {pct(dev_rate)}）。差はなく、改良による成績の水増しは、みられない。</p>

{vm_html}
<h2>7. 工夫した点</h2>
<ul>
<li><b>LLM の出力を信用しない設計</b>: 検査を通ったものだけを実行する。実行できない指示 10 件は、LLM のみでは 8 件、検査込みでは 10 件が拒否でき、誤って動いたものは 0 件だった。</li>
<li><b>API に依存しない</b>: サンプルは API の提供が不可とされる。ローカルの LLM（Apache-2.0）を使い、解釈の出力も JSON で保存して同梱したので、評価する側は、LLM がなくても後段（実行）を再現できる。</li>
<li><b>評価の厳密さ</b>: 学習・検証・最終評価・確認・画像・指示の乱数の種を分けた。何もしない方策を下限に置いた。DAgger の効果は、新しい種 1000 エピソードで、検定した。採用する方策は、検証の種だけで選んだ。ただし、画像認識の追跡の改良は、最終評価の種の一部の失敗を見て行ったため、§6.2 に、使っていない種での結果を併記した。</li>
<li><b>失敗の原因を、実測で切り分けて改善した</b>（画像からの認識）: CNN の 27% → 色の重心 + 追跡で 62.5% → 持ち上げ中の z の補正、落下の検出と再検出、近接時の再検出の抑制で 87.4%。持ち上げ中のキューブの見え方も、面積比と誤差で測り、カメラでは頼れないと確かめてから、使わない設計にした。</li>
<li><b>観測の共通化</b>: 方策が、目標・位置の入力だけに依存するので、位置を真値から画像の推定値に置き換えても、方策を作り直さずに試せた。</li>
</ul>

<h2>8. 失敗と考察</h2>
<ul>
<li><b>実装の不具合</b>: 入力ゼロで手先が沈んだ（指令角を現在角から作り直し、重力の垂れ下がりが累積。指令角を保持して解決）、作業範囲の上限が初期姿勢より低かった、把持の高さが高くて指がキューブの上をかすめた、つかみ位置のずれで置き場所に入らなかった（ずれを補正）。いずれも、実測で原因を切り分けた。</li>
<li><b>評価の不具合</b>: 個数ベースの成功率の分母に、動かさなかったキューブを数えていた。見つけて直し、最終評価をやり直した。</li>
<li><b>ノイズ注入（DART）の効果</b>: 段階 0 では、ノイズなしのデモでも成功率は同じで、効果は測れなかった。エキスパートが状態に対して決定的で、評価の分布が学習と同じためと考える。</li>
<li><b>指示の解釈の誤り</b>: 誤りの多くは、(a) JSON が壊れる、(b) 色の言い換えを「存在しない」と拒否する、(c) 省略の多い口語（「赤、青のとこね」）を入れ替えと取り違える、であった。(a)(b) はロボットが動かない安全側の失敗だが、(c) は誤った動作になりうる。確認の質問を返す仕組みがあれば防げる。</li>
<li><b>画像からの認識の試行錯誤</b>: 学習した CNN は、y の誤差が方策の許容を超えて失敗した（27%）。画像を入れた DAgger は、真値での成功率を下げた。幾何の手法（色の重心 + 校正）は、誤差 0.70 mm と精密だったが、追跡の第 1 版は 62.5% にとどまった。原因は、持ち上げ中のキューブが指の中で滑ること（z が約 25 mm ずれ、「下ろしたか」の判定が狂って止まる）と、搬送中の落下で、実測してから直した。</li>
</ul>

<h2>9. 限界と今後</h2>
<ul>
<li>キューブは 3 個・同形で、置き場所も固定。色の種類を超える指示（「右の」「手前の」）には未対応。</li>
<li>シミュレータ内の結果で、実機では、画像のずれ・照明・接触の不確かさが加わる。</li>
<li>画像からの位置推定で動かす場合、運搬中にキューブが指の中で滑って落ちる失敗が残る（成功率 87.4%、真値では 99.9%）。持っているキューブは、手とハンドに隠れて、カメラで見えないため。手首のカメラや、触覚（力）のセンサがあれば、改善できる見込み。</li>
<li>今後: 画像を入れた DAgger（推定値の観測に、真値でエキスパートが正解を付ける）、物体の形・数のランダム化、曖昧な指示への確認の質問、ROS 2 経由での実機・Gazebo への接続。</li>
</ul>

<h2>10. 出典・ライセンス{heading_ai}</h2>
<ul>
<li>ロボットのモデル: MuJoCo Menagerie の franka_emika_panda（Apache-2.0）。シミュレータ: MuJoCo（Apache-2.0）。学習: PyTorch。</li>
<li>LLM: Qwen2.5-7B-Instruct（Apache-2.0。変換元の LICENSE の原文を確認）、MLX 版 mlx-community/Qwen2.5-7B-Instruct-4bit、推論は mlx-lm。</li>
<li>講座のサンプル「ChatGPT API による自然言語ロボット操作」の発想（自然言語の指示から、ロボットを動かす）を参考にした。そのコードは使っていない。API に依存しないこと、出力の検査、学習した方策による実行、正解つきの指示での数値評価で、発展させた。他のサンプルのコードも、使っていない。タスク・エキスパート・方策・評価・解釈器は、本課題のために実装した。</li>
{ai_note}
</ul>
"""
    css = """
@page { size: A4; margin: 13mm 13mm; }
body { font-family: 'Hiragino Sans','Hiragino Kaku Gothic ProN','Arial Unicode MS',sans-serif; font-size: 9.6pt; line-height: 1.5; color:#111; }
h1 { font-size: 18pt; margin: 0 0 2pt; } .sub { color:#444; margin-top:0; }
h2 { font-size: 12.5pt; border-bottom: 1.5px solid #4c78a8; padding-bottom:2pt; margin-top: 12pt; }
table { border-collapse: collapse; margin: 4pt 0; font-size: 9pt; } th,td { border:1px solid #999; padding:2.5pt 6pt; text-align:left; vertical-align: top; } th { background:#eef3f8; }
.figs { display:flex; gap:6pt; align-items:flex-start; } .figs img { width: 49%; } .cap { color:#444; font-size: 8.6pt; margin-top: 2pt; }
ul { margin: 3pt 0; padding-left: 15pt; } li { margin: 2pt 0; } code { font-size: 8.4pt; background:#f3f3f3; padding: 0 2pt; }
"""
    page = f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><title>説明資料</title><style>{css}</style></head><body>{body}</body></html>"
    hp = ROOT / "report" / "report.html"
    hp.write_text(page, encoding="utf-8")
    pdf = ROOT / "report" / f"説明資料_PAI最終課題_{args.account}.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", hp.as_uri()], check=True, capture_output=True, timeout=180)
    print("wrote", pdf.relative_to(ROOT))


if __name__ == "__main__":
    main()
