"""レポート用の図を作る。"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import _path  # noqa: F401
from _path import ROOT
from sorting.render import FONT_CANDIDATES

for p in FONT_CANDIDATES:
    try:
        font_manager.fontManager.addfont(p)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
        break
    except Exception:
        continue

R = ROOT / "results"
CAT = {"single": "単独", "swap": "入れ替え", "multi": "複数", "sort": "色ごと", "exclude": "除外", "casual": "口語", "invalid": "実行不可"}

# 方策の学習曲線（段階 1）
hist = json.load(open(R / "train_history.json"))
fig, ax = plt.subplots(figsize=(5.2, 3.3))
ax.plot([h["epoch"] for h in hist], [h["train"] for h in hist], label="学習")
ax.plot([h["epoch"] for h in hist], [h["val"] for h in hist], label="検証（別のエピソード）")
ax.set_xlabel("エポック"); ax.set_ylabel("MSE 損失"); ax.set_yscale("log"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(R / "fig_loss.png", dpi=150); plt.close(fig)

# デモの本数（段階 0）
ab = json.load(open(R / "baseline" / "ablation.json"))
fig, ax = plt.subplots(figsize=(5.2, 3.3))
xs = [str(r["demos"]) for r in ab["n_demos"]]
ax.bar(xs, [r["full_success_rate"] * 100 for r in ab["n_demos"]], color="#4c78a8")
for i, r in enumerate(ab["n_demos"]):
    ax.text(i, r["full_success_rate"] * 100 + 1, f'{r["full_success_rate"]*100:.0f}%', ha="center")
ax.set_xlabel("デモの本数"); ax.set_ylabel("3 個とも成功 [%]"); ax.set_ylim(0, 112)
ax.set_title("デモの本数と成功率（段階 0、各 100 エピソード）", fontsize=9)
fig.tight_layout(); fig.savefig(R / "fig_demos.png", dpi=150); plt.close(fig)

# 指示の解釈（カテゴリ別、few-shot あり / なし）
few = json.load(open(R / "instruction_eval.json"))
zero = json.load(open(R / "instruction_eval_zeroshot.json"))
cats = list(CAT)
fig, ax = plt.subplots(figsize=(6.4, 3.3))
w = 0.38
for k, (res, lab, col) in enumerate(((few, "few-shot あり", "#4c78a8"), (zero, "ゼロショット", "#f58518"))):
    ys = [res["by_category"][c]["correct"] / res["by_category"][c]["n"] * 100 for c in cats]
    ax.bar([i + (k - 0.5) * w for i in range(len(cats))], ys, w, label=lab, color=col)
ax.set_xticks(range(len(cats))); ax.set_xticklabels([f'{CAT[c]}\n({few["by_category"][c]["n"]})' for c in cats], fontsize=8)
ax.set_ylabel("正解率 [%]"); ax.set_ylim(0, 112); ax.legend(loc="lower left", fontsize=8)
ax.set_title("指示の解釈の正解率（カテゴリ別、LLM の出力のみ）", fontsize=9)
fig.tight_layout(); fig.savefig(R / "fig_instr.png", dpi=150); plt.close(fig)

# CNN の誤差（検証）
ct = json.load(open(R / "cubenet_train.json"))
fig, ax = plt.subplots(figsize=(5.2, 3.3))
for k, name in enumerate(("x", "y", "z")):
    ax.plot([h["epoch"] for h in ct["history"]], [h["val_mae_mm"][k] for h in ct["history"]], label=name)
ax.set_xlabel("エポック"); ax.set_ylabel("検証の平均絶対誤差 [mm]"); ax.set_yscale("log"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(R / "fig_cubenet.png", dpi=150); plt.close(fig)
print("saved figures")

# 画像からの位置推定の方式の比較（閉ループの成功率）
def _load(name):
    p = R / name
    return json.load(open(p)) if p.exists() else None

cnn, col1, col2 = _load("vision_eval_before_dagger.json"), _load("vision_eval_color_v1_200.json"), _load("vision_eval_200.json")
if cnn and col1 and col2:
    fin = _load("vision_eval.json") or col2
    bars = [("真値の観測\n（上限）", fin["state_success"], "#999999"),
            ("CNN\n（全結合）", cnn["vision_success"], "#e45756"),
            ("色の重心\n+ 追跡（第 1 版）", col1["vision_success"], "#f58518"),
            ("色の重心 + 追跡\n+ 再検出（最終版）", fin["vision_success"], "#4c78a8")]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(range(len(bars)), [b[1] * 100 for b in bars], color=[b[2] for b in bars])
    for i, b in enumerate(bars):
        ax.text(i, b[1] * 100 + 1.5, f"{b[1]*100:.1f}%", ha="center")
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=7.5)
    ax.set_ylabel("3 個とも成功 [%]"); ax.set_ylim(0, 112)
    ax.set_title("画像から位置を推定した閉ループの成功率（方策は同じ）", fontsize=9)
    fig.tight_layout(); fig.savefig(R / "fig_vision.png", dpi=150); plt.close(fig)
    print("saved fig_vision")
