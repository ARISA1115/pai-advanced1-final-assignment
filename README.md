# 日本語で指示できる、学習した AI のロボットアーム（Physical AI 応用編1 最終課題）

机の上の 3 色のキューブを、ロボットアーム（Franka Emika Panda、MuJoCo）が、指示された置き場所へ運びます。
「赤と青を入れ替えて」のような**日本語の指示**を、学習した AI（LLM と方策）と、画像からの位置推定を組み合わせて実行します。API キーは使わず、全て手元で動きます。

| AI | 役割 | 方法 |
|---|---|---|
| ① 指示の解釈 | 日本語の指示 → 目標（どのキューブを、どの置き場所へ） | ローカルの LLM（Qwen2.5-7B-Instruct 4bit、MLX）+ 出力の検査 |
| ② 動作 | 観測 → 手先の変位と把持 | 模倣学習（Behavior Cloning）+ DAgger の MLP 方策 |
| ③ 認識 | カメラ画像 → キューブの位置 | **最終版: 色の領域の重心（カメラの幾何で校正）+ 追跡**。学習した CNN も試したが、精度が足りず（閉ループ 27%）置き換えた |

- 動画: YouTube に提出（等倍速。URL は、提出フォームに記載）
- 数値は、すべて `results/*.json` にあります（手で書き写していません）。説明資料は `report/` にあります。

## 段階

| 段階 | 内容 | 主な結果ファイル |
|---|---|---|
| 0 ベースライン | キューブを、同じ色の置き場所へ運ぶ（色固定） | `results/baseline/` |
| 1 目標の一般化 | 動かす数・割り当てをエピソードごとに変える。方策は目標を観測で受け取る | `results/eval_random.json`、`results/dagger_random.json` |
| 2 自然言語 | 日本語の指示を、LLM で目標に変換して実行。実行できない指示は拒否 | `results/instruction_eval.json`、`results/e2e_eval.json` |
| 3 画像からの認識 | カメラ画像から位置を推定し、その推定値で閉ループ制御（CNN → 色の重心 + 追跡 の順に改善） | `results/vision_eval.json`（最終）、`results/vision_eval_before_dagger.json`（CNN）、`results/vision_eval_color_v1_200.json`、`results/front_loc_heldout.json` |

## 主な結果

| 段階 | 評価 | 結果 |
|---|---|---|
| 0 色固定 | 新しい種 1000 エピソード、3 個とも成功 | BC のみ 97.8%（失敗 22）→ DAgger 99.8%（失敗 2）。差は確か（Fisher の正確確率検定で p < 0.001） |
| 1 目標を変えたタスク | 1000 エピソード | 学習した方策 99.9%（エキスパート 99.8%、何もしない 0%）。BC のみ（99.5%）との差は、確かめられなかった |
| 2 日本語の指示 | 正解つきの指示 62 件 | 解釈の正解率 91.9%。実行できない指示 10 件は、検査込みで 10/10 拒否（誤って動いたもの 0 件）。指示から動作まで 95.2% |
| 3 画像からの位置推定 | 1000 エピソード | **87.4%**（真値の観測では 99.9%）。学習した CNN だけでは 27%。誤差の中央値は x 1.2 / y 0.5 mm |

画像からの推定で残る失敗は、搬送中にキューブがグリッパーから外れて落ちるものです（再実行して調べた 48 件はすべて）。詳しくは、`report/` の説明資料の生成スクリプトと、`results/` を参照してください。

評価の乱数の種は、学習・検証・最終評価・確認・画像・指示で、重ならないように分けています（`scripts/` の各ファイルの先頭を参照）。

## 追加: 画像だけで動く方策（視覚運動方策）

上の方策はキューブの位置を入力にしますが、こちらは**カメラ画像から、直接、手先の動きを出します**。入力は、画像、ロボット自身の手先・グリッパー、今のタスク（運ぶ色と置き場所）だけで、キューブの位置は含みません。正解は、真の位置を見るエキスパート（学習時のみ）が付けます。実装は `src/sorting/visuomotor.py`、最終の条件は `scripts/vm_final.sh` です。

| 評価 | 結果 | ファイル |
|---|---|---|
| **最終**: 学習にも設定の選択にも使っていない種 810000〜、200 エピソード | **3 個とも成功 76.0%（152/200、95% 信頼区間 69.6〜81.4%）**、個数ベース 85.2%。事前に決めた基準（50%）を超えた | `results/vm_final.json` |
| 日本語の指示から動作まで（保存済みの LLM の出力を使用） | 実行した 49 件（各 3 回）の平均 72.8%。実行できない指示 10/10 を拒否 | `results/vm_instruction_eval.json` |

**位置を推定して動かす方式（87.4%、指示から動作まで 95.2%）には届いていません。** 比較と改善の過程（種 800000〜、各 200 エピソード）:

| 構成 | 3 個とも成功 | ファイル |
|---|---|---|
| CNN + DAgger + 順番を画像で進める | 66.5% | `results/vm_eval_fresh.json` |
| DAgger をもう 1 回 | 66.5%（改善せず） | `results/vm_eval_d4_fresh.json` |
| 3 つの方策の出力の平均 | 66.5%（改善せず） | `results/vm_sel_ens.json` |
| 手先が止まったら少し動かす処理を追加 | 69.5% | `results/vm_sel_d3_unstick.json` |
| **最初に運ぶ色の偏りを直すデータを追加して再学習（採用）** | **79.0%**（直前の構成と同じ種で比べて p = 0.016、McNemar 検定） | `results/vm_sel_d5_unstick.json` |
| 拡散モデル（握る・開くは分類、デノイザーは残差 MLP）。学習は 8 エポック中 4 で打ち切り | 8.5% | `results/vm3g_eval_fresh.json` |

- 79.0% は設定の選択に使った種の数字で、発表する数字は、別の種で測った 76.0% です。
- 評価の条件は、ステップ上限 800 です。運ぶ順番は、手を開いた位置の規則と画像の判定で進めます。動作そのものは方策が決めます。
- 拡散モデルの結果は、打ち切った時点のものです。手法そのものの優劣を示すものではありません。
- 事前学習した ResNet18 と行動の系列を出す版（`results/vm2_eval_*.json`）は、23.0%、17.0% でした。

再現: `bash scripts/vm_final.sh`（重み `checkpoints/vm_dagger5.pt` を同梱。動画の生成も含みます）。

## 仕組み

```
日本語の指示 → LLM → JSON → 検査(to_task) → 目標 (goal, active) ──┐   拒否 → ロボットを動かさない
前方斜め上のカメラ → 色の重心 + 追跡 → キューブの位置 ──────────────┤
手先・グリッパー（ロボット自身の関節）──────────────────────────────┴→ 観測 17 次元 → 方策 → 行動 4 次元 → MuJoCo
```

- **観測（17 次元）**: 手先の位置、グリッパーの開き、対象キューブの手先と置き場所に対する相対位置、目標の置き場所の one-hot、持ち上がっているか、残りの数、手先の高さ など（[src/sorting/env.py](src/sorting/env.py)）
- **行動（4 次元）**: 手先の変位（x, y, z。1 制御あたり最大 2 cm）と、グリッパーの開閉（20 Hz、減衰最小二乗法の逆運動学）
- **エキスパート**: 幾何に基づく規則（[src/sorting/expert.py](src/sorting/expert.py)）。デモの生成と、DAgger の正解付けに使う
- **LLM の出力は信用しない**: [src/sorting/instruction.py](src/sorting/instruction.py) の検査（存在しない色、同じ置き場所への衝突、形式の誤り）を通ったものだけを実行します

## 再現手順

Python 3.12、macOS（Apple Silicon）で確認しました。GPU は不要です。

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 学習済みの方策（checkpoints/）と、LLM の出力（results/instruction_eval.json）を同梱しているので、次はそのまま動きます
python -m pytest -q tests
python scripts/evaluate.py --episodes 100 --task random --policy checkpoints/policy_random.pt
python scripts/e2e_eval.py --parsed results/instruction_eval.json        # 指示 → 動作の評価（LLM は不要）

# 学習から作り直す場合
python scripts/collect.py --episodes 600 --task random --out data/demos_random.npz
python scripts/train.py --data data/demos_random.npz --out checkpoints/policy_random_bc.pt
python scripts/dagger.py --iters 2 --episodes 300 --task random
python scripts/calibrate_front.py                                         # 前方カメラの校正（校正済みの値を checkpoints/front_calib.json に同梱）
python scripts/vision_eval.py --estimator color --episodes 200            # 画像から推定した位置で動かす評価（--shard k/n で並列化）
# 比較用: 学習した CNN の推定器
python scripts/vision_collect.py --episodes 300 --seed0 50000 --out data/vision_train.npz
python scripts/vision_collect.py --episodes 60 --seed0 55000 --out data/vision_val.npz
python scripts/vision_train.py && python scripts/vision_eval.py --estimator cnn
python scripts/make_figures.py && python scripts/make_video.py

# 指示の解釈（LLM）は、別の仮想環境で。モデルは初回に自動で取得されます（約 4 GB）
python3.12 -m venv .venv-llm && source .venv-llm/bin/activate && pip install -r requirements-llm.txt
python scripts/parse_instructions.py                                      # 62 件の正解つき指示で評価
python scripts/parse_instructions.py --text "赤と青を入れ替えて"          # 1 件を変換して表示
```

## ファイル構成

| パス | 内容 |
|---|---|
| `assets/franka_emika_panda/` | MuJoCo Menagerie の Franka Emika Panda（Apache-2.0）と、本課題のシーン `sorting_scene.xml` |
| `src/sorting/` | `env.py`（環境）、`expert.py`、`policy.py`、`instruction.py`（検査）、`vision.py`（CNN）、`render.py` |
| `scripts/` | 収集・学習・DAgger・評価・動画・図の生成 |
| `tests/instructions.json` | 正解つきの指示 62 件 |
| `tests/test_smoke.py` | スモークテスト |
| `checkpoints/` | 学習済みの方策、CNN |
| `results/` | 評価結果（json）、図 |
| `report/` | 説明資料の生成スクリプト |

## 制約（正直な範囲）

- シミュレータ内の結果です。実機では、画像のずれ・照明・接触の不確かさが加わります。
- キューブは 3 個・同形・同じ大きさで、置き場所は固定です。
- 指示は、色（赤・緑・青）で、キューブと置き場所を指す形に限ります。「右の」「手前の」などには未対応です。
- 画像からの位置推定は、持ち上げ中のキューブが、指の中で滑って落ちる場面に弱く、真値の観測（99.9%）より成功率が下がります（87.4%）。持っているキューブは、手とハンドに隠れるため、カメラで見られません。
- `scripts/vision_dagger.py`（推定値の観測で DAgger）は、方策の真値での成功率を下げたため、採用していません。

## 出典・ライセンス

- ロボットのモデル: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) の `franka_emika_panda`（Apache-2.0。[assets/franka_emika_panda/LICENSE](assets/franka_emika_panda/LICENSE)）
- シミュレータ: [MuJoCo](https://github.com/google-deepmind/mujoco)（Apache-2.0）／ 学習: [PyTorch](https://pytorch.org/)
- LLM: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)（Apache-2.0）の MLX 4bit 版 [mlx-community/Qwen2.5-7B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit)、推論は [mlx-lm](https://github.com/ml-explore/mlx-lm)。重みは、このリポジトリに含めていません。
- 本リポジトリのコード: MIT（[LICENSE](LICENSE)）
