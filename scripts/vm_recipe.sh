#!/bin/bash
# 視覚運動方策（画像を入力にした方策）の、最終の作り方。実際に実行した順番と、設定をそのまま残している。
# 実行環境: Apple Silicon の GPU（MPS）で学習する。CPU だけでは、約 9 倍遅く、6 エポックでは、成功率 0% だった。
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

# 1. エキスパート（ノイズ付き、キューブの運ぶ順番は、色の番号の順）のデータ。320 エピソードを 2 回（種 100000〜、200000〜）
for k in 0 1 2 3 4 5 6 7; do python scripts/vm_collect.py --shard $k/8 --episodes 320 --seed0 100000 --out data/vm/expert_$k.npz & done; wait
for k in 0 1 2 3 4 5 6 7; do python scripts/vm_collect.py --shard $k/8 --episodes 320 --seed0 200000 --out data/vm/expert2_$k.npz & done; wait

# 2. 模倣学習（BC）。前半 30 エポック（エキスパートのデータ 1 回分）、後半 30 エポック（2 回分、学習率を半分）
python scripts/vm_train.py --data "data/vm/expert_*.npz" --epochs 30 --arch base --out checkpoints/vm_base_e30.pt
python scripts/vm_train.py --data "data/vm/expert_*.npz" "data/vm/expert2_*.npz" --init checkpoints/vm_base_e30.pt --epochs 30 --arch base --lr 5e-4 --out checkpoints/vm_base_e60.pt

# 3. DAgger: 30 エポックの方策で動かし（種 300000〜）、特権情報を見るエキスパートが正解を付ける
for k in 0 1 2 3 4 5 6 7; do python scripts/vm_collect.py --shard $k/8 --episodes 320 --seed0 300000 --policy checkpoints/vm_base_e30.pt --out data/vm/dagger2_$k.npz & done; wait

# 4. 全データで、仕上げの学習（60 エポックの方策から、25 エポック）
python scripts/vm_train.py --data "data/vm/expert_*.npz" "data/vm/expert2_*.npz" "data/vm/dagger2_*.npz" --init checkpoints/vm_base_e60.pt --epochs 25 --arch base --lr 5e-4 --out checkpoints/vm_dagger2.pt

# 5. 評価（学習・DAgger に使っていない種 10000〜、200 エピソード）
for k in 0 1 2 3 4 5 6 7; do python scripts/vm_eval.py --policy checkpoints/vm_dagger2.pt --episodes 200 --shard $k/8 --out results/vm_eval_dagger2_part$k.json & done; wait
