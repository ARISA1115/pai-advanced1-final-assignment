"""スモークテスト: 環境・エキスパート・学習済みの方策・指示の検査が動く。数十秒以内で終わる。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sorting.env import OBS_DIM, SortEnv  # noqa: E402
from sorting.expert import expert_action  # noqa: E402
from sorting.instruction import InstructionError, canonical, to_task, truth_canonical  # noqa: E402
from sorting.policy import load_policy  # noqa: E402


def run_episode(env, act_fn, max_steps=800):
    ok = False
    for _ in range(max_steps):
        _, ok = env.step(act_fn())
        if ok:
            break
    return ok


def test_expert_color_task():
    env = SortEnv(0, task="color"); env.reset(10001)
    assert env.obs().shape == (OBS_DIM,)
    assert run_episode(env, lambda: expert_action(env)) and env.n_placed() == 3


def test_expert_given_goal_swap():
    env = SortEnv(0); env.reset(10002, goal=[1, 0, 2], active=[True, True, False])   # 赤と緑を入れ替え
    assert run_episode(env, lambda: expert_action(env))
    assert env.n_active() == 2 and env.n_placed() == 2


def test_learned_policy_acts():
    policy = load_policy(str(ROOT / "checkpoints" / "policy_random.pt"))
    env = SortEnv(0)
    a = policy.act(env.obs())
    assert a.shape == (4,) and np.all(np.isfinite(a)) and abs(a[3]) == 1.0


def test_instruction_validation():
    goal, active = to_task({"moves": [{"cube": "red", "to": "blue"}, {"cube": "blue", "to": "red"}]})
    assert list(goal) == [2, 1, 0] and list(active) == [True, False, True]
    bad = [
        {"error": "実行できない"},
        {"moves": [{"cube": "red", "to": "blue"}, {"cube": "green", "to": "blue"}]},   # 同じ置き場所
        {"moves": [{"cube": "yellow", "to": "blue"}]},                                  # 存在しない色
        {"moves": [{"cube": "red", "to": "blue"}, {"cube": "red", "to": "green"}]},     # 同じキューブに 2 つの目標
        {"moves": []},
    ]
    for b in bad:
        with pytest.raises(InstructionError):
            to_task(b)


def test_instruction_set_is_consistent():
    data = json.load(open(ROOT / "tests" / "instructions.json"))
    assert len(data) == 62 and len({d["id"] for d in data}) == 62
    for d in data:           # 正解が、検査を通る形になっている
        c = truth_canonical(d["truth"])
        if c != "error":
            to_task({"moves": [{"cube": a, "to": b} for a, b in d["truth"]["moves"]]})
            assert canonical({"moves": [{"cube": a, "to": b} for a, b in d["truth"]["moves"]]}) == c


def test_front_localizer_and_tracker():
    """画像（前方カメラの色の重心）から、キューブの初期位置を 6 mm 以内で推定でき、追跡が動く。"""
    from sorting.vision import FrontLocalizer, Tracker
    env = SortEnv(0)
    tr = Tracker(env, FrontLocalizer(env))
    for seed in (10001, 10002, 10003):
        env.reset(seed)
        est = tr.reset()
        assert tr.missing == 0
        for i in range(3):
            assert np.linalg.norm(est[i][:2] - env.cube_pos(i)[:2]) < 0.006
    policy = load_policy(str(ROOT / "checkpoints" / "policy_random.pt"))
    for _ in range(5):                       # 画像から作った観測で、数ステップ動かせる
        env.step(policy.act(env.obs(tr.update())))


def test_visuomotor_policy_shapes_and_planner():
    """視覚運動方策: 画像と状態から、4 次元の動作が出る。順番を進める部分（グリッパーの動きだけ）が動く。"""
    import torch
    from sorting.visuomotor import FrontCam, Planner, VisuoPolicy, VisuoPolicyFiLM, proprio_vec
    env = SortEnv(0, task="random"); env.fixed_order = True; env.reset(10001)
    cam = FrontCam(env); planner = Planner(env)
    img = cam.frame(); vec = proprio_vec(env, planner.target)
    assert img.shape == (3, 96, 96) and img.dtype == np.uint8 and vec.shape == (10,)
    for cls in (VisuoPolicy, VisuoPolicyFiLM):
        m = cls().eval()
        a = m.act(img, vec)
        assert a.shape == (4,) and np.all(np.isfinite(a)) and abs(a[3]) == 1.0
    assert planner.target in range(3)


def test_unstick_moves_only_when_stalled():
    import numpy as np
    from sorting.env import SortEnv
    from sorting.visuomotor import Unstick
    env = SortEnv(0, task="random"); env.reset(0)
    un = Unstick(env, window=5, span=0.008, steps=3, seed=0)
    a = np.array([0.0, 0.0, 0.0, -1.0])
    out = [un.filter(a) for _ in range(5)]                    # 手先を動かさないと、5 ステップ目に動かし始める
    assert np.allclose(out[0], a) and not np.allclose(out[4][:3], 0)
    assert out[4][3] == -1.0 and un.count == 1                # 握る・開くは、方策の出力のまま
