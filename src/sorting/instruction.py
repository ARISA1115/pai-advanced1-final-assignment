"""自然言語の指示 → 目標（どのキューブを、どの置き場所へ）への変換と、検査。

LLM の出力は信用せず、必ずここで検査してから実行に回す（存在しない色、同じ置き場所への衝突は拒否）。
このファイルは、LLM のライブラリに依存しない。
"""
from __future__ import annotations

import json
import re

import numpy as np

COLORS = ("red", "green", "blue")
IDX = {c: i for i, c in enumerate(COLORS)}
JA = {"red": "赤", "green": "緑", "blue": "青"}


class InstructionError(ValueError):
    pass


def extract_json(text: str) -> dict:
    """出力から、最初の { ... }（括弧の対応が取れる範囲）を取り出して JSON として読む。"""
    start = text.find("{")
    if start < 0:
        raise InstructionError("JSON が見つからない")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as e:
                    raise InstructionError(f"JSON が壊れている: {e}")
    raise InstructionError("括弧が閉じていない")


def to_task(parsed: dict):
    """検査して、環境へ渡す (goal, active) にする。不正なら InstructionError。"""
    if "error" in parsed:
        raise InstructionError(f"指示を実行できない: {parsed['error']}")
    moves = parsed.get("moves")
    if not isinstance(moves, list) or not moves:
        raise InstructionError("moves が空、または形式が違う")
    goal = np.arange(3)
    active = np.zeros(3, bool)
    used_pads: set[str] = set()
    for m in moves:
        if not isinstance(m, dict):
            raise InstructionError("moves の要素の形式が違う")
        cube, to = m.get("cube"), m.get("to")
        if cube not in IDX or to not in IDX:
            raise InstructionError(f"存在しない色: cube={cube!r}, to={to!r}")
        if active[IDX[cube]]:
            raise InstructionError(f"同じキューブに複数の目標: {cube}")
        if to in used_pads:
            raise InstructionError(f"同じ置き場所に複数のキューブ: {to}")
        used_pads.add(to)
        goal[IDX[cube]] = IDX[to]
        active[IDX[cube]] = True
    return goal, active


def describe(goal, active) -> str:
    parts = [f"{JA[COLORS[i]]}→{JA[COLORS[goal[i]]]}の置き場所" for i in range(3) if active[i]]
    return "、".join(parts) if parts else "（動かさない）"


def canonical(parsed: dict):
    """採点用に、順序に依らない形へ。エラーは 'error'。"""
    if "error" in parsed:
        return "error"
    try:
        to_task(parsed)
    except InstructionError:
        return "invalid"
    return tuple(sorted((m["cube"], m["to"]) for m in parsed["moves"]))


def truth_canonical(truth: dict):
    if truth.get("error"):
        return "error"
    return tuple(sorted(tuple(x) for x in truth["moves"]))
