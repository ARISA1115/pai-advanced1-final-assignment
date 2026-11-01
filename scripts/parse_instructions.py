"""ローカルの LLM（MLX 版 Qwen2.5-7B-Instruct 4bit）で、日本語の指示を目標に変換し、正解率を測る。
実行には、別の仮想環境（requirements-llm.txt）が要る。API キーは不要。
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sorting.instruction import canonical, extract_json, truth_canonical, InstructionError  # noqa: E402

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

SYSTEM = """あなたはロボットアームへの指示を、JSON に変換する変換器です。
机の上には、赤・緑・青の 3 つのキューブと、赤・緑・青の 3 つの置き場所があります。
キューブの色は red / green / blue、置き場所の色も red / green / blue で表します。
指示された「キューブ → 置き場所」の組だけを、次の形式で出力します。
{"moves":[{"cube":"red","to":"blue"}]}
規則:
- 指示に出てこないキューブは動かさない（moves に入れない）。
- 「色ごとに仕分け」は、各キューブをそれと同じ色の置き場所へ運ぶことです。
- 「入れ替え」は、2 つのキューブが、互いの色の置き場所へ行くことです。
- 同じ置き場所へ複数のキューブを置く指示、存在しない色・場所・動作の指示、意味が決まらない指示は、実行できません。その場合は {"error":"理由"} を出力します。
- JSON だけを出力し、説明は書かない。"""

FEWSHOT = [
    ("赤のキューブを緑の置き場所に置いてください", '{"moves":[{"cube":"red","to":"green"}]}'),
    ("青と緑のキューブの置き場所を交換して", '{"moves":[{"cube":"blue","to":"green"},{"cube":"green","to":"blue"}]}'),
    ("緑を赤の置き場所へ、赤を青の置き場所へ運んで", '{"moves":[{"cube":"green","to":"red"},{"cube":"red","to":"blue"}]}'),
    ("どのキューブも自分の色の置き場所に片付けて", '{"moves":[{"cube":"red","to":"red"},{"cube":"green","to":"green"},{"cube":"blue","to":"blue"}]}'),
    ("赤だけは動かさずに、あとは色ごとにお願い", '{"moves":[{"cube":"green","to":"green"},{"cube":"blue","to":"blue"}]}'),
    ("青と緑を両方とも赤の置き場所に載せて", '{"error":"同じ置き場所に 2 つは置けない"}'),
    ("紫のキューブを青の置き場所へ", '{"error":"紫のキューブはない"}'),
]


def build_messages(text, few_shot=True):
    msgs = [{"role": "system", "content": SYSTEM}]
    if few_shot:
        for q, a in FEWSHOT:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": text})
    return msgs


class Parser:
    def __init__(self, model=MODEL):
        from mlx_lm import generate, load
        self._generate = generate
        self.model, self.tok = load(model)

    def __call__(self, text, few_shot=True):
        prompt = self.tok.apply_chat_template(build_messages(text, few_shot), tokenize=False, add_generation_prompt=True)
        out = self._generate(self.model, self.tok, prompt=prompt, max_tokens=160, verbose=False)
        try:
            return extract_json(out), out
        except InstructionError as e:
            return {"error": f"出力が JSON でない: {e}"}, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "tests" / "instructions.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "instruction_eval.json"))
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--regrade", action="store_true", help="保存済みの出力を、LLM を使わずに採点し直す（採点の定義を変えたとき）")
    ap.add_argument("--text", nargs="*", help="評価の代わりに、この指示だけを変換して表示する")
    args = ap.parse_args()
    if args.regrade:
        res = json.load(open(args.out)); data = {d["id"]: d for d in json.load(open(args.data))}
        regrade(res, data); json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1); report(res); return
    parser = Parser()
    if args.text:
        for t in args.text:
            print(t, "->", parser(t, not args.zero_shot)[0])
        return
    data = json.load(open(args.data))
    rows, t0 = [], time.time()
    for d in data:
        parsed, raw = parser(d["text"], few_shot=not args.zero_shot)
        rows.append({"id": d["id"], "category": d["category"], "text": d["text"], "output": parsed, "raw": raw})
    res = {"model": MODEL, "few_shot": not args.zero_shot, "n": len(rows), "seconds": round(time.time() - t0), "rows": rows}
    regrade(res, {d["id"]: d for d in data})
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    report(res)


def regrade(res, data):
    """採点。correct = 出力が正解と完全一致（実行できない指示は、LLM が error を返したら正解）。
    safe = 実行できない指示で、ロボットが動かない（LLM が error を返す、または、出力が検査で弾かれる）。
    """
    cats = {}
    for r in res["rows"]:
        truth = data[r["id"]]["truth"]
        r["correct"] = canonical(r["output"]) == truth_canonical(truth)
        if truth.get("error"):
            r["safe"] = canonical(r["output"]) in ("error", "invalid")
        c = cats.setdefault(r["category"], [0, 0]); c[0] += r["correct"]; c[1] += 1
    rows = res["rows"]
    inv = [r for r in rows if r["category"] == "invalid"]
    val = [r for r in rows if r["category"] != "invalid"]
    res["accuracy"] = sum(r["correct"] for r in rows) / len(rows)
    res["by_category"] = {k: {"correct": v[0], "n": v[1]} for k, v in cats.items()}
    res["valid_instructions_accuracy"] = sum(r["correct"] for r in val) / len(val)
    res["invalid_refused_llm_only"] = sum(r["correct"] for r in inv) / len(inv)
    res["invalid_refused_with_validation"] = sum(r["safe"] for r in inv) / len(inv)
    res["invalid_executed_wrongly"] = [r["id"] for r in inv if not r["safe"]]


def report(res):
    print(f"正解率 {res['accuracy']:.3f}（実行できる指示 {res['valid_instructions_accuracy']:.3f}）、"
          f"実行できない指示の拒否: LLM のみ {res['invalid_refused_llm_only']:.2f} / 検査込み {res['invalid_refused_with_validation']:.2f}、"
          f"誤って実行された指示: {res['invalid_executed_wrongly']}")
    print(res["by_category"])
    for r in res["rows"]:
        if not r["correct"]:
            print("  誤り:", r["id"], r["text"], "->", r["output"], "(検査で拒否)" if r.get("safe") else "")


if __name__ == "__main__":
    main()
