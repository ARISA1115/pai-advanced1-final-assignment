#!/bin/bash
# 公開前の点検。何かが出力されたら、公開してはいけない（0 件で、最後に OK と出る）。
cd "$(dirname "$0")/.." || exit 1
fail=0
TEXT=(--include='*.py' --include='*.md' --include='*.json' --include='*.txt' --include='*.html' --include='*.sh' --include='*.xml' --include='*.yaml' --include='*.toml')
EX=(--exclude=check_public.sh --exclude-dir=.venv --exclude-dir=.venv-llm --exclude-dir=.git --exclude-dir=data --exclude-dir=assets --exclude-dir=__pycache__ --exclude-dir=.pytest_cache)
echo "--- 絶対パス（/Users/ など）"; grep -rIn "${TEXT[@]}" "${EX[@]}" -E "/Users/|/home/[a-z]" . && fail=1
echo "--- メールアドレスの形式"; grep -rIn "${TEXT[@]}" "${EX[@]}" -E "[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}" . | grep -v "noreply" && fail=1
echo "--- コミットの作者（コミットがある場合）"
if git rev-parse HEAD >/dev/null 2>&1; then git log --format='%an <%ae> | %cn <%ce>' | sort -u | grep -v -E "noreply" && fail=1; fi
echo "--- PDF と動画は、目視で確認すること"
[ $fail -eq 0 ] && echo "OK: 点検で見つかったものはない" || { echo "NG: 上の出力を直すこと"; exit 1; }
