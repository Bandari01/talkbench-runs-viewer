#!/bin/bash
# Incrementally back up run directories from the ai-ds-research checkouts into data/.
# Uses APFS clonefile (cp -c) — fast and near-zero extra disk usage.
set -euo pipefail

GITHUB_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data"

for d in ai-ds-research ai-ds-research-1 ai-ds-research-2 ai-ds-research-3 ai-ds-research-4; do
  for sub in talk-bench talk-bench-talkdesk; do
    src="$GITHUB_DIR/$d/$sub/data/runs"
    [ -d "$src" ] || continue
    tgt="$DEST/$sub"
    mkdir -p "$tgt"
    new=0
    for run in "$src"/*/; do
      name=$(basename "$run")
      [ -e "$tgt/$name" ] && continue
      cp -c -R "$run" "$tgt/$name"
      new=$((new + 1))
    done
    echo "$d/$sub: +$new new runs ($(ls "$tgt" | wc -l | tr -d ' ') total)"
  done
done
