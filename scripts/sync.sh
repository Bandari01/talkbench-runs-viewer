#!/bin/bash
# Incrementally back up run directories from the ai-ds-research checkouts into data/.
# Uses APFS clonefile (cp -c) — fast and near-zero extra disk usage.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data"

# The checkouts live in a different place on each machine: look next to this
# repo plus the known locations, and use whichever roots exist.
ROOTS=("$(cd "$(dirname "$0")/../.." && pwd)" "$HOME/GitHub" "$HOME/Documents/tau2-bench-fork")

seen=""
for root in "${ROOTS[@]}"; do
  [ -d "$root" ] || continue
  root="$(cd "$root" && pwd -P)"
  case ":$seen:" in *":$root:"*) continue;; esac
  seen="$seen:$root"
  for d in ai-ds-research ai-ds-research-1 ai-ds-research-2 ai-ds-research-3 ai-ds-research-4; do
    for sub in talk-bench talk-bench-talkdesk; do
      src="$root/$d/$sub/data/runs"
      [ -d "$src" ] || continue
      mkdir -p "$DEST"
      new=0
      for run in "$src"/*/; do
        name=$(basename "$run")
        [ -e "$DEST/$name" ] && continue
        cp -c -R "$run" "$DEST/$name"
        new=$((new + 1))
      done
      echo "$root/$d/$sub: +$new new runs"
    done
  done
done
