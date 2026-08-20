# talkbench-runs-viewer

Consolidated backup of all `talk-bench/data/runs` and `talk-bench-talkdesk/data/runs` directories
from the five local `ai-ds-research*` checkouts, plus a web viewer to browse them.

## Data layout

```
data/<run-dir>/
```

- `<run-dir>`: the original run directory, copied verbatim (results.json, per-scenario
  audio/transcripts/tool logs/telemetry). Runs from all `ai-ds-research*` checkouts and
  both projects (talk-bench, talk-bench-talkdesk) are merged into one flat folder;
  run directory names are globally unique.

The backup was made with APFS clones (`cp -c -R`), so it is a real independent copy that
consumes almost no additional disk space. `data/` is gitignored (~49 GB).

To re-sync new runs from the source repos:

```bash
./scripts/sync.sh
```

(only copies run directories that don't already exist in `data/`).

## Viewer

```bash
uv sync
uv run uvicorn app.main:app --port 8377
```

Then open http://localhost:8377

### LLM config (per-scenario "Ask AI" chat)

Copy `.env.example` to `.env` and fill in the model + credentials
(`VIEWER_LLM_MODEL`, litellm naming, default `azure/gpt-4.1`).
Restart the server after changes.

Features:

- Run list with search + agent-family filter and starring, sorted by time
- Per-run summary: primary score, pass counts, talk-bench evaluation scores, per-scenario table
- Per-scenario view: goal, transcript, audio playback (conversation / user / agent wav),
  tool-call log with errors, and raw file access for every artifact
