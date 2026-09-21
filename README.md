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
consumes almost no additional disk space. `data/` is gitignored (tens of GB).

To re-sync new runs from the source repos:

```bash
./scripts/sync.sh
```

(only copies run directories that don't already exist in `data/`).

The viewer's **Sync** button does the same thing over HTTP. It looks for the checkout in
`~/GitHub/ai-ds-research` and `~/Documents/tau2-bench-fork/ai-ds-research`, syncing every
location that exists (a run found in more than one checkout is taken from the first).
Set `VIEWER_SOURCE_REPOS` to a `:`-separated list of paths to use different locations.

## Viewer

```bash
uv sync
uv run uvicorn app.main:app --port 8377
```

`data/` can live elsewhere — set `VIEWER_DATA_DIR` to point the archive at shared storage
(e.g. a mounted Azure Files share) so uploads from every host land in one place:

```bash
VIEWER_DATA_DIR=/mnt/talkbench-runs uv run uvicorn app.main:app --port 8377
```

Run **one** instance per storage location: the upload staging table and the run index are
per-process, so several instances sharing a mount would drop chunked uploads and serve
stale lists. Uploads stage in `<data dir>/.uploads/` and are renamed into place, which
keeps the commit atomic on a network mount.

Then open http://localhost:8377

### Share a read-only public link

The share command keeps the data on this computer and exposes the viewer through a
temporary Cloudflare Tunnel URL:

```bash
./scripts/share.sh
```

Copy the `https://...trycloudflare.com` URL printed in the terminal. The link works
while this computer and the command stay running. Shared visitors can browse,
use Ask AI, and sync local source repositories, but cannot star, upload, or
replace files.

The archive is tens of GB, so it is served from this computer instead of copied
to a hosting provider.

### Run test (launch talk-bench from the viewer)

The **Run test** page (sidebar) runs exactly

```bash
uv run talk-bench run --config <agent.yaml> --domain <domain> --split <split> --source tau2|talk_bench [--scenario-ids a,b]
```

inside the local `talk-bench` checkout (the first `ai-ds-research/talk-bench` the viewer finds, so
that checkout's own `.env` and `talkbench.yaml` apply) and copies the run into this archive while it
runs.
The page has the inputs of that command and nothing else; every other `talk-bench run` option
keeps its CLI default.

- **agent.yaml** — the file passed as `--config`, edited as a **form** (default) or as **YAML**
  text; a switch above the editor toggles the two. The form has one control per key: `name`,
  `module` (a dropdown of the agent classes talk-bench ships, or a custom dotted path), the
  concurrency cap, and under `params` one field per keyword argument of the chosen class's
  constructor — text, number, checkbox or dropdown as the argument's type dictates, with the
  constructor default as placeholder and a one-line hint. Empty fields are left out of the file.
  Keys the form does not know (from an imported file, or added with *Add parameter*) appear as
  generic fields typed from their values, and nested values the form has no control for are
  edited as a YAML fragment. The `mcp` block is a checkbox that reveals `server_url` and the
  identity transport; a collapsed section holds the `simulator`, `evaluation` and gate
  overrides. The YAML is regenerated from the form on every edit, shown collapsed below it,
  validated by the server the way talk-bench's `AgentConfig` does, and written as shown. In
  YAML mode the text is written as-is, comments included; loading YAML into the form drops
  comments. Fill it from a **preset** (one per agent module talk-bench ships: cascaded LLM +
  TTS/ASR, the same agent over text, OpenAI Realtime, Gemini Live, ElevenLabs SDK voice/text,
  Talkdesk Autopilot voice/text, or a custom module), **import** a YAML / JSON file or one of
  the checkout's `samples/*.yaml`, or take the `agent_config` recorded in any **archived run**
  (also reachable from a run page via *Re-run this agent…*). Configs can be saved for reuse
  (`<data dir>/.agent_configs.json`)
- **--source**, **--domain**, **--split** — dropdowns read from the checkout's own data files: the
  domains talk-bench ships and, per domain, the split names its `splits.json` (tau2:
  `split_tasks.json`) defines, with scenario counts. *All domains* offers the union of those
  splits, each applied to every domain that defines it
- **--scenario-ids** (optional) — the split's scenarios are listed with a checkbox and a one-line
  description each (filterable; *Tick all shown* / *Clear*); ticking some runs only those, ticking
  none runs the whole split. *Re-run this agent…* pre-ticks exactly the scenarios that run covered
  when they were a subset of the split. The exact command is previewed below the settings

The agent YAML is written to `<data dir>/.test_jobs/<job>/agent.yaml`. Jobs show live progress
and a streaming log; a running test can be stopped (talk-bench has no graceful stop, so the
process group is killed) and later **resumed** with `talk-bench run --resume`. Runs are written
to the checkout's `data/runs` as usual and cloned into the archive as scenarios finish and when
the run ends, so they appear in the run list.

Requirements: `uv` on the machine running the viewer (or a `.venv/bin/talk-bench` shim in the
checkout), and a populated `.env` in the checkout. Agents that use an MCP tool server (Talkdesk,
ElevenLabs with `tool_source: mcp`) still need that server started by hand with the same
`--source` / `--domain`; the page warns when the YAML has an `mcp.server_url`. The page is
disabled in read-only shared mode and with `VIEWER_DISABLE_TEST_RUNS=true`.

### LLM config ("Ask AI" chat on run and scenario pages)

Copy `.env.example` to `.env` and fill in the model + credentials
(`VIEWER_LLM_MODEL`, litellm naming, default `azure/gpt-4.1`).
For Azure, use `azure/<deployment-name>`; the deployment name may differ from
the underlying model name shown in the catalog.
Restart the server after changes.

Features:

- Sidebar navigation (shadcn/ui-style layout): **All runs / Starred / Hidden** views,
  **Voice / Text** modality filters, and agent-family filters with live counts
  (TalkDesk / ElevenLabs / Gemini-3.1-flash-live / GPT-Realtime-2.1 / others — see
  `agentGroup` in `app/static/index.html` for the mapping);
  light / dark / system theme toggle in the top bar
- Run list with debounced search (press `/` to focus, note text is searched too), sortable
  columns on every table (click a header; names sort naturally, so `tau2-2` < `tau2-10`),
  starring, and per-run notes. The **Mode** column marks each run Voice or Text — read from
  the agent module in `run_context`, falling back to whether the scenarios recorded `.wav`
  (see `_run_modality` in `app/main.py`)
- **Compare runs** (sidebar, or "Compare with…" on any run page): pick two runs and they are
  aligned by scenario id — both pass / only A / only B / both fail, aggregates over the shared
  scenarios only (goal score, tool recall & precision, turns, duration, end reasons), and a
  per-scenario table with an "only where the two disagree" filter. Comparing a text run with a
  voice run flags that `Experience` and `Eval final` are not comparable across modalities: a
  text run has no `responsiveness` / `voice_consistency`, so those never pull its experience
  score down. Use `Passed` / `Goal score` / `Resolution` / `Tool recall` instead
- Hide runs you don't want on the home list with the eye icon; the **Hidden** sidebar view
  shows them so you can bring them back. The list is per browser (`localStorage`, never sent
  to the server), so every computer — including each visitor of a shared link — has its own
- Per-run summary: primary score, pass counts, talk-bench evaluation scores, per-scenario table
- The AI features are modality-aware (`_modality_brief` / `_run_report_prompt` in
  `app/main.py`): on a text run the prompt drops ASR/transcription from the root-cause list and
  the context drops `responsiveness` / `voice_consistency` / `latency`, which talk-bench stores
  as `0` and `null` for text and which the model otherwise reports as a catastrophic failure
- Per-run "Ask AI" and one-click "AI Run 报告": the model sees the aggregate scores, a result
  line for every scenario, and goal / tool-error / transcript excerpts for the failed ones.
  The report button generates a structured write-up — totals, failure causes grouped by root
  cause with a distribution chart, high-frequency problems with evidence, and prioritized
  advice; the chat answers free-form follow-ups
- Per-scenario view: goal, transcript, audio playback (conversation / user / agent wav),
  tool-call log with errors, and raw file access for every artifact
- **Run test** page: write or import an `agent.yaml`, pick source / domain / split, and launch
  `talk-bench run` from the browser, with live log, stop / resume, and automatic import of the run
  into the archive (see *Run test* above)
- Per-scenario "Ask AI" and one-click "AI 场景报告": the report button generates a structured
  write-up of one scenario — what the task required, what the agent actually did (every tool
  call with arguments, results, and db changes), an evidence-based verdict on how well it did,
  and a mermaid flowchart of the scenario (rendered in the page; mermaid.js is lazy-loaded
  from a CDN only when a diagram appears, with the source shown as a fallback offline)
