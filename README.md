# talkbench-runs-viewer

Backup archive of `talk-bench` / `talk-bench-talkdesk` run directories plus a web viewer to browse them.

## Data

```
data/<run-dir>/        # one run directory, copied verbatim (results.json, per-scenario artifacts)
```

Run directory names are globally unique, so runs from every checkout and both projects sit in one
flat folder. `data/` is gitignored (tens of GB); set `VIEWER_DATA_DIR` to keep it elsewhere.
Stars, notes, test jobs and test schedules live in dotfiles next to the runs (`data/.stars.json`,
`data/.run_notes.json`, `data/.test_jobs/`, `data/.test_schedules.json`).

Getting runs in:

- **Sync sources** (top bar): clones every new or changed run from the local `ai-ds-research`
  checkouts — `~/GitHub/ai-ds-research`, `~/Documents/tau2-bench-fork/ai-ds-research`, or the
  `:`-separated paths in `VIEWER_SOURCE_REPOS` — with APFS `cp -c`, so the copy costs almost no disk.
- **Drag & drop**: drop one run folder, or a folder of runs, anywhere on the run list.
- **Run test**: runs started from the viewer are copied in automatically (see below).

## Viewer

```bash
uv sync
uv run uvicorn app.main:app --port 8377
```

Open http://localhost:8377. Run one instance per data directory: the run index and upload staging
are per-process.

There is no authentication: every visitor can star, note, import runs and start tests. If you expose
the viewer through a tunnel, put the access control there (e.g. `ngrok http 8377 --oauth google
--oauth-allow-domain <your domain>`).

### Features

- Sidebar views (All / Starred / Hidden), Voice / Text filter, agent-family filters with counts
  (`agentGroup` in `app/static/index.html`), light / dark / system theme
- Run list: search (`/`; notes are searched too), sortable columns, stars, per-run notes, and
  per-browser hidden runs (`localStorage`, never sent to the server)
- Run page: scores, per-scenario table, raw result files, **Ask AI** chat and a one-click
  **AI Run 报告** (totals, failure causes grouped by root cause, evidence, advice)
- Scenario page: goal, transcript with tool calls merged chronologically, audio, scores, raw files,
  **Ask AI** and a one-click **AI 场景报告** with a mermaid flowchart (mermaid.js is loaded from a CDN
  when a diagram appears; the source stays visible offline)
- Transcript times on a voice run are positions in the wavs (click one to play from there). talk-bench
  stamps a turn when it is finalized — the user's speech end, or ~1.5 s after the agent's audio ends —
  while the wavs start at the timeline's `greeting_wait_start`, so `_audio_alignment` in `app/main.py`
  places each turn's speech from `latency_markers.ndjson` (user) and `vad_events.ndjson` (agent); a
  turn it can't place shows its finalization time, marked `≈`
- **Compare runs**: two runs aligned by scenario id — both pass / only A / only B / both fail,
  aggregates over the shared scenarios, per-scenario table with an "only where the two disagree"
  filter. Text-vs-voice pairs are flagged: `Experience` and `Eval final` are not comparable across
  modalities because a text run has no `responsiveness` / `voice_consistency`
- The AI prompts are modality-aware (`_MODALITY_BRIEF` / `_run_report_prompt` in `app/main.py`):
  on a text run the context drops `responsiveness` / `voice_consistency` / `latency`, which
  talk-bench stores as `0` / `null` for text

### Run test

The **Run test** page runs exactly

```bash
uv run talk-bench run --config <agent.yaml> --domain <domain> --split <split> --source tau2|talk_bench [--scenario-ids a,b] [--speech-complexity <preset>]
```

inside the local `talk-bench` checkout (the first one the viewer finds, so its own `.env` and
`talkbench.yaml` apply) — or inside the sibling `talk-bench-talkdesk` checkout when the agent
module's package is one that workspace provides (`talkdesk_agent.*`), since it is importable only
from that venv; there `talk-bench-talkdesk/talkbench.yaml` applies and both `.env` files load,
`talk-bench-talkdesk/.env` first (its `TALKDESK_*` values win). The agent YAML is edited as text
and written as-is to
`<data dir>/.test_jobs/<job>/agent.yaml`; fill it from a preset (one per agent module talk-bench
ships, with QA and STG variants of the Talkdesk Autopilot ones), an imported YAML / JSON file or one of either checkout's `samples/*.yaml`, or the
`agent_config` recorded in an archived run (also via *Re-run this agent…* on a run page).
Talkdesk, ElevenLabs and Gemini Live agents open with the YAML collapsed: only the values that
change per test get fields (MCP server URL, `agent_system_id`, `agent_id`), each with a *Paste*
button that reduces a copied log line or agent link to the value. The fields edit the YAML text in
place; *Edit full YAML* opens the rest. The server validates the YAML the way talk-bench's
`AgentConfig` does. Source / domain / split are read from the checkout's data files; ticking
scenarios sets `--scenario-ids`. Jobs show live progress and a streaming log, can be stopped (the
process group is killed; talk-bench has no graceful stop) and resumed with `--resume`, and the run
is cloned into the archive as scenarios finish and when the run ends. At most 4 tests run at once:
Start test and Resume are refused beyond that (a scheduled run waits for a slot), and Resume is
refused while another test is still running in the same run directory. Tests run in their own
session, so restarting the viewer does not stop them: on startup it re-attaches to running tests,
and a test that ended while the viewer was down is settled from its log (done when the run
completed, cancelled if it was being stopped, otherwise interrupted) and archived.

**User simulator.** `talk-bench` (default) is talk-bench's own `EventDrivenSimulator`. `tau2 native`
has tau2-bench's own voice user simulator play the caller in a voice run, and the page then runs

```bash
uv run python <viewer>/tau2_user_sim/launch.py run …same arguments…
```

in the same checkout. `tau2_user_sim/launch.py` is the talk-bench CLI with one change made in that
process only: each voice scenario's simulator is `Tau2NativeVoiceUserSimulator`
(`tau2_user_sim/tau2_native_user_simulator.py`). Nothing in `ai-ds-research` is modified. What the
caller says and when it hangs up come from the vendored tau2 package at runtime: tau2's voice
guidelines (`simulation_guidelines_voice.md`), persona and `<scenario>` prompt, its `generate` call
with the CUSTOMER role reminder, the `###STOP###` / `###TRANSFER###` / `###OUT-OF-SCOPE###` tokens,
`[Both parties silent for X seconds]` annotations, and its interruption-decision prompt. TTS/ASR,
audio effects, *when* the simulator is woken, and scoring stay talk-bench's. tau2's 200 ms tick
loop maps onto talk-bench's wakes: agent turn finished → reply; silence threshold → check-in;
agent speaking for N s → tau2's YES/NO interruption policy. Native talk-bench tasks work too: their
`user_scenario` is rendered in tau2's instruction layout. Limits:

- voice agents only — a text run on tau2 tasks already uses tau2's own simulator (`Tau2FaithfulUserSimulator`)
- no backchannels ("uh-huh"), because talk-bench has no backchannel action
- no user-side tools, the same as the text-channel tau2 simulator, so tasks where the caller
  must act on their own phone (tau2 telecom) cannot finish; the log warns
- the simulator's LLM is `simulator.llm_model` from the checkout's `talkbench.yaml`; the
  interruption decision uses it too unless `TAU2_USER_SIM_DECISION_LLM` is set in the checkout's `.env`
  or the shell that starts the viewer — tests do not see the viewer's own `.env` (tau2 itself uses `gpt-4.1`)
- talk-bench's simulator / mid-turn / terminal-hangup gates are switched off for these runs, because
  they would override tau2's decisions

`--speech-complexity` (tau2 tasks only) picks tau2's per-task caller persona, voice and audio effects
for either simulator. `regular` is tau2's default: noisy audio and terse callers who interrupt. It
is the only setting where the tau2 simulator interrupts; without it the caller is tau2's default
persona, as in `control`. Runs made this way carry `user_simulator.json` and show a **τ² user** tag in
the run list. Resume keeps the simulator.

Requirements: `uv` (or `.venv/bin/talk-bench` in the checkout) and a populated `.env` in the
checkout. Agents that use an MCP tool server (Talkdesk, ElevenLabs with `tool_source: mcp`) need
that server started by hand with the same `--source` / `--domain` (the Talkdesk presets' YAML comments
show the command; the line under the editor says "tools via MCP" when the YAML has an `mcp.server_url`).

### Scheduled tests

**Schedule…** next to *Start test* saves the editor's test (agent YAML, source / domain / split,
ticked scenarios, user simulator, speech complexity) with a timing: every day, weekdays, chosen
days, every N hours, or any 5-field cron expression (`minute hour day-of-month month day-of-week`,
with `*`, ranges, steps, lists and `mon`/`jan` names). Times are the viewer host's local time; the
form previews the next three runs. The **Scheduled tests** table lists next / last run, and offers
*Run now*, *Edit…* (loads the schedule into the editor; *Update schedule* saves it back), *Pause* /
*Resume* and delete. Jobs a schedule started say so in the jobs table.

A thread in the viewer process starts due tests through the same code as *Start test*, so:

- tests only run while the viewer runs and the Mac is awake; a run missed while it was stopped or
  asleep starts once when it is back (several missed runs collapse into one); resuming a paused
  schedule does not make up runs
- a due run is skipped while the schedule's previous test is still running, and waits for a free
  slot while 4 tests run
- a schedule must not fire more often than every 15 minutes
- the YAML is validated when the schedule is saved, but the checkout, its `.env` and any MCP tool
  server have to be in place when the run starts

Schedules are stored in `data/.test_schedules.json`, changed under an `flock` on
`data/.test_schedules.lock`, so a second viewer process on the same data dir (e.g. a `--reload` dev
server) does not start a run twice.

### LLM config (Ask AI)

Copy `.env.example` to `.env` and set `VIEWER_LLM_MODEL` (litellm naming, default `azure/gpt-4.1`;
for Azure use the deployment name) plus the provider key. Restart the server after changes.
