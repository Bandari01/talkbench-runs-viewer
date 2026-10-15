# talkbench-runs-viewer

Backup archive of `talk-bench` / `talk-bench-talkdesk` run directories plus a web viewer to browse them.

## Data

```
data/<run-dir>/        # one run directory, copied verbatim (results.json, per-scenario artifacts)
```

Run directory names are globally unique, so runs from every checkout and both projects sit in one
flat folder. `data/` is gitignored (tens of GB); `VIEWER_DATA_DIR` in `.env` (absolute, or relative
to this repo) keeps it elsewhere — restart the viewer after changing it.
Stars, notes, deleted-run names, test jobs and scheduled tests live in dotfiles next to the runs (`data/.stars.json`,
`data/.run_notes.json`, `data/.deleted_runs.json`, `data/.test_jobs/`, `data/.test_schedules.json`).

Getting runs in:

- **Run test**: runs started from the viewer are written straight into the data dir
  (`talk-bench run … --output-dir <VIEWER_DATA_DIR>`), so there is nothing to copy or sync (see below).
- **Sync sources** (top bar), for runs started from a terminal: moves runs out of the local
  `ai-ds-research` checkouts — `~/GitHub/ai-ds-research`, `~/Documents/tau2-bench-fork/ai-ds-research`,
  or the `:`-separated paths in `VIEWER_SOURCE_REPOS`. Every new or changed run is cloned with APFS
  `cp -c` (almost no disk), then the checkout's copy is **deleted** once the archive holds every file
  of it (same size and mtime) and nothing will write to it again: it has `results.json`, *Resume* has
  nothing left to redo or score in it, no test of the viewer runs in it, and none of its files changed
  in the last 2 minutes (a talk-bench started in a terminal may be writing it). The runs kept back stay
  where `--resume` finds them and go at a later Sync, once finished. The checkouts' `data/` is
  gitignored, so a moved run exists only in the archive from then on. A terminal run started with
  `--output-dir <VIEWER_DATA_DIR>` needs no sync either.
- **Drag & drop**: drop one run folder, or a folder of runs, anywhere on the run list.

**Selecting runs**: the checkboxes on the run list (shift-click selects a range, the header box
every run shown) open a bar to hide, show or delete the selected runs at once; changing the
filters drops runs that are no longer shown from the selection.

**Deleting a run** (trash icon on a run-list row, the selection bar, or *Delete run* on the run page,
after a confirmation) deletes only the run's files in the data dir, plus its star and note — for a
run started from Run test, or one Sync moved, that is the only copy. Nothing in an `ai-ds-research`
checkout is touched: a copy still there (a run that can be resumed) stays, and the name goes into
`data/.deleted_runs.json` so Sync sources skips it. Dropping the run in again, or a Run test job
copying it (e.g. a resume), takes it off that list. A run a test is still writing cannot be deleted.

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
- A Talkdesk Autopilot scenario's transcript is laid out from `records.json` instead: one bubble per
  message, in the platform's order (conversation.json can merge several messages into one turn), the
  agent's internal hand-offs (DISPATCH / COORDINATOR / ACTION_AGENT / RESPOND) behind the
  **内部流转** toggle, and tool calls only `tool_log.json` lists placed by time. `_record_spans`
  places each message's speech from the VAD segments
- **中文翻译** on the transcript puts a Simplified Chinese translation under each turn (same LLM as
  Ask AI). It is saved as `.transcript_zh.json` beside `conversation.json` — a dotfile, so it stays
  out of the file list — and shows straight away on later visits while the reader leaves
  translations on; a scenario not translated yet is only translated on a click
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
Every agent opens with the YAML collapsed (*Edit full YAML* opens it; a YAML error opens it too).
For Talkdesk, ElevenLabs, Gemini Live and OpenAI Realtime agents the values that change per test
get fields (MCP server URL, `agent_system_id`, `agent_id`), each with a *Paste* button that reduces
a copied log line or agent link to the value. The fields edit the YAML text in place. The server validates the YAML the way talk-bench's
`AgentConfig` does. Source / domain / split are read from the checkout's data files; ticking
scenarios sets `--scenario-ids`. Jobs show live progress and a streaming log, can be stopped (the
process group is killed; talk-bench has no graceful stop) and resumed with `--resume`. The run dir
is in the data dir from the start, so it shows up in the run list within seconds and is re-read as
scenarios start and finish. A test whose run dir is in a checkout (a job recorded before that change, or
a resume of a run a checkout holds) clones it in as scenarios finish; once it completes with nothing
left to resume, its checkout copy is deleted, as Sync does. At most 4 tests run at once:
Start test and Resume are refused beyond that (a scheduled run waits for a slot), and Resume is
refused while another test is still running in the same run directory. Tests run in their own
session, so restarting the viewer does not stop them: on startup it re-attaches to running tests,
and a test that ended while the viewer was down is settled from its log (done when the run
completed, cancelled if it was being stopped, otherwise interrupted) and archived.

**Resume.** `talk-bench run --resume <run dir>` runs a run's errored and unfinished scenarios again
(and scores finished ones left unscored), with the agent and task set its `run_config.json`
recorded, and merges them into the same `results.json`. The page offers it in three places: on the
newest test of each run directory once it ended with scenarios left (stopped, crashed, or done with
errored scenarios: the `Errors: N` line after `Score:`), in *Resume a partial run*, which lists
every run in the checkouts' `data/runs` that has such scenarios — also runs started from a
terminal — plus the ones tests on this page wrote into the data dir, and as *Resume…* on the page
of such a run. A run is resumed where it lives: in the checkout if one holds it, else in the data dir. The counts come from the files talk-bench's
resume reads (each scenario's `run_meta.json` end reason, `results.json`); runs that left no
scenario directory are not listed (start the test again instead), and runs without a
`run_config.json` (made before `--resume` existed) cannot be resumed.

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
ticked scenarios, user simulator, speech complexity) together with a start time, and the viewer
starts it **once** at that time. The time is picked in the browser's time zone (shown under the
field) and must be in the future, at most a year ahead. The **Scheduled tests** table lists the
tests still waiting, with *Edit…* (loads it into the editor; *Update schedule* saves the test and
time back) and delete. Once started, a scheduled test leaves that table and shows up in the jobs
table marked *scheduled · <name>*. Recurring schedules are not supported.

A thread in the viewer process starts due tests through the same code as *Start test*, so:

- tests only start while the viewer runs and the Mac is awake; one whose time passed while the
  viewer was stopped or the Mac asleep starts as soon as the viewer is back
- while 4 tests run, a due test waits for a free slot
- the YAML is validated when the schedule is saved, but the checkout, its `.env` and any MCP tool
  server have to be in place at the start time; a test that cannot start then stays in the table
  as *not started* with the reason until it is edited or deleted

Scheduled tests are stored in `data/.test_schedules.json`, changed under an `flock` on
`data/.test_schedules.lock`, so a second viewer process on the same data dir (e.g. a `--reload` dev
server) does not start one twice.

### LLM config (Ask AI)

Copy `.env.example` to `.env` and set `VIEWER_LLM_MODEL` (litellm naming, default `azure/gpt-4.1`;
for Azure use the deployment name) plus the provider key. Restart the server after changes.
