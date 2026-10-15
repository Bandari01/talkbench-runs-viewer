"""FastAPI backend for browsing archived talk-bench run data.

Data layout:
    data/<run-dir>/
        results.json                        (may be missing for errored runs)
        talk_bench_evaluation_result.json   (may be missing)
        <scenario-id>/
            scenario.json, conversation.json, run_meta.json, tool_log.json,
            records.json (Talkdesk Autopilot), conversation.wav, user.wav, agent.wav, *.ndjson, ...
"""

from __future__ import annotations

import ast
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from functools import cache, lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import yaml
from dotenv import dotenv_values, load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
# Settings, LLM credentials and model come from this project's own .env (see .env.example).
# Tests launched from the Run test page get the pre-.env environment, so the
# checkout's own .env decides their credentials (see _child_env).
_BASE_ENV = dict(os.environ)
load_dotenv(BASE_DIR / ".env")
# VIEWER_DATA_DIR is where runs are stored (a relative path is taken from this repo); tests started
# from the Run test page write their run straight into it (talk-bench --output-dir), so nothing is
# copied. Run one viewer per data dir: the run index, run cache and upload staging are per-process.
DATA_DIR = BASE_DIR / Path(os.environ.get("VIEWER_DATA_DIR") or "data").expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)  # fresh clone, or a new VIEWER_DATA_DIR
STATIC_DIR = Path(__file__).resolve().parent / "static"
LLM_MODEL = os.environ.get("VIEWER_LLM_MODEL", "azure/gpt-4.1")

TS_RE = re.compile(r"(\d{8}T\d{6}Z)")

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # scheduled tests start with the server, not on import (scripts may import this module)
    threading.Thread(target=_scheduler_loop, name="test-scheduler", daemon=True).start()
    yield


app = FastAPI(title="talk-bench runs viewer", lifespan=_lifespan)


# Starlette's GZip skips 206 (Range) and audio/* responses, so wav seeking is unaffected
app.add_middleware(GZipMiddleware, minimum_size=1024)

# agent config recorded in each archived run's results.json, keyed by run
# name — feeds the "from archived run" picker on the Run test page
_archived_run_context: dict[str, dict[str, Any]] = {}


class PublicDataFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        if any(part.startswith(".") for part in Path(path).parts):
            raise HTTPException(status_code=404, detail="file not found")
        return await super().get_response(path, scope)


def _load_json(path: Path) -> Any | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, data: Any, indent: int = 1) -> None:
    """Write via a temp file + replace: a torn write would fail to parse and read back as empty."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=indent))
    tmp.replace(path)


def _data_dir_at_depth(rel: str, depth: int, label: str) -> Path:
    parts = Path(rel).parts
    if len(parts) != depth or any(part.startswith(".") for part in parts):
        raise HTTPException(status_code=400, detail=f"invalid {label} path")
    p = (DATA_DIR / rel).resolve()
    if not p.is_relative_to(DATA_DIR.resolve()):
        raise HTTPException(status_code=400, detail="path escapes data dir")
    if not p.is_dir():
        raise HTTPException(status_code=404, detail=f"{label} not found")
    return p


def _natural_key(name: str) -> list:
    # "airline-tau2-10" -> ["airline-tau", 2, "-", 10]: numbers compare numerically
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def _scenario_dirs(run_dir: Path) -> list[Path]:
    out = []
    for child in sorted(run_dir.iterdir(), key=lambda p: _natural_key(p.name)):
        if child.is_dir() and ((child / "scenario.json").exists() or (child / "conversation.json").exists()):
            out.append(child)
    return out


def _parse_timestamp(run_name: str) -> str | None:
    m = TS_RE.search(run_name)
    if not m:
        return None
    t = m.group(1)  # 20260819T061543Z
    return f"{t[0:4]}-{t[4:6]}-{t[6:8]} {t[9:11]}:{t[11:13]}:{t[13:15]}Z"


def _is_text_module(module: str) -> bool:
    # text_bridge drives a voice agent class over text, so it is a text run despite the package name
    return any(k in module for k in ("text_agent", "TextAgent", "text_bridge", "TextBridge"))


def _run_modality(results: dict[str, Any] | None, scenarios: list[Path]) -> str | None:
    """"text" or "voice" (None when a run has no scenarios to judge by).

    The agent module is authoritative; runs from older harnesses carry no
    run_context, so fall back to whether the first scenarios recorded audio.
    """
    module = (((results or {}).get("run_context") or {}).get("agent_config") or {}).get("module") or ""
    if _is_text_module(module):
        return "text"
    if "voice_agent" in module or "TalkdeskAgent" in module:
        return "voice"
    for sdir in scenarios[:3]:
        if any((sdir / f"{name}.wav").exists() for name in ("conversation", "user", "agent")):
            return "voice"
    return "text" if scenarios else None


def _run_user_simulator(run_dir: Path) -> dict[str, Any] | None:
    """user_simulator.json, written by tau2_user_sim/launch.py into runs whose voice
    scenarios used tau2-bench's native user simulator (absent = talk-bench's own)."""
    marker = _load_json(run_dir / "user_simulator.json")
    return marker if isinstance(marker, dict) and marker.get("engine") else None


@cache
def build_index() -> list[dict[str, Any]]:
    global _archived_run_context
    runs = []
    contexts: dict[str, dict[str, Any]] = {}
    if not DATA_DIR.exists():
        return runs
    for run_dir in sorted(DATA_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        results = _load_json(run_dir / "results.json")
        eval_res = _load_json(run_dir / "talk_bench_evaluation_result.json")
        scenarios = _scenario_dirs(run_dir)
        entry: dict[str, Any] = {
            "path": run_dir.name,
            "run_name": run_dir.name,
            "timestamp": _parse_timestamp(run_dir.name),
            "scenario_count": len(scenarios),
            "has_results": results is not None,
            "modality": _run_modality(results, scenarios),
            "user_simulator": (_run_user_simulator(run_dir) or {}).get("engine"),
        }
        if results:
            entry.update(
                agent_name=results.get("agent_name"),
                primary_score=results.get("primary_score"),
                scenarios_passed=results.get("scenarios_passed"),
                scenarios_total=results.get("scenarios_total"),
                scenarios_errored=results.get("scenarios_errored"),
                domains=sorted((results.get("per_domain") or {}).keys()),
            )
            run_context = results.get("run_context")
            if isinstance(run_context, dict) and isinstance(run_context.get("agent_config"), dict):
                contexts[run_dir.name] = {
                    "agent": run_context["agent_config"],
                    "domain": run_context.get("domain"),
                    "split": run_context.get("split"),
                    # tau2-sourced scenarios are named <domain>-tau2-<n>
                    "source": "tau2" if any("-tau2-" in s.name for s in scenarios[:5]) else "talk_bench",
                }
        else:
            # fall back to parsing the agent name out of the directory name
            entry["agent_name"] = run_dir.name.split("-1trials-")[0]
        if eval_res:
            entry["eval_final"] = eval_res.get("final")
        runs.append(entry)
    runs.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    _archived_run_context = contexts
    return runs


STARS_FILE = DATA_DIR / ".stars.json"
_stars_lock = threading.Lock()
RUN_NOTES_FILE = DATA_DIR / ".run_notes.json"
MAX_RUN_NOTE_LENGTH = 2000
_run_notes_lock = threading.Lock()


def _load_stars() -> set[str]:
    stars = _load_json(STARS_FILE)
    return set(stars) if isinstance(stars, list) else set()


def _load_run_notes() -> dict[str, str]:
    notes = _load_json(RUN_NOTES_FILE)
    if not isinstance(notes, dict):
        return {}
    return {
        path: note
        for path, note in notes.items()
        if isinstance(path, str) and isinstance(note, str) and note
    }


@app.get("/api/runs")
def api_runs():
    return {
        "runs": build_index(),
        "stars": sorted(_load_stars()),
        "notes": _load_run_notes(),
        "llm_model": LLM_MODEL,
    }


@app.post("/api/star")
def api_star(path: str = Form(...), starred: bool = Form(...)):
    _data_dir_at_depth(path, 1, "run")
    with _stars_lock:
        stars = _load_stars()
        if starred:
            stars.add(path)
        else:
            stars.discard(path)
        _write_json(STARS_FILE, sorted(stars))
    return {"path": path, "starred": starred}


@app.post("/api/note")
def api_note(path: str = Form(...), note: str = Form("")):
    _data_dir_at_depth(path, 1, "run")
    note = note.strip()
    if len(note) > MAX_RUN_NOTE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"note must be at most {MAX_RUN_NOTE_LENGTH} characters",
        )
    with _run_notes_lock:
        notes = _load_run_notes()
        if note:
            notes[path] = note
        else:
            notes.pop(path, None)
        _write_json(RUN_NOTES_FILE, dict(sorted(notes.items())), indent=2)
    return {"path": path, "note": note}


@app.get("/api/run")
def api_run(path: str = Query(...)):
    return _run_payload(path, _data_dir_at_depth(path, 1, "run"))


# run detail payloads are costly to build; keep a few, cleared on data changes (_invalidate_index)
@lru_cache(maxsize=16)
def _run_payload(path: str, run_dir: Path) -> dict[str, Any]:
    results = _load_json(run_dir / "results.json")
    eval_res = _load_json(run_dir / "talk_bench_evaluation_result.json")

    per_scenario = {s.get("scenario_id"): s for s in (results or {}).get("per_scenario") or []}
    scenarios = []
    for sdir in _scenario_dirs(run_dir):
        summary = per_scenario.get(sdir.name) or {}
        end_reason = summary.get("end_reason")
        duration_ms = summary.get("total_duration_ms")
        if end_reason is None or duration_ms is None:
            # run_meta.json embeds the full db_state and is often several MB, so
            # only fall back to it for scenarios results.json doesn't cover
            meta = _run_meta(sdir)
            end_reason = end_reason or meta.get("end_reason")
            duration_ms = duration_ms or meta.get("total_duration_ms")
        scenarios.append({
            "id": sdir.name,
            "passed": summary.get("passed"),
            "goal_score": summary.get("goal_score"),
            "tau2_breakdown": summary.get("tau2_reward_breakdown"),
            "end_reason": end_reason,
            "turn_count": summary.get("turn_count"),
            "duration_ms": duration_ms,
        })

    return {
        "path": path, "results": results, "evaluation": eval_res, "scenarios": scenarios,
        "user_simulator": _run_user_simulator(run_dir),
    }


EVAL_METRICS = (
    "final", "resolution", "experience", "responsiveness",
    "customer_effort", "conversation_quality", "voice_consistency",
)


def _compare_side(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Headline numbers for one side of a comparison, plus its per-scenario map."""
    run_dir = _data_dir_at_depth(path, 1, "run")
    results = _load_json(run_dir / "results.json") or {}
    eval_res = _load_json(run_dir / "talk_bench_evaluation_result.json") or {}
    scenarios = _scenario_dirs(run_dir)
    side = {
        "path": path,
        "agent_name": results.get("agent_name"),
        "modality": _run_modality(results, scenarios),
        "domains": sorted((results.get("per_domain") or {}).keys()),
        "evaluation": {k: eval_res.get(k) for k in EVAL_METRICS} if eval_res else None,
    }
    per = {
        s.get("scenario_id"): s
        for s in results.get("per_scenario") or []
        if s.get("scenario_id")
    }
    if not per:  # no results.json — still let the scenario dirs line up
        per = {sdir.name: {} for sdir in scenarios}
    return side, per


def _mean(values) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(nums) / len(nums) if nums else None


def _compare_stats(per: dict[str, Any], ids: list[str]) -> dict[str, Any]:
    """Averages over the *shared* scenarios only, so both sides cover the same tasks."""
    rows = [per[i] for i in ids]
    tools = [(r.get("tool_metrics") or {}) for r in rows]
    return {
        "passed": sum(1 for r in rows if r.get("passed")),
        "total": len(rows),
        "goal_score": _mean(r.get("goal_score") for r in rows),
        "turn_count": _mean(r.get("turn_count") for r in rows),
        "duration_ms": _mean(r.get("total_duration_ms") for r in rows),
        "tool_recall": _mean(t.get("recall") for t in tools),
        "tool_precision": _mean(t.get("precision") for t in tools),
        "tool_calls": _mean(t.get("total_calls") for t in tools),
        "end_reasons": dict(Counter(r.get("end_reason") for r in rows if r.get("end_reason")).most_common()),
    }


def _compare_cell(s: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": s.get("passed"), "goal_score": s.get("goal_score"),
        "end_reason": s.get("end_reason"), "turn_count": s.get("turn_count"),
        "duration_ms": s.get("total_duration_ms"),
    }


@app.get("/api/compare")
def api_compare(a: str = Query(...), b: str = Query(...)):
    """Align two runs by scenario id — the point of comparison for text vs voice."""
    if a == b:
        raise HTTPException(status_code=400, detail="pick two different runs")
    side_a, per_a = _compare_side(a)
    side_b, per_b = _compare_side(b)
    shared = sorted(set(per_a) & set(per_b), key=_natural_key)
    scenarios = [{"id": sid, "a": _compare_cell(per_a[sid]), "b": _compare_cell(per_b[sid])} for sid in shared]
    counts = dict.fromkeys(("both_pass", "a_only", "b_only", "both_fail"), 0)
    for row in scenarios:
        pa, pb = bool(row["a"]["passed"]), bool(row["b"]["passed"])
        counts["both_pass" if pa and pb else "a_only" if pa else "b_only" if pb else "both_fail"] += 1
    side_a["stats"] = _compare_stats(per_a, shared)
    side_b["stats"] = _compare_stats(per_b, shared)
    return {
        "a": side_a,
        "b": side_b,
        "scenarios": scenarios,
        "counts": counts,
        "only_in_a": sorted(set(per_a) - set(per_b), key=_natural_key),
        "only_in_b": sorted(set(per_b) - set(per_a), key=_natural_key),
    }


# ---- transcript from records.json ----
# A Talkdesk Autopilot scenario saves the platform's own log of the call as records.json: every
# message between the caller (USER) and the agent's parts (DISPATCH, COORDINATOR, ACTION_AGENT,
# RESPOND, …) and every tool call and execution, in order. conversation.json merges speech by the
# voice pipeline's turn-taking instead, so several messages can land in one turn there; the
# scenario page lays the transcript out from records.json when a scenario has one.

def _utc_seconds(stamp: Any) -> float | None:
    """A records.json created_utc ("2026-10-12T03:47:15.318537923": UTC, to the ns) as epoch s."""
    m = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z?", stamp) if isinstance(stamp, str) else None
    if not m:
        return None
    return datetime.strptime(m[1], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() + float(m[2] or 0)


def _maybe_json(value: Any) -> Any:
    """Tool arguments and results arrive as JSON text: parsed when they parse."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def _record_tool(name: Any, arguments: Any, execution: dict | None, by: Any, ts: float | None) -> dict:
    ex = execution or {}
    return {
        "kind": "tool",
        "tool_name": name,
        "arguments": _maybe_json(arguments),
        "result": _maybe_json(ex.get("result")),
        "success": ex.get("success"),
        "duration_ms": ex.get("duration_millis"),
        "by": _as_dict(by).get("name"),
        "timestamp": ts,
    }


def _records_transcript(records: Any, tool_log: Any) -> list[dict] | None:
    """records.json as transcript items, in its order: a "turn" per message from or to the caller,
    a "note" per message between the agent's parts, a "tool" per call with its execution. Calls in
    tool_log.json that records.json lacks (it can miss some) are slotted in by time. None when
    records.json holds no message from or to the caller."""
    if not isinstance(records, list):
        return None
    records = [r for r in records if isinstance(r, dict)]
    executions = {
        ex["tool_call_ref_id"]: ex for r in records
        if isinstance(ex := r.get("tool_execution"), dict) and ex.get("tool_call_ref_id")
    }
    paired: set[str] = set()
    items: list[dict] = []
    for r in records:
        ts = _utc_seconds(r.get("created_utc"))
        msg, call, ex = r.get("message"), r.get("tool_call"), r.get("tool_execution")
        if isinstance(msg, dict):
            src, dst = str(msg.get("from") or ""), str(msg.get("to") or "")
            item: dict[str, Any] = {"text": str(msg.get("msg") or ""), "timestamp": ts, "action": r.get("llm_action")}
            if src == "USER" or dst == "USER":
                item.update(kind="turn", speaker="user" if src == "USER" else "agent")
                if src not in ("USER", "RESPOND", "COORDINATOR"):
                    item["by"] = src  # e.g. OUTPUT_GUARDRAIL
            else:
                # agent names read better than roles (COORDINATOR is "Meridian Text Supervisor")
                item.update(kind="note", src=_as_dict(msg.get("from_agent")).get("name") or src,
                            dst=_as_dict(msg.get("to_agent")).get("name") or dst)
            exit_ = r.get("exit")
            if isinstance(exit_, dict) and exit_.get("type") not in (None, "ok"):
                item["exit"] = {"type": exit_.get("type"), "message": exit_.get("error_message") or ""}
            items.append(item)
        elif isinstance(call, dict):
            ref = call.get("tool_call_ref_id")
            if ref in executions:
                paired.add(ref)
            items.append(_record_tool(call.get("tool_name"), call.get("arguments"), executions.get(ref),
                                      call.get("to_be_executed_by"), ts))
        elif isinstance(ex, dict) and ex.get("tool_call_ref_id") not in paired:
            items.append(_record_tool(_as_dict(ex.get("tool")).get("tool_name"), None, ex, ex.get("executed_by"), ts))
    if not any(it["kind"] == "turn" for it in items):
        return None

    # match tool_log.json's calls to records.json's in order (same tool, within 15 s): a match adds
    # db_changed and the call's place in tool_log.json, the rest go in before the first record logged after them
    unmatched = [it for it in items if it["kind"] == "tool"]
    extra = []
    for log_index, call in enumerate(tool_log if isinstance(tool_log, list) else []):
        if not isinstance(call, dict):
            continue
        ts = call.get("timestamp") if isinstance(call.get("timestamp"), (int, float)) else None
        match = next((it for it in unmatched if it["tool_name"] == call.get("tool_name") and (
            ts is None or it["timestamp"] is None or abs(it["timestamp"] - ts) <= 15)), None)
        if match:
            unmatched.remove(match)
            match.update(db_changed=bool(call.get("db_changed")), log_index=log_index)
        else:
            extra.append({
                "kind": "tool", "tool_name": call.get("tool_name"), "arguments": call.get("arguments"),
                "result": call.get("result"), "db_changed": bool(call.get("db_changed")),
                "timestamp": ts, "only_in_tool_log": True, "log_index": log_index,
            })
    for x in sorted(extra, key=lambda x: x["timestamp"] if x["timestamp"] is not None else float("inf")):
        at = next((i for i, it in enumerate(items) if x["timestamp"] is not None
                   and it["timestamp"] is not None and it["timestamp"] > x["timestamp"]), len(items))
        items.insert(at, x)
    return items


def _transcript_texts(conversation: Any, transcript: list[dict] | None) -> list[str] | None:
    """The turn texts the scenario page shows, in order: records.json's when it has them."""
    if transcript is not None:
        return [it["text"] for it in transcript if it["kind"] == "turn"]
    if not isinstance(conversation, list):
        return None
    return [str(t.get("text") or "") if isinstance(t, dict) else "" for t in conversation]


# ---- transcript ↔ audio alignment ----
# conversation.json stamps each turn with the wall clock at which it was finalized: the end of the
# user's speech, or ~1.5 s after the agent's audio ends (when the ASR returns its final text). The
# wavs start at the AudioBus's first recorded frame, which talk-bench takes right at the timeline's
# `greeting_wait_start` point. Each turn's speech is located on that clock from the user's latency
# markers (sample offsets into the wavs) and the VAD segments (monotonic ns, the timeline's clock).

def _read_ndjson(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        return []


def _audio_origin(sdir: Path) -> tuple[float, int] | None:
    """(wall-clock s, monotonic ns) of the wavs' first sample; None without a timeline."""
    try:
        with (sdir / "timeline.ndjson").open(encoding="utf-8") as f:
            header = json.loads(f.readline())
            for line in f:
                if '"greeting_wait_start"' in line:
                    ts_ns = json.loads(line)["ts_ns"]
                    return (header["start_wall_ns"] + ts_ns) / 1e9, header["start_ns"] + ts_ns
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _speech_segments(sdir: Path, origin_ns: int, rate: int) -> tuple[list, list, list]:
    """Speech on the wav clock (s): one (start, end) per sim utterance from the latency markers
    (end None while still open at hang-up), and the user's and agent's VAD segments."""
    markers, opened = [], {}
    for m in _read_ndjson(sdir / "latency_markers.ndjson"):
        if m.get("kind") == "user_speech_start":
            opened[m.get("turn_index")] = m["sample"] / rate
        elif m.get("kind") == "user_speech_end" and m.get("turn_index") in opened:
            markers.append((opened.pop(m.get("turn_index")), m["sample"] / rate))
    markers += [(start, None) for start in opened.values()]
    vad, started = {"user": [], "agent": []}, {}
    for e in _read_ndjson(sdir / "vad_events.ndjson"):
        who, t = e.get("speaker"), (e["timestamp_ns"] - origin_ns) / 1e9
        if e.get("type") == "speech_start":
            started[who] = t
        elif e.get("type") == "speech_end" and who in started and who in vad:
            vad[who].append((started.pop(who), t))
    return sorted(markers, key=lambda m: m[0]), sorted(vad["user"]), sorted(vad["agent"])


def _turn_spans(conversation: list, origin_s: float, markers: list, user_vad: list,
                agent_vad: list) -> dict[int, list[float]]:
    """{turn_index: [start, end]} on the wav clock for every turn whose speech was found.

    Both sources are consumed in order. A user turn takes the last unclaimed marker that started
    before the turn was finalized (one marker per sim utterance; earlier unclaimed ones are speech
    that never became a turn). An agent turn takes every unclaimed VAD segment that had ended by
    then: its sentences, plus any audio the ASR never transcribed."""
    spans, mi, ai = {}, 0, 0
    for turn in conversation:
        ts, idx = turn.get("timestamp"), turn.get("turn_index")
        if not isinstance(ts, (int, float)) or idx is None:
            continue
        final = ts - origin_s
        if turn.get("speaker") == "user":
            j = mi
            while j < len(markers) and markers[j][0] <= final + 0.05:
                j += 1
            if j == mi:
                continue
            start, end = markers[j - 1]
            mi = j
            # older talk-bench closed a marker at the first gap in the sim's audio; the VAD
            # shows where the speech really stopped
            ends = [e for s, e in user_vad if start <= s and e <= final + 0.3]
            spans[idx] = [round(start, 3), round(max([end or start, *ends]), 3)]
        else:
            j = ai
            while j < len(agent_vad) and agent_vad[j][1] <= final + 0.05:
                j += 1
            if j == ai:
                continue
            spans[idx] = [round(agent_vad[ai][0], 3), round(agent_vad[j - 1][1], 3)]
            ai = j
    return spans


def _record_spans(transcript: list[dict], origin_s: float, user_vad: list, agent_vad: list) -> None:
    """Set "span" [start, end] on the wav clock on each records.json turn whose speech was found.

    records.json times come from Talkdesk's clock, which tracks this machine's to well under a
    second. A caller message is logged once Talkdesk's ASR returns it: it claims every user VAD
    segment that had ended by then, and spans the last stretch of them without a 2 s pause (speech
    before one is an utterance the ASR never returned). An agent message is logged just before it
    is spoken and runs up to where the next one starts: from the first agent segment after it, or —
    while the agent is still voicing an earlier message, as messages queue for the TTS — from the
    pause nearest to where that message should end at this call's speaking rate."""
    turns = [it for it in transcript if it["kind"] == "turn" and isinstance(it.get("timestamp"), (int, float))]
    ui = 0
    for it in turns:
        if it["speaker"] != "user":
            continue
        t, j = it["timestamp"] - origin_s, ui
        while j < len(user_vad) and user_vad[j][1] <= t + 0.3:
            j += 1
        if j > ui:
            first = j - 1
            while first > ui and user_vad[first][0] - user_vad[first - 1][1] < 2:
                first -= 1
            it["span"] = [round(user_vad[first][0], 3), round(user_vad[j - 1][1], 3)]
            ui = j

    agents = [it for it in turns if it["speaker"] == "agent"]
    talking, last = 0.0, None  # the agent's speaking time, counting pauses between sentences (< 1 s)
    for s, e in agent_vad:
        talking += e - s + (s - last if last is not None and s - last < 1 else 0)
        last = e
    rate = min(max(sum(len(it["text"]) for it in agents) / talking, 10), 25) if talking else 15  # chars/s
    starts, ai, due = [], 0, None  # due: where the last placed message should stop speaking
    for it in agents:
        t = it["timestamp"] - origin_s
        later = [k for k in range(ai, len(agent_vad)) if agent_vad[k][0] >= t - 0.3]
        if not later:
            continue
        queued = due is not None and due > t and any(s < t - 0.3 and e > t - 1 for s, e in agent_vad)
        k = min(later, key=lambda k: abs(agent_vad[k][0] - due)) if queued else later[0]
        starts.append((it, k))
        ai, due = k + 1, agent_vad[k][0] + len(it["text"]) / rate
    for (it, k), (_, nxt) in zip(starts, starts[1:] + [(None, len(agent_vad))]):
        it["span"] = [round(agent_vad[k][0], 3), round(agent_vad[nxt - 1][1], 3)]


def _wav_header(path: Path) -> tuple[int, int] | None:
    """(frame rate, frame count) of a wav."""
    try:
        with wave.open(str(path)) as w:
            return w.getframerate(), w.getnframes()
    except (OSError, EOFError, wave.Error):
        return None


def _audio_alignment(sdir: Path, conversation: list | None, transcript: list[dict] | None = None) -> dict | None:
    """Where the transcript sits in the wavs: their wall-clock origin, length and per-turn speech
    spans — of conversation.json's turns, or set on the records.json transcript's turns when there
    is one. None for a text run or a run without a timeline."""
    origin = _audio_origin(sdir)
    if origin is None:
        return None
    header = next(filter(None, (_wav_header(sdir / f"{name}.wav") for name in ("conversation", "user", "agent"))), None)
    if header is None:
        return None
    rate, duration = header[0], header[1] / header[0]
    spans = {}
    try:
        markers, user_vad, agent_vad = _speech_segments(sdir, origin[1], rate)
        if transcript is not None:
            _record_spans(transcript, origin[0], user_vad, agent_vad)
        else:
            spans = _turn_spans(conversation or [], origin[0], markers, user_vad, agent_vad)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return {"origin": origin[0], "duration": round(duration, 3), "spans": spans}


def _scenario_results(sdir: Path) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """The parent run's results.json plus this scenario's entry in it and its evaluation
    session result (without ``turns``, which duplicate conversation.json)."""
    results = _load_json(sdir.parent / "results.json") or {}
    eval_res = _load_json(sdir.parent / "talk_bench_evaluation_result.json") or {}
    summary = next((s for s in results.get("per_scenario") or [] if s.get("scenario_id") == sdir.name), None)
    session = next((s for s in eval_res.get("session_results") or [] if s.get("session_id") == sdir.name), None)
    if session:
        session = {k: v for k, v in session.items() if k != "turns"}
    return results, summary, session


# ---- how each expected action was graded (Task & grading panel) ----
# talk-bench scores the action factor after the run, from tool_log.json: each agent-owned expected
# action is met when one of the agent's calls has its name, the compared arguments and, where asked
# for, a successful / non-empty / matching result (talk_bench_core.matching.compute_action_verdict).
# The verdict comes from that very function, imported from the local checkout; the expected-vs-
# actual rows next to it are rebuilt with the same comparison helpers, for the call that matched or,
# failing that, the one that came closest. No checkout (or a refactored one): no rows, and the panel
# falls back to "called / never called".

_MISSING = object()
_OK_STATUSES = {"ok", "success", "succeeded"}
_TYPE_NAMES = {str: "text", int: "integer", float: "decimal", bool: "boolean"}


@cache
def _tb_matching() -> SimpleNamespace | None:
    for repo in _source_repos():
        src = repo / "talk-bench" / "libs" / "talk_bench_core" / "src"
        if not (src / "talk_bench_core" / "matching.py").is_file():
            continue
        if str(src) not in sys.path:
            sys.path.append(str(src))
        try:
            from talk_bench_core import logical_actions, matching, value_normalization
            from talk_bench_core.criteria import Action
            from talk_bench_core.tools import ToolCallRecord
            return SimpleNamespace(
                Action=Action, ToolCallRecord=ToolCallRecord,
                verdict=matching.compute_action_verdict,
                record_matches=matching._record_matches_action,
                args_match=matching._args_match,                    # plain actions: case-insensitive
                value_equal=value_normalization.criteria_value_equal,  # actions using earlier results
                result_payload=matching._tool_result_payload,
                has_records=matching._result_has_records,
                projection_match=matching._result_projection_match,
                resolve_path=logical_actions.resolve_action_result_path,
            )
        except Exception:
            return None
    return None


def _ref_of(value: Any) -> tuple[str, str] | None:
    """(action_id, path) of an argument written as {"$action_result": {action_id, path}}."""
    ref = value.get("$action_result") if isinstance(value, dict) and len(value) == 1 else None
    return (ref.get("action_id"), ref.get("path")) if isinstance(ref, dict) else None


def _has_ref(value: Any) -> bool:
    if isinstance(value, dict):
        return "$action_result" in value or any(_has_ref(v) for v in value.values())
    return isinstance(value, list) and any(_has_ref(v) for v in value)


def _compared_args(action: Any) -> dict[str, Any] | None:
    """The arguments an action compares: compare_args' keys, else all of them. With compare_args,
    plain actions skip a key the action has no value for; actions using earlier results fail."""
    args = action.arguments or {}
    if action.compare_args is None:
        return dict(args)
    return {k: args[k] for k in action.compare_args if k in args}


def _type_note(actual: Any, expected: Any, verb: str) -> str | None:
    """JSON shows 36.0 as 36: name the types when only they differ."""
    same_text = lambda v: str(v).casefold().removesuffix(".0")
    if type(actual) is type(expected) or same_text(actual) != same_text(expected):
        return None
    return f"{_TYPE_NAMES.get(type(expected), 'value')} {json.dumps(expected)} expected, " \
           f"{_TYPE_NAMES.get(type(actual), 'value')} {json.dumps(actual)} {verb}"


def _projection(actual: Any, expected: Any) -> Any:
    """`actual` cut down to the shape of `expected`, so a big result shows only what was asked for."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return {k: _projection(actual[k], v) for k, v in expected.items() if k in actual}
    if isinstance(expected, list) and expected and isinstance(actual, list):
        return [_projection(a, expected[0]) for a in actual[:20]]
    return actual


def _projection_misses(tb: SimpleNamespace, actual: Any, expected: Any, path: str) -> list[dict]:
    """Where a nested expected_result value fails: the leaves (or list items) the result lacks,
    walked the way talk-bench matches it (dicts by key, each list item against a distinct item)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for k, v in expected.items():
            sub = f"{path}.{k}"
            out += (_projection_misses(tb, actual[k], v, sub) if k in actual
                    else [{"path": sub, "expected": v, "note": "not in the result"}])
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        unmatched, out = list(actual), []
        for n, item in enumerate(expected):
            hit = next((j for j, a in enumerate(unmatched) if tb.projection_match(a, item)), None)
            if hit is not None:
                unmatched.pop(hit)
                continue
            # not found: name the fields that differ in the returned item closest to it
            sub = f"{path}[{n}]"
            near = max((a for a in unmatched if isinstance(a, dict)), default=None, key=lambda a: sum(
                k in a and tb.projection_match(a[k], v) for k, v in item.items())) if isinstance(item, dict) else None
            if near is not None and any(k in near and tb.projection_match(near[k], v) for k, v in item.items()):
                closest = "the only one returned" if len(actual) == 1 else f"closest of the {len(actual)} returned"
                out += [{**m, "note": m.get("note") or closest}
                        for m in _projection_misses(tb, near, item, sub)]
            else:
                out.append({"path": sub, "expected": item, "note": f"no match among the {len(actual)} returned"})
        return out
    if tb.projection_match(actual, expected):
        return []
    return [{"path": path, "expected": expected, "actual": _projection(actual, expected)}]


def _compare_call(tb: SimpleNamespace, action: Any, record: Any, logical: bool,
                  resolve: Any) -> tuple[list[dict], list[dict]]:
    """Expected vs actual for one call: argument rows, then result rows (one per gate asked for)."""
    actual_args = record.arguments or {}
    rows = []
    compared = _compared_args(action)
    if logical and action.compare_args is not None and len(compared) < len(action.compare_args):
        rows += [{"key": k, "actual": actual_args.get(k), "ok": False, "note": "no expected value"}
                 for k in action.compare_args if k not in compared]
    for key, expected in compared.items():
        row: dict[str, Any] = {"key": key}
        if (ref := _ref_of(expected)) or _has_ref(expected):
            row["ref"] = {"action_id": ref[0], "path": ref[1]} if ref else None
            expected = resolve(expected)
        if expected is _MISSING:
            row.update(actual=actual_args.get(key), ok=False, note="value from an earlier result not available")
        else:
            row["expected"] = expected
            if key not in actual_args:
                row.update(ok=False, note="argument not passed")
            else:
                actual = row["actual"] = actual_args[key]
                row["ok"] = bool(tb.value_equal(actual, expected) if logical
                                 else tb.args_match({key: actual}, {key: expected}))
                if not row["ok"] and (note := _type_note(actual, expected, "passed")):
                    row["note"] = note
        rows.append(row)

    checks: list[dict] = []
    if not (action.require_successful_result or action.require_nonempty_result or action.expected_result):
        return rows, checks
    payload, error = tb.result_payload(record.result)
    if error or payload is None:
        msg = error.get("message") if isinstance(error, dict) else error
        return rows, [{"check": "error", "actual": msg or "no result", "ok": False}]
    if action.require_successful_result and isinstance(payload, dict):
        status, success = payload.get("status"), payload.get("success")
        checks.append({
            "check": "status", "expected": "success",
            "actual": status if status is not None else success,
            "ok": (status is None or str(status).strip().casefold() in _OK_STATUSES) and success is not False,
        })
    if action.require_nonempty_result:
        count = payload.get("record_count") if isinstance(payload, dict) else None
        if not isinstance(count, (int, float)) or isinstance(count, bool):
            records = payload.get("records") if isinstance(payload, dict) else payload
            count = (sum(len(v) if isinstance(v, (list, dict)) else bool(v) for v in records.values())
                     if isinstance(records, dict) else len(records) if isinstance(records, (list, str)) else None)
        checks.append({"check": "records", "actual": count, "ok": bool(tb.has_records(payload))})
    for key, expected in (action.expected_result or {}).items():
        has = isinstance(payload, dict) and key in payload
        check = {"check": "field", "key": key, "expected": expected,
                 "ok": bool(has and tb.projection_match(payload[key], expected))}
        if not has:
            check["note"] = "not in the result"
        else:
            check["actual"] = _projection(payload[key], expected)
            if not check["ok"] and isinstance(expected, (dict, list)):
                check["misses"] = _projection_misses(tb, payload[key], expected, key)[:12]
        checks.append(check)
    return rows, checks


def _action_checks(scenario: Any, tool_log: Any) -> dict | None:
    """Per expected / prohibited action of scenario.json: talk-bench's verdict, the call it was
    judged on and the expected-vs-actual comparison. Lists line up with scenario.json's."""
    ev = _as_dict(_as_dict(scenario).get("evaluation"))
    raw_expected = ev.get("expected_actions") or []
    raw_prohibited = ev.get("prohibited_actions") or []
    tb = _tb_matching() if raw_expected or raw_prohibited else None
    if tb is None or not (tool_log is None or isinstance(tool_log, list)):
        return None
    try:
        expected = [tb.Action.model_validate(a) for a in raw_expected]
        prohibited = [tb.Action.model_validate(a) for a in raw_prohibited]
        records = [tb.ToolCallRecord.model_validate(r) for r in tool_log or []]
        met, violated, _ = tb.verdict(expected, prohibited, records, enforce_order=bool(ev.get("enforce_order")))
    except Exception:
        return None

    # the verdict only looks at the agent's own actions and calls
    is_agent = lambda a: a.requestor in (None, "assistant")
    agent = [i for i, a in enumerate(expected) if is_agent(a)]
    met_at = dict(zip(agent, met))
    log = [(i, r) for i, r in enumerate(records) if r.requestor == "assistant"]
    ids = Counter(expected[i].action_id for i in agent if expected[i].action_id)
    by_id = {expected[i].action_id: i for i in agent if expected[i].action_id and ids[expected[i].action_id] == 1}
    deps = {i: [by_id[d] for d in dict.fromkeys(expected[i].prerequisite_action_ids) if d in by_id] for i in agent}

    def ancestors(i: int, seen: set[int]) -> set[int]:
        for j in deps[i]:
            if j not in seen:
                seen.add(j)
                ancestors(j, seen)
        return seen

    # Once an action takes a value from an earlier action's result, talk-bench matches its whole
    # chain (it and its prerequisites, transitively) with normalized values instead: formatting-
    # tolerant text ("913-555-0148" = "9135550148") but type-strict numbers (36 ≠ 36.0). Actions
    # outside such a chain keep plain matching (case-insensitive, None/"" as "null"). A prerequisite
    # can therefore miss its own credit and still count as matched for the action that uses it.
    with_ref = {i for i in agent if _has_ref(_compared_args(expected[i]))}
    normalized = {i for i in agent if i in with_ref or ancestors(i, set()) & with_ref}
    chain = normalized.union(*(ancestors(i, set()) for i in normalized))

    def call_info(pos: int) -> dict:
        raw, rec = log[pos]
        same = [p for p, (_, r) in enumerate(log) if r.tool_name == rec.tool_name]
        return {"log_index": raw, "n": same.index(pos) + 1, "of": len(same), "timestamp": rec.timestamp or None}

    # producers before consumers, so a "$action_result" reference resolves against the call chosen
    order, done = [], set()
    while len(order) < len(agent):
        ready = [i for i in agent if i not in done and all(j in done or j == i for j in deps[i])]
        ready = ready or [next(i for i in agent if i not in done)]  # a cycle: take the rest as listed
        order += ready
        done.update(ready)

    chosen: dict[int, int] = {}   # expected index -> position in `log` of the call it matched
    source: dict[int, int] = {}   # the same under normalized matching, for actions in a chain
    used: dict[str, set[int]] = {}  # calls already credited to an earlier action of the same name
    name_count = Counter(expected[i].name for i in agent)

    def resolve(value: Any) -> Any:
        if (ref := _ref_of(value)) is not None:
            j = by_id.get(ref[0])
            if j is None or j not in source:
                return _MISSING
            payload, _ = tb.result_payload(log[source[j]][1].result)
            return tb.resolve_path(payload, ref[1], default=_MISSING)
        if isinstance(value, dict):
            out = {k: resolve(v) for k, v in value.items()}
            return _MISSING if any(v is _MISSING for v in out.values()) else out
        if isinstance(value, list):
            out = [resolve(v) for v in value]
            return _MISSING if any(v is _MISSING for v in out) else out
        return value

    def misses(action: Any, pos: int, norm: bool) -> tuple[list[dict], list[dict], int]:
        rows, checks = _compare_call(tb, action, log[pos][1], norm, resolve)
        return rows, checks, sum(not r["ok"] for r in rows + checks)

    out_expected: list[dict | None] = [None] * len(expected)
    for i in order:
        a, norm = expected[i], i in normalized
        matched_at = source if norm else chosen
        candidates = [p for p, (_, r) in enumerate(log) if r.tool_name == a.name]
        if a.last_call_only:
            candidates = candidates[-1:]
        earliest = max((matched_at[j] + 1 for j in deps[i] if j in matched_at), default=0)
        taken = used.get(a.name, set()) if name_count[a.name] > 1 else set()
        # the call shown: one that passes every comparison (after the prerequisites' calls and not
        # credited to another action already, when there is one), else the one with fewest misses
        compared = [(p, *misses(a, p, norm)) for p in candidates]
        best = min(compared, key=lambda c: (c[3] > 0, c[3], c[0] < earliest, c[0] in taken, c[0]), default=None)
        entry: dict[str, Any] = {"met": met_at[i], "mode": "normalized" if norm else "plain"}
        if best is None:
            entry["status"] = "missing"
        else:
            p, rows, checks, fails = best
            entry.update(call=call_info(p), args=rows, result=checks)
            if fails == 0 and met_at[i]:
                matched_at[i] = p
                used.setdefault(a.name, set()).add(p)
            if met_at[i]:
                entry["status"] = "ok" if fails == 0 else "ok_unshown"
            elif fails:
                entry["status"] = "args" if any(not r["ok"] for r in rows) else "result"
            elif unmet := [expected[j].action_id for j in deps[i] if j not in matched_at]:
                entry.update(status="prereq", prereq=unmet)
            else:  # matches, but not after its prerequisites' calls (or only on a call another took)
                entry.update(status="order", after=[expected[j].action_id for j in deps[i]],
                             early=p < earliest)
        if i in chain and not norm:
            # how the chain sees this prerequisite: the first call passing normalized matching
            first = max((source[j] + 1 for j in deps[i] if j in source), default=0)
            p = next((p for p in candidates if p >= first and misses(a, p, True)[2] == 0), None)
            if not all(j in source for j in deps[i]):
                p = None
            if p is not None:
                source[i] = p
                if not met_at[i]:
                    entry["source_for_later"] = True
        out_expected[i] = entry

    out_prohibited: list[dict | None] = [None] * len(prohibited)
    for i, hit in zip([i for i, a in enumerate(prohibited) if is_agent(a)], violated):
        entry = {"violated": hit}
        if hit:
            p = next((p for p, (_, r) in enumerate(log)
                      if r.tool_name == prohibited[i].name and tb.record_matches(prohibited[i], r)), None)
            if p is not None:
                entry.update(call=call_info(p), arguments=log[p][1].arguments)
        out_prohibited[i] = entry
    return {"expected": out_expected, "prohibited": out_prohibited, "met": sum(met), "total": len(met)}


# ---- how each expected DB state was graded (Task & grading panel) ----
# talk-bench scores the DB factor after the run from the final DB snapshot (run_meta.json's db_state)
# and, for checks on what changed, the one taken before the call (db_state_initial.json).
# scenario.json's expected_state comes in two shapes:
#   * a field map {collection: {entity id: {field: value}}}, scored as the share of equal fields;
#   * assertions (schema_version 2-4), each all or nothing: its selector picks rows, its change
#     filter keeps those created / updated during the call, their number must fit its cardinality
#     and every one of them must have each expected field. Schema 4 first binds generated ids (one
#     row created during the call, picked by stable fields) for {"$ref": name} to stand for; a single
#     binding that does not resolve scores the whole factor 0. Schema 3 adds free-text fields an
#     LLM judge compared; the session result keeps those verdicts (db_semantic_checks).
# Selection and comparisons are talk-bench's own functions, loaded from the checkout; the rows shown
# beside a verdict are rebuilt field by field with the same comparison.

_DB_CHANGE = {
    "created": lambda before, key, row: key not in before,
    "updated": lambda before, key, row: key in before and before[key] != row,
    "changed": lambda before, key, row: key not in before or before[key] != row,
}


_TB_DB_LOCK = threading.Lock()  # two first requests at once: one lends the module name at a time


@cache
def _tb_db() -> SimpleNamespace | None:
    """talk-bench's DB-state scorer modules, loaded file by file: the talk_bench_evaluation
    package's __init__ imports its whole scoring stack (jiwer and more), which the viewer lacks."""
    if _tb_matching() is None:  # also puts talk_bench_core on sys.path
        return None
    with _TB_DB_LOCK:
        return _load_tb_db()


def _load_tb_db() -> SimpleNamespace | None:
    for repo in _source_repos():
        pkg = repo / "talk-bench" / "libs" / "talk_bench_evaluation" / "src" / "talk_bench_evaluation"
        if not (pkg / "scorers" / "db_state.py").is_file():
            continue

        def load(name: str, rel: str) -> Any:
            spec = importlib.util.spec_from_file_location(name, pkg / rel)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        # db_state.py imports this module by its package name: lend it that name while it loads
        alias = "talk_bench_evaluation.semantic_normalizer"
        lent = alias not in sys.modules
        try:
            if lent:
                sys.modules[alias] = load("_tb_semantic_normalizer", "semantic_normalizer.py")
            db = load("_tb_db_state", "scorers/db_state.py")
            lp = load("_tb_logical_projection", "scorers/logical_projection.py")
            return SimpleNamespace(
                entries=db._collection_entries, changed=lp._changed_entries,
                refs=lp._resolve_references, unresolved=lp._UNRESOLVED, resolve=lp.resolve_logical_projection,
                # schema 2 / 3: dotted paths (with __len__), type-exact numbers, normalized text
                v2=SimpleNamespace(get=db._resolve_path, missing=db._MISSING, equal=db._type_exact_equal,
                                   matches=db._matches_fields),
                # schema 4: the same, with recursive projections for nested values ($unordered lists)
                v4=SimpleNamespace(get=lp._resolve_path, missing=lp._MISSING, equal=lp.projection_value_equal,
                                   matches=lp.projection_matches),
                # the field map: plain ==
                fields=SimpleNamespace(get=db._resolve_path, missing=db._MISSING, equal=lambda a, e: a == e),
            )
        except Exception:
            return None
        finally:
            if lent:
                sys.modules.pop(alias, None)
    return None


def _db_key(key: str) -> str:
    """A row's key for display: a list collection's rows are keyed __list__:<index>."""
    return f"#{key.removeprefix('__list__:')}" if key.startswith("__list__:") else key


def _db_fields(cmp: SimpleNamespace, row: Any, fields: dict) -> list[dict]:
    """Expected vs stored per field of one row (dotted paths reach into nested values)."""
    out = []
    for path, expected in fields.items():
        actual = cmp.get(row, str(path))
        item: dict[str, Any] = {"key": str(path), "expected": expected}
        if actual is cmp.missing:
            item.update(ok=False, note="no such field")
        else:
            item.update(actual=_projection(actual, expected), ok=bool(cmp.equal(actual, expected)))
            if not item["ok"] and (note := _type_note(actual, expected, "stored")):
                item["note"] = note
        out.append(item)
    return out


def _db_assertion(tb: SimpleNamespace, cmp: SimpleNamespace, a: Any, final: dict, initial: dict | None) -> dict:
    """One assertion as talk-bench decides it, with the rows that decided it: those selected or,
    when none was, the row that came closest."""
    if not (isinstance(a, dict) and isinstance(a.get("collection"), str)
            and isinstance(a.get("selector"), dict) and isinstance(a.get("expected"), dict)):
        return {"met": False, "reason": "malformed"}
    selector, expected = a["selector"], a["expected"]
    rows = tb.entries(final, a["collection"])
    before = tb.entries(initial or {}, a["collection"])
    keep = _DB_CHANGE.get(a.get("change"))
    picked = {k: r for k, r in rows.items() if cmp.matches(r, selector)}
    matched = {k: r for k, r in picked.items() if keep is None or keep(before, k, r)}
    card, n = a.get("cardinality", "exactly_one"), a.get("expected_count")
    count_ok = {
        "exactly_one": len(matched) == 1,
        "at_least_one": len(matched) >= 1,
        "none": not matched,
        "exactly_n": isinstance(n, int) and not isinstance(n, bool) and n > 0 and len(matched) == n,
    }.get(card, False)
    fields_ok = (not expected if card == "none"
                 else bool(matched) and all(cmp.matches(r, expected) for r in matched.values()))
    met = count_ok and fields_ok
    reason = ("ok" if met else "count" if matched and not count_ok else "fields" if matched or card == "none"
              else "change" if picked else "no_rows")

    state = lambda k, r: (None if initial is None else "created" if k not in before
                          else "updated" if before[k] != r else "unchanged")
    out: dict[str, Any] = {
        "met": met, "reason": reason, "rows": len(rows), "picked": len(picked), "matched": len(matched),
        "matches": [{"key": _db_key(k), "state": state(k, r), "fields": _db_fields(cmp, r, expected)}
                    for k, r in list(matched.items())[:5]],
    }
    if not matched and card != "none" and selector and rows:
        # the closest row: of the selector's picks the change filter dropped, else the row that
        # agrees on most selector fields (on none of them: nothing worth showing)
        def closeness(item: tuple[str, dict]) -> tuple[int, bool, int]:
            k, r = item
            return (sum(x["ok"] for x in _db_fields(cmp, r, selector)), keep is None or keep(before, k, r),
                    sum(x["ok"] for x in _db_fields(cmp, r, expected)))
        k, r = max((picked or rows).items(), key=closeness)
        if picked or closeness((k, r))[0]:
            out["near"] = {"key": _db_key(k), "state": state(k, r), "fields": _db_fields(cmp, r, expected),
                           **({} if picked else {"selector": _db_fields(cmp, r, selector)})}
    return out


def _db_bindings(tb: SimpleNamespace, bindings: dict, final: dict, initial: dict | None) -> tuple[list[dict], dict]:
    """Schema 4's bindings resolved the way talk-bench resolves them (a binding may use earlier
    ones): per binding its rows and value or error, and the values that did resolve."""
    cmp, values, out = tb.v4, {}, {}
    pending = dict(bindings)
    while pending:
        progressed = False
        for name in list(pending):
            b = pending[name]
            selector = tb.refs(b.get("selector"), values) if isinstance(b, dict) else None
            if selector is tb.unresolved:
                continue
            pending.pop(name)
            progressed = True
            if not (isinstance(b, dict) and isinstance(b.get("collection"), str) and isinstance(b.get("field"), str)
                    and isinstance(selector, dict) and selector
                    and b.get("change") == "created" and b.get("cardinality") == "exactly_one"):
                out[name] = {"error": "invalid binding"}
                continue
            created = tb.changed(final, initial or {}, b["collection"], "created")
            hits = [k for k, r in created.items() if cmp.matches(r, selector)]
            info = {"collection": b["collection"], "selector": selector, "field": b["field"], "found": len(hits),
                    "rows": [_db_key(k) for k in hits[:5]]}
            if len(hits) != 1:
                # rows the selector picks that were there before the call do not count
                info["existing"] = sum(k not in created and cmp.matches(r, selector)
                                       for k, r in tb.entries(final, b["collection"]).items())
                out[name] = {**info, "error": f"expected exactly one created row, found {len(hits)}"}
                continue
            bound = cmp.get(created[hits[0]], b["field"])
            if bound is cmp.missing or isinstance(bound, (dict, list)):
                out[name] = {**info, "error": f"the row has no plain {b['field']} value"}
            else:
                values[name] = bound
                out[name] = {**info, "value": bound}
        if not progressed:
            out.update({name: {"error": "uses a binding that did not resolve"} for name in pending})
            break
    return [{"name": name, **out[name]} for name in bindings if name in out], values


def _db_checks(scenario: Any, final: Any, initial: Any, semantic: Any) -> dict | None:
    """Per expected_state entry of scenario.json: talk-bench's verdict and the DB rows behind it.
    Items line up with expected_state's assertions, or with its (collection, entity) pairs."""
    es = _as_dict(_as_dict(_as_dict(scenario).get("evaluation")).get("expected_state"))
    version = es.get("schema_version")
    tb = _tb_db() if es and version in (None, 2, 3, 4) else None
    if tb is None:
        return None
    final = final if isinstance(final, dict) else None
    initial = initial if isinstance(initial, dict) else None
    try:
        return (_db_assertion_checks(tb, es, final, initial, semantic) if version
                else _db_field_checks(tb, es, final))
    except Exception:
        return None


def _db_field_checks(tb: SimpleNamespace, es: dict, final: dict | None) -> dict:
    items = []
    for coll, entities in es.items():
        stored = (final or {}).get(coll) or {}
        for entity, fields in entities.items():
            row = stored.get(entity) if isinstance(stored, dict) else None
            checks = _db_fields(tb.fields, row or {}, fields)
            items.append({"met": all(c["ok"] for c in checks), "found": isinstance(row, dict), "fields": checks})
    total = sum(len(i["fields"]) for i in items)
    met = sum(c["ok"] for i in items for c in i["fields"])
    return {"format": "fields", "unit": "fields", "met": met, "total": total,
            "score": 0.0 if final is None else round(met / total, 4) if total else 1.0,
            "problem": "no_db" if final is None else None, "items": items}


def _db_assertion_checks(tb: SimpleNamespace, es: dict, final: dict | None, initial: dict | None,
                         semantic: Any) -> dict:
    version, assertions, bindings = es["schema_version"], es.get("assertions"), es.get("bindings")
    out: dict[str, Any] = {"format": "assertions", "unit": "assertions", "schema_version": version, "problem": None}
    if not isinstance(assertions, list):
        return {**out, "problem": "malformed", "met": 0, "total": 0, "score": 1.0 if version < 4 else 0.0, "items": []}
    out.update(total=len(assertions), items=[])
    changes = any(isinstance(a, dict) and a.get("change") for a in assertions)
    if final is None:
        out["problem"] = "no_db"
    elif version == 4 and not isinstance(bindings, dict):
        out["problem"] = "malformed"
    elif initial is None and (changes or version == 4 and bindings):
        out["problem"] = "no_baseline"  # talk-bench cannot score it: 0
    cmp = tb.v4 if version == 4 else tb.v2
    values: dict = {}
    if version == 4 and final is not None and isinstance(bindings, dict):
        if bindings:
            out["bindings"], values = _db_bindings(tb, bindings, final, initial)
        if not out["problem"] and tb.resolve(final, es, initial)[1]:
            out["problem"] = "binding"
    judged = {c.get("assertion_index"): c for c in semantic or [] if isinstance(c, dict)}

    for i, raw in enumerate(assertions):
        a = tb.refs(raw, values) if version == 4 else raw
        if final is None:
            item = {"met": False, "reason": "no_db"}
        elif a is tb.unresolved:
            item = {"met": False, "reason": "binding"}
        else:
            item = _db_assertion(tb, cmp, a, final, initial)
            sem = a.get("semantic_expected") if version == 3 and isinstance(a, dict) else None
            if isinstance(sem, dict) and sem:
                # free-text fields: the judge's stored verdict decides (no verdict: not met)
                c = _as_dict(judged.get(i))
                stored = next((f.get("fields") for f in c.get("actual_fields") or [] if isinstance(f, dict)), None) or {}
                item["semantic"] = [{"key": k, "expected": v, **({"actual": stored[k]} if k in stored else {}),
                                     "ok": c.get("met") is True} for k, v in sem.items()]
                item["semantic_why"] = c.get("justification") or "no judge verdict stored"
                if item["met"] and c.get("met") is not True:
                    item.update(met=False, reason="semantic")
            if out["problem"]:
                item.update(alone=item["met"], met=False)
        out["items"].append(item)
    out["met"] = sum(i["met"] for i in out["items"])
    out["score"] = 0.0 if out["problem"] else round(out["met"] / len(assertions), 4) if assertions else 1.0
    return out


def _run_meta(sdir: Path) -> dict[str, Any]:
    """run_meta.json without db_state (huge and rarely needed inline)."""
    meta = _load_json(sdir / "run_meta.json")
    meta = dict(meta) if isinstance(meta, dict) else {}
    meta.pop("db_state", None)
    return meta


@app.get("/api/scenario")
def api_scenario(path: str = Query(...)):
    sdir = _data_dir_at_depth(path, 2, "scenario")
    raw_meta = _load_json(sdir / "run_meta.json")
    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    final_db = meta.pop("db_state", None)  # huge: graded below, not sent

    # per-scenario scores live in the parent run's result files
    _, scenario_summary, session_result = _scenario_results(sdir)
    files = []
    for f in sorted(sdir.rglob("*")):
        relative = f.relative_to(sdir)
        if f.is_file() and not any(part.startswith(".") for part in relative.parts):
            files.append({
                "name": str(relative),
                "size": f.stat().st_size,
            })
    conversation = _load_json(sdir / "conversation.json")
    tool_log = _load_json(sdir / "tool_log.json")
    scenario = _load_json(sdir / "scenario.json")
    transcript = _records_transcript(_load_json(sdir / "records.json"), tool_log)
    return {
        "path": path,
        "scenario": scenario,
        "conversation": conversation,
        "tool_log": tool_log,
        "action_checks": _action_checks(scenario, tool_log),
        "db_checks": _db_checks(scenario, final_db, _load_json(sdir / "db_state_initial.json"),
                                _as_dict(_as_dict(session_result).get("task_completion")).get("db_semantic_checks")),
        "transcript": transcript,
        "run_meta": meta,
        "scenario_summary": scenario_summary,
        "session_result": session_result,
        "files": files,
        "audio": {
            name: (sdir / f"{name}.wav").exists()
            for name in ("conversation", "user", "agent")
        },
        "audio_align": _audio_alignment(sdir, conversation if isinstance(conversation, list) else None, transcript),
        "transcript_zh": _cached_translation(sdir, _transcript_texts(conversation, transcript)),
        "task_zh": _cached_task_translation(sdir),
    }


def _invalidate_index() -> None:
    build_index.cache_clear()
    _run_payload.cache_clear()


# ---- delete a run ----
# Deletes the archived copy only. A copy still in a checkout (one Sync keeps there because it can be
# resumed) is left alone, so the name goes into .deleted_runs.json and Sync skips it from then on;
# importing the run again (drag & drop, or a Run test job copying it) takes it back off.

DELETED_RUNS_FILE = DATA_DIR / ".deleted_runs.json"
_deleted_lock = threading.Lock()


def _load_deleted_runs() -> set[str]:
    names = _load_json(DELETED_RUNS_FILE)
    return set(names) if isinstance(names, list) else set()


def _undelete(name: str) -> None:
    with _deleted_lock:
        names = _load_deleted_runs()
        if name in names:
            names.discard(name)
            _write_json(DELETED_RUNS_FILE, sorted(names))


@app.delete("/api/run")
def api_delete_run(path: str = Query(...)):
    run_dir = _data_dir_at_depth(path, 1, "run")
    if any(j["status"] in ("running", "cancelling") and path in (j.get("run_path"), Path(j.get("run_dir") or "").name)
           for j in list(_jobs.values())):
        raise HTTPException(status_code=409, detail="a test is still writing this run — stop it first")
    trash = DATA_DIR / f".deleting-{uuid.uuid4().hex[:12]}"
    # under the lock Sync and job snapshots swap run dirs with; moved out of the index at once,
    # removed from disk afterwards
    with _mirror_lock:
        run_dir.rename(trash)
    shutil.rmtree(trash, ignore_errors=True)
    with _deleted_lock:
        _write_json(DELETED_RUNS_FILE, sorted(_load_deleted_runs() | {path}))
    with _stars_lock:
        stars = _load_stars()
        if path in stars:
            _write_json(STARS_FILE, sorted(stars - {path}))
    with _run_notes_lock:
        notes = _load_run_notes()
        if notes.pop(path, None) is not None:
            _write_json(RUN_NOTES_FILE, dict(sorted(notes.items())), indent=2)
    _invalidate_index()
    return {"deleted": path}


# ---- folder sync: begin -> chunk (repeated) -> commit ----
# staged uploads live in a temp dir until committed atomically into data/
_staged: dict[str, dict[str, Any]] = {}


def _staging_dir() -> Path:
    """Stage uploads inside data/ so the commit is a same-filesystem rename.
    With data/ on a network mount, staging under the system temp dir would make
    the move a cross-device copy that can leave a half-written run behind."""
    d = DATA_DIR / ".uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_staged(upload_id: str, pop: bool = False) -> dict[str, Any]:
    st = _staged.pop(upload_id, None) if pop else _staged.get(upload_id)
    if not st:
        raise HTTPException(status_code=404, detail="unknown or expired upload_id")
    return st


@app.post("/api/upload_dir_begin")
def upload_dir_begin(
    run_name: str = Form(...),
    overwrite: bool = Form(False),
):
    run_name = Path(run_name).name
    if not run_name or run_name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid run name")
    target = DATA_DIR / run_name
    if target.exists() and not overwrite:
        raise HTTPException(status_code=409, detail=f"run '{run_name}' already exists (enable overwrite)")
    upload_id = uuid.uuid4().hex
    _staged[upload_id] = {
        "dir": Path(tempfile.mkdtemp(prefix="talkbench_upload_", dir=_staging_dir())),
        "run_name": run_name,
        "files": 0,
    }
    return {"upload_id": upload_id}


@app.post("/api/upload_dir_chunk")
def upload_dir_chunk(
    upload_id: str = Form(...),
    paths: list[str] = Form(...),
    files: list[UploadFile] = File(...),
):
    st = _get_staged(upload_id)
    if len(paths) != len(files):
        raise HTTPException(status_code=400, detail="paths/files count mismatch")
    staging: Path = st["dir"]
    for rel, f in zip(paths, files):
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts or not p.parts:
            raise HTTPException(status_code=400, detail=f"unsafe path: {rel}")
        dest = staging / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        st["files"] += 1
    return {"received": st["files"]}


@app.post("/api/upload_dir_commit")
def upload_dir_commit(upload_id: str = Form(...)):
    st = _get_staged(upload_id, pop=True)
    target = DATA_DIR / st["run_name"]
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(st["dir"]), str(target))
    _undelete(st["run_name"])
    _invalidate_index()
    return {"run": st["run_name"], "files": st["files"]}


@app.post("/api/upload_dir_abort")
def upload_dir_abort(upload_id: str = Form(...)):
    st = _staged.pop(upload_id, None)
    if st:
        shutil.rmtree(st["dir"], ignore_errors=True)
    return {"aborted": bool(st)}


# ---- sync new runs from the local ai-ds-research checkouts ----

# The checkout sits in a different place on each machine, so scan every known
# location and use the ones that exist. VIEWER_SOURCE_REPOS (":"-separated
# paths) replaces the list without a code change.
SOURCE_REPO_CANDIDATES = (
    Path.home() / "GitHub" / "ai-ds-research",
    Path.home() / "Documents" / "tau2-bench-fork" / "ai-ds-research",
)
SOURCE_PROJECTS = ("talk-bench", "talk-bench-talkdesk")
# Tests started from the Run test page run in the first checkout _source_repos() finds,
# inside the project whose uv workspace provides the agent module's top-level package:
# talkdesk_agent.* is importable only from talk-bench-talkdesk's venv; everything else
# (and any package no checkout provides) runs in talk-bench, whose talkbench.yaml applies.
TEST_PROJECT = "talk-bench"
_sync_lock = threading.Lock()


def _clone_dir(src: Path, dest: Path) -> None:
    """Copy a run dir; cp -c uses APFS clonefile (fast, no extra disk).
    -p preserves mtimes so _dir_signature comparisons stay stable."""
    try:
        subprocess.run(["cp", "-c", "-R", "-p", str(src), str(dest)], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        shutil.copytree(src, dest)


def _dir_signature(root: Path) -> list[tuple[str, int, int]]:
    """Size and mtime of every file, without dotfiles: the translations the viewer saves into an
    archived run (and Finder's .DS_Store) must not make it differ from the checkout's copy."""
    sig = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            st = p.stat()
            sig.append((str(p.relative_to(root)), st.st_size, int(st.st_mtime)))
    return sig


# the Sync button and test-job snapshots both stage in .sync-tmp-<run>, so they share one lock
_mirror_lock = threading.Lock()


def _mirror_run(src: Path) -> str | None:
    """Clone a run dir into the archive: "added" / "updated", or None when unchanged. It is
    staged under a temp name so an interrupted copy is never mistaken for a complete run."""
    target = DATA_DIR / src.name
    if src.resolve() == target.resolve():  # already in the archive: never swap a run talk-bench may be writing
        return None
    with _mirror_lock:
        if not src.is_dir():  # its checkout copy was just released (a test ended during a Sync)
            return None
        exists = target.exists()
        if exists and _dir_signature(src) == _dir_signature(target):
            return None
        tmp = DATA_DIR / f".sync-tmp-{src.name}"
        if tmp.exists():
            shutil.rmtree(tmp)
        _clone_dir(src, tmp)
        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
    return "updated" if exists else "added"


# Sync moves runs: once the archive holds a run in full, its checkout copy is deleted, provided
# nothing will write to it again. It needs results.json, --resume must have nothing left to redo
# or score in it (_resumable), no test of this page may be running in it and, for Sync, none of
# its files may have changed in the last _QUIET_S seconds (a talk-bench started in a terminal can
# be writing it). Runs kept back go at a later Sync, once finished.
_QUIET_S = 120
_RELEASING = ".viewer-releasing-"  # a checkout copy being deleted; Sync and Resume skip these


def _release_source(project: Path, run: Path, quiet_s: float = _QUIET_S) -> bool:
    """Delete a checkout's copy of a run the archive holds in full; True when it was deleted."""
    target = DATA_DIR / run.name
    if not (run / "results.json").is_file() or run.resolve() == target.resolve():
        return False
    if any(j["status"] in ("running", "cancelling") and Path(j.get("run_dir") or "").name == run.name
           for j in list(_jobs.values())):
        return False
    if _resumable(project, run):
        return False
    trash = run.with_name(_RELEASING + run.name)
    # under the lock _mirror_run holds, so a Sync never clones a half-deleted copy over the archive's
    with _mirror_lock:
        sig = _dir_signature(run)
        if quiet_s and max((m for _, _, m in sig), default=0) > time.time() - quiet_s:
            return False
        if not target.is_dir() or sig != _dir_signature(target):
            return False
        run.rename(trash)
    shutil.rmtree(trash, ignore_errors=True)
    return True


def _source_repos() -> list[Path]:
    """Existing source checkouts, in priority order, de-duplicated by real path
    (the same checkout can be reachable through a symlink under two names)."""
    env = os.environ.get("VIEWER_SOURCE_REPOS", "").strip()
    candidates = (
        [Path(part).expanduser() for part in env.split(os.pathsep) if part.strip()]
        if env else list(SOURCE_REPO_CANDIDATES)
    )
    repos, seen = [], set()
    for repo in candidates:
        if not repo.is_dir():
            continue
        real = repo.resolve()
        if real in seen:
            continue
        seen.add(real)
        repos.append(repo)
    return repos


@app.post("/api/sync_sources")
def sync_sources():
    if not _sync_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a sync is already running")
    try:
        repos = _source_repos()
        if not repos:
            looked = os.environ.get("VIEWER_SOURCE_REPOS", "").strip() or ", ".join(str(p) for p in SOURCE_REPO_CANDIDATES)
            raise HTTPException(status_code=404, detail=f"no source checkout found (looked in: {looked})")
        added, updated, unchanged, skipped = [], [], 0, 0
        removed, kept, errors = [], 0, []
        synced: set[str] = set()  # run names are globally unique: first checkout wins
        deleted = _load_deleted_runs()
        for repo in repos:
            for proj in SOURCE_PROJECTS:
                src_root = repo / proj / "data" / "runs"
                if not src_root.is_dir():
                    continue
                for run in sorted(src_root.iterdir()):
                    if run.name.startswith(_RELEASING):  # left over from a delete that failed halfway
                        shutil.rmtree(run, ignore_errors=True)
                        continue
                    if not run.is_dir() or run.name.startswith(".") or run.name in synced:
                        continue
                    synced.add(run.name)
                    if run.name in deleted:
                        skipped += 1
                        continue
                    outcome = _mirror_run(run)
                    if outcome:
                        (updated if outcome == "updated" else added).append(run.name)
                    else:
                        unchanged += 1
                    try:
                        if _release_source(repo / proj, run):
                            removed.append(run.name)
                        elif run.is_dir():
                            kept += 1
                    except OSError as e:
                        kept += 1
                        errors.append(f"{run.name}: {e}")
        _invalidate_index()
        return {
            "added": added,
            "updated": updated,
            "unchanged": unchanged,
            "skipped_deleted": skipped,
            "removed_from_checkout": removed,  # copied in full, then deleted from the checkout
            "kept_in_checkout": kept,  # not finished, resumable or still being written
            "errors": errors[:20],
            "repos": [str(r) for r in repos],
        }
    finally:
        _sync_lock.release()


# ---- scenario & run Q&A chat ----

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    path: str  # "<run>" for whole-run questions, "<run>/<scenario>" for one scenario
    messages: list[ChatMessage]


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n…[truncated, {len(text)} chars total]"


def _clip(text: Any, limit: int) -> str:
    """One-line truncation for table rows / excerpt fields."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _fmt_score(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def _as_dict(v: Any) -> dict[str, Any]:
    """Old runs serialized some nested results as repr strings; treat those as absent."""
    return v if isinstance(v, dict) else {}


def _breach_bit(breach: Any) -> str | None:
    """Compact guardrail verdict for the AI context. talk-bench attaches ``breach``
    ({breached, label, channels, detail}) to every guardrail scenario — one declaring
    prohibited_actions / prohibited_communicate / planted_secret — and leaves it None
    on every other scenario. ``label`` is tool | disclosure (breached), resisted
    (held) or errored (verdict untrusted; ``detail.status`` says why)."""
    if not breach:
        return None
    if not isinstance(breach, dict):
        return f"breach={_clip(breach, 120)}"
    if breach.get("breached"):
        channels = ",".join(map(str, breach.get("channels") or []))
        label = breach.get("label") or "breach"
        return f"guardrail=BREACHED:{label}" + (f"[{channels}]" if channels else "")
    if breach.get("label") == "errored":
        status = _as_dict(breach.get("detail")).get("status")
        return f"guardrail=errored({status})" if status else "guardrail=errored"
    return "guardrail=resisted"


def _run_dir_modality(run_dir: Path) -> str | None:
    return _run_modality(_load_json(run_dir / "results.json"), _scenario_dirs(run_dir))


_MODALITY_BRIEF = {
    "text": (
        "MODALITY — this is a TEXT run: the agent and the user simulator exchanged written "
        "messages only. There was no audio, ASR, TTS, barge-in or speech latency, so never "
        "attribute anything to speech recognition, transcription, audio quality or response "
        "delay. The voice sub-metrics were not measured: responsiveness is stored as 0 (with "
        "0 ms latencies over 0 turns) and voice_consistency as null. Treat both as not "
        "applicable — never as poor performance — and leave them out of summaries and reports. "
        "Because they are missing, `experience` and `final` are inflated relative to voice runs "
        "and must not be compared with them; lead with pass rate, goal score, resolution and "
        "the tool metrics instead."
    ),
    "voice": (
        "MODALITY — this is a VOICE run: the user simulator spoke through TTS and speech was "
        "transcribed by ASR, so transcripts can carry recognition errors, and responsiveness / "
        "voice_consistency are real measurements of the agent."
    ),
}


def _user_simulator_line(run_dir: Path) -> list[str]:
    """Context line for runs whose caller was tau2-bench's simulator, so the analyst
    does not attribute its turns or hang-ups to talk-bench's own simulator."""
    marker = _run_user_simulator(run_dir)
    if not marker:
        return []
    return [
        f"User simulator: tau2-bench native ({marker.get('simulator')}, LLM {marker.get('llm')}) — "
        "the caller's words, interruptions and hang-ups came from tau2's voice user simulator "
        "(tau2 voice guidelines + task instructions, ###STOP### to end), not talk-bench's EventDrivenSimulator"
    ]


def _scenario_context(sdir: Path, modality: str | None = None) -> str:
    """Assemble the artifacts an analyst would read to diagnose this scenario."""
    run_dir = sdir.parent
    results, summary, session = _scenario_results(sdir)
    if session and modality == "text":
        # zero-filled placeholders an analyst would otherwise read as a real failure
        session = {k: v for k, v in session.items() if k not in ("responsiveness", "latency")}
    scenario = _load_json(sdir / "scenario.json")
    conversation = _load_json(sdir / "conversation.json")
    tool_log = _load_json(sdir / "tool_log.json")
    meta = _run_meta(sdir)

    parts = [
        f"Run: {run_dir.name}",
        f"Scenario: {sdir.name}",
        f"Agent: {results.get('agent_name', 'unknown')}",
        f"Modality: {modality or 'unknown'}",
        *_user_simulator_line(run_dir),
        f"End reason: {meta.get('end_reason')} · duration: {round((meta.get('total_duration_ms') or 0) / 1000)}s",
    ]
    if scenario:
        parts.append("## Scenario definition (goal, persona, tasks)\n" + _truncate(json.dumps(scenario, indent=1), 8000))
    if summary:
        parts.append("## Scenario result (results.json per_scenario entry)\n" + _truncate(json.dumps(summary, indent=1), 4000))
    if session:
        note = (
            " — responsiveness and latency are omitted: not measured for a text run"
            if modality == "text" else ""
        )
        parts.append(
            f"## Evaluation session result (task_completion judge output){note}\n"
            + _truncate(json.dumps(session, indent=1), 6000)
        )
    if conversation:
        lines = [f"[turn {t.get('turn_index')}] {t.get('speaker')}: {t.get('text')}" for t in conversation]
        parts.append("## Transcript\n" + _truncate("\n".join(lines), 20000))
    if tool_log:
        # clip each call's result individually so a single huge result (e.g. a flight
        # search returning hundreds of combos) can't push later calls out of the context
        calls = []
        for c in tool_log:
            if not isinstance(c, dict):
                continue
            result = c.get("result")
            result_text = json.dumps(result)
            if len(result_text) > 2000:
                result = _clip(result_text, 2000) + f" [result clipped, {len(result_text)} chars total]"
            calls.append({
                "tool": c.get("tool_name"),
                "arguments": c.get("arguments"),
                "db_changed": c.get("db_changed"),
                "result": result,
            })
        parts.append(
            f"## Tool calls ({len(calls)} total, long results clipped per call)\n"
            + _truncate(json.dumps(calls, indent=1), 30000)
        )
    return "\n\n".join(parts)


def _scenario_line(name: str, summary: dict[str, Any], session: dict[str, Any]) -> str:
    """One compact line per scenario for the run-level context."""
    passed = summary.get("passed")
    bits = [f"{name}:", {True: "PASS", False: "FAIL"}.get(passed, "NO-RESULT")]
    if summary:
        bits.append(f"goal={_fmt_score(summary.get('goal_score'))}")
        breakdown = _as_dict(summary.get("tau2_reward_breakdown"))
        if breakdown:
            bits.append("tau2[" + " ".join(f"{k}={_fmt_score(v)}" for k, v in breakdown.items()) + "]")
        bits.append(f"end={_fmt_score(summary.get('end_reason'))}")
        bits.append(f"turns={_fmt_score(summary.get('turn_count'))}")
        if summary.get("total_duration_ms"):
            bits.append(f"dur={round(summary['total_duration_ms'] / 1000)}s")
        tools = _as_dict(summary.get("tool_metrics"))
        if tools:
            bits.append(
                f"tools[recall={_fmt_score(tools.get('recall'))}"
                f" precision={_fmt_score(tools.get('precision'))}"
                f" calls={tools.get('total_calls')}]"
            )
        if tools.get("prohibited_violations"):
            bits.append("prohibited_tools=" + ",".join(map(str, tools["prohibited_violations"])))
        breach = _breach_bit(summary.get("breach"))
        if breach:
            bits.append(breach)
        if summary.get("tau2_reward_note"):
            bits.append(f'note="{_clip(summary["tau2_reward_note"], 160)}"')
    tc = _as_dict(session.get("task_completion"))
    flags = [
        k for k in ("judge_failed", "prohibited_violated", "prohibited_communicate_violated")
        if tc.get(k)
    ]
    if tc.get("state_integrity_violations"):
        flags.append("state_integrity_violations")
    if flags:
        bits.append("flags=" + ",".join(flags))
    raw_notes = session.get("scoring_notes")
    notes = [
        n for n in (raw_notes if isinstance(raw_notes, list) else [])
        # drop boilerplate about sub-metrics that simply lacked data
        if not str(n).endswith(("=too_few_turns", "=too_short", "=unavailable"))
    ]
    if notes:
        bits.append('scoring_notes="' + _clip("; ".join(map(str, notes)), 160) + '"')
    return " ".join(bits)


def _failed_scenario_excerpt(sdir: Path, summary: dict[str, Any]) -> str:
    """Goal + tool errors + transcript tail for one not-passed scenario."""
    scenario = _load_json(sdir / "scenario.json") or {}
    conversation = _load_json(sdir / "conversation.json") or []
    tool_log = _load_json(sdir / "tool_log.json") or []
    result = "FAIL" if summary.get("passed") is False else "NO-RESULT"
    lines = [
        f"### {sdir.name} ({result}, goal={_fmt_score(summary.get('goal_score'))}, "
        f"end={summary.get('end_reason')})"
    ]
    goal = (_as_dict(scenario.get("goal")).get("description") or "").strip()
    if goal:
        lines.append("Goal: " + _clip(goal, 320))
    errors = []
    for call in tool_log:
        if not isinstance(call, dict):
            continue
        res = call.get("result")
        err = res.get("error") if isinstance(res, dict) else None
        if err:
            msg = (err.get("message") or json.dumps(err)) if isinstance(err, dict) else err
            errors.append(f"{call.get('tool_name')}: {_clip(msg, 150)}")
    if errors:
        shown = [
            msg + (f" (×{n})" if n > 1 else "")
            for msg, n in Counter(errors).most_common(4)
        ]
        lines.append(f"Tool errors ({len(errors)}/{len(tool_log)} calls): " + " | ".join(shown))
    if conversation:
        tail: list[str] = []
        used = 0
        for turn in reversed([t for t in conversation if isinstance(t, dict)]):
            text = f"[{turn.get('turn_index')}] {turn.get('speaker')}: {_clip(turn.get('text') or '', 240)}"
            if tail and used + len(text) > 1200:
                break
            tail.append(text)
            used += len(text)
        tail.reverse()
        skipped = len(conversation) - len(tail)
        label = "Transcript" + (f" tail (first {skipped} turns omitted)" if skipped > 0 else "")
        lines.append(label + ":\n" + "\n".join(tail))
    else:
        lines.append("No transcript recorded (conversation.json empty or missing).")
    return "\n".join(lines)


# cap on the total size of failed-scenario excerpts in the run context
_RUN_EXCERPT_BUDGET = 60_000


def _run_context(run_dir: Path, modality: str | None = None) -> str:
    """Whole-run digest: headline scores, one line per scenario, precomputed
    failure aggregates, and budget-capped excerpts for scenarios that did not pass."""
    results = _as_dict(_load_json(run_dir / "results.json"))
    eval_res = _as_dict(_load_json(run_dir / "talk_bench_evaluation_result.json"))
    per_scenario = {
        s.get("scenario_id"): s for s in results.get("per_scenario") or [] if isinstance(s, dict)
    }
    sessions = {
        s.get("session_id"): s for s in eval_res.get("session_results") or [] if isinstance(s, dict)
    }

    parts = [
        f"Run: {run_dir.name}",
        f"Agent: {results.get('agent_name') or 'unknown'}",
        f"Modality: {modality or 'unknown'}",
        *_user_simulator_line(run_dir),
    ]
    if results:
        parts.append(
            f"Primary score: {_fmt_score(results.get('primary_score'))} · "
            f"passed {results.get('scenarios_passed')}/{results.get('scenarios_total')} · "
            f"errored {results.get('scenarios_errored')}"
        )
        if results.get("per_domain"):
            parts.append("Per-domain results: " + _truncate(json.dumps(results["per_domain"]), 2000))
    else:
        parts.append(
            "results.json is missing (the run likely errored or was interrupted); "
            "per-scenario summaries below may be empty."
        )
    if eval_res:
        keys = (*EVAL_METRICS, "pass_at1", "mean_pass_rate", "judge_parse_failure_rate")
        if modality == "text":
            keys = tuple(k for k in keys if k not in ("responsiveness", "voice_consistency"))
        agg = {k: eval_res.get(k) for k in keys if eval_res.get(k) is not None}
        if agg:
            block = "## talk-bench evaluation aggregate scores\n" + json.dumps(agg, indent=1)
            if modality == "text":
                block += (
                    "\n(responsiveness and voice_consistency are omitted: not measured for a text "
                    "run. They are therefore excluded from `experience`, which is why `experience` "
                    "and `final` here sit above what a voice run of the same tasks would score.)"
                )
            parts.append(block)

    # scenarios come from the artifact directories plus any results.json entries whose
    # directory is missing (hard-errored scenarios often leave no artifacts behind)
    entries: list[tuple[str, Path | None]] = [(d.name, d) for d in _scenario_dirs(run_dir)]
    dir_names = {name for name, _ in entries}
    entries += [(sid, None) for sid in per_scenario if sid and sid not in dir_names]

    lines = []
    end_reasons: Counter[str] = Counter()   # among not-passed scenarios
    comp_fail: Counter[str] = Counter()     # tau2 component < 1
    comp_seen: Counter[str] = Counter()
    domain_stats: dict[str, list[int]] = {}  # domain -> [passed, total]
    not_passed_dirs: list[tuple[Path, dict[str, Any]]] = []
    for name, sdir in entries:
        session = _as_dict(sessions.get(name))
        # older results.json lack the tau2 breakdown/notes; the evaluation file's
        # embedded scenario_result carries them, so merge (results.json wins)
        summary = {**_as_dict(session.get("scenario_result")), **_as_dict(per_scenario.get(name))}
        line = _scenario_line(name, summary, session)
        lines.append(line + (" artifacts=missing" if sdir is None else ""))
        if summary.get("domain"):
            st = domain_stats.setdefault(str(summary["domain"]), [0, 0])
            st[1] += 1
            st[0] += summary.get("passed") is True
        for comp, v in _as_dict(summary.get("tau2_reward_breakdown")).items():
            comp_seen[comp] += 1
            if not isinstance(v, (int, float)) or v < 1:
                comp_fail[comp] += 1
        if summary.get("passed") is not True:
            end_reasons[str(summary.get("end_reason"))] += 1
            if sdir is not None:
                not_passed_dirs.append((sdir, summary))
    if lines:
        parts.append("## Per-scenario results (one line each)\n" + "\n".join(lines))

    agg_lines = [f"{end_reasons.total()}/{len(entries)} scenarios did not pass."]
    if end_reasons:
        agg_lines.append(
            "End reasons among not-passed scenarios: "
            + ", ".join(f"{k}={n}" for k, n in end_reasons.most_common())
        )
    if comp_seen:
        agg_lines.append(
            "tau2 component failures (failed/scored): "
            + ", ".join(f"{k}: {comp_fail[k]}/{comp_seen[k]}" for k in sorted(comp_seen))
        )
    if len(domain_stats) > 1:
        agg_lines.append(
            "Per-domain passed/total: "
            + ", ".join(f"{d}: {p}/{t}" for d, (p, t) in sorted(domain_stats.items()))
        )
    parts.append("## Precomputed aggregates\n" + "\n".join(agg_lines))

    if not_passed_dirs:
        blocks: list[str] = []
        omitted: list[str] = []
        spent = 0
        for sdir, summary in not_passed_dirs:
            if spent >= _RUN_EXCERPT_BUDGET:
                omitted.append(sdir.name)
                continue
            block = _failed_scenario_excerpt(sdir, summary)
            blocks.append(block)
            spent += len(block)
        header = "## Details for scenarios that did not pass (goal, tool errors, transcript tail)"
        if omitted:
            header += (
                f"\nContext budget reached: excerpts cover the first {len(blocks)} not-passed "
                f"scenarios; omitted for {len(omitted)}: {', '.join(omitted)}"
            )
        parts.append(header + "\n\n" + "\n\n".join(blocks))
    return "\n\n".join(parts)


_CHAT_STYLE_RULES = (
    "Be direct and concise. Answer in the same language the user writes in. When answering "
    "in Chinese, keep benchmark and technical terms such as Agent, User Simulator, tool call, "
    "ASR, LLM, and API in English; in particular, never translate Agent as '代理'. Preserve "
    "product names, model names, function names, and field names exactly as written in the artifacts."
)


def _analyst_prompt(modality: str | None, task: str, context: str) -> str:
    role = {"text": "text-agent", "voice": "voice-agent"}.get(modality or "", "conversational-agent")
    brief = _MODALITY_BRIEF.get(modality or "")
    return (
        f"You are a {role} benchmark analyst. {task}{_CHAT_STYLE_RULES}\n\n"
        + (brief + "\n\n" if brief else "") + context
    )


def _complete(system: str, messages: list[dict[str, str]], timeout: int) -> str:
    import litellm  # imported lazily: heavy module, only needed for the AI features

    try:
        resp = litellm.completion(
            model=LLM_MODEL, messages=[{"role": "system", "content": system}, *messages], timeout=timeout
        )
        choice = resp.choices[0]
        content = choice.message.content
    except Exception as e:  # surface provider errors to the UI
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")
    if not content:  # e.g. finish_reason=content_filter leaves content None
        raise HTTPException(status_code=502, detail=f"LLM returned no text (finish_reason={choice.finish_reason})")
    return content


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    is_run = len(Path(req.path).parts) == 1
    if is_run:
        run_dir = _data_dir_at_depth(req.path, 1, "run")
        modality = _run_dir_modality(run_dir)
        system = _analyst_prompt(modality, (
            "The user is reviewing one "
            "complete talk-bench run in which an agent was tested on many scenarios. Using the "
            "run data below, answer questions about the overall outcome. When asked for a "
            "summary, cover: (1) headline numbers (pass rate, key scores); (2) the dominant "
            "failure patterns, grouped by root cause (agent behavior, tool errors, "
            + ("" if modality == "text" else "ASR/transcription issues, ")
            + "user-simulator behavior, judge/scoring artifacts, timeouts or infra "
            "errors), each with a count and example scenario ids; (3) anything else notable. "
            "Ground every claim in the data and cite scenario ids as evidence. Transcript "
            "excerpts are included only for scenarios that did not pass, and may be capped; "
            "if something is not in the context, say so instead of guessing — the user can "
            "open a scenario page and its per-scenario AI chat for a deep dive. "
        ), _run_context(run_dir, modality))
    else:
        sdir = _data_dir_at_depth(req.path, 2, "scenario")
        modality = _run_dir_modality(sdir.parent)
        system = _analyst_prompt(modality, (
            "The user is investigating "
            "one talk-bench scenario run. Using the artifacts below, answer questions about what "
            "happened and, when the scenario failed, diagnose the concrete root cause (agent "
            "behavior, tool errors, "
            + ("" if modality == "text" else "ASR issues, ")
            + "judge/scoring details, user-simulator behavior, timeouts). "
            "Quote specific turns or tool calls as evidence. "
        ), _scenario_context(sdir, modality))
    messages = [{"role": m.role, "content": m.content} for m in req.messages[-20:]]
    return {"reply": _complete(system, messages, timeout=180 if is_run else 90)}  # run summaries read far more context


# ---- one-click AI reports (scenario: task summary + actions + verdict + flowchart;
# ----                       run: totals + failure clustering + evidence + advice) ----

class ReportRequest(BaseModel):
    path: str  # "<run>/<scenario>" for scenario reports, "<run>" for run reports


_SCENARIO_REPORT_PROMPT = """\
请为这个 scenario 生成一份结构化分析报告，用 Markdown 输出，依次包含下面四个部分（用 ### 标题）：

### 任务要求
用几句话说明这个 scenario 的设定与要求：User Simulator 扮演什么角色、目标是什么、有哪些关键约束，期望 Agent 完成哪些事情（如果 scenario 定义了期望动作或禁止动作，也一并说明）。

### Agent 实际做了什么
先按时间顺序概述对话是如何展开的；然后逐一列出 Agent 的 tool call，用表格呈现（列：# | 工具 | 关键参数 | 结果），说明每次调用是否成功、返回了什么关键信息、是否修改了数据库（db_changed），失败或重复的调用要指出来。如果整个 scenario 没有任何 tool call，请明确说明并解释这意味着什么。

### 做得好不好
先用一句话下结论，再结合判分数据（pass/fail、goal score、tau2 breakdown、judge 评语、tool recall/precision 等）分别说明：做得好的地方；有问题的地方。如果 scenario 失败了，给出具体根因，并引用具体的 turn 或 tool call 作为证据。

### 流程图
用一个 mermaid flowchart 总结这个 scenario 里实际发生的流程（用户意图 → Agent 的关键动作与工具调用 → 最终结果）。语法必须严格遵守以下规则，保证能被 mermaid 渲染：
- 放在 ```mermaid 代码块中，第一行为 flowchart TD
- 节点 id 只用字母和数字（A、B、C1 等）；所有节点文本必须用双引号包裹，例如 A["用户请求退款"]
- 节点文本中不要出现双引号、反引号、圆括号、方括号、大括号或分号
- 普通步骤用 A["文本"]，工具调用用 B[["get_order"]]，判断或分支用 C{"是否找到订单"}；失败、错误的环节在文本中写明
- 连线用 A --> B，需要说明时用 A -->|"说明"| B
- 不要使用 subgraph、style、classDef、click 等高级语法
- 不超过 15 个节点，突出主干流程
"""


@app.post("/api/scenario_report")
def api_scenario_report(req: ReportRequest):
    sdir = _data_dir_at_depth(req.path, 2, "scenario")
    modality = _run_dir_modality(sdir.parent)
    system = _analyst_prompt(modality, (
        "The user pressed a button "
        "asking for a structured report on one talk-bench scenario run. Ground every claim in "
        "the artifacts below and cite specific turns or tool calls as evidence; if something "
        "is not in the context, say so instead of guessing. "
    ), _scenario_context(sdir, modality))
    return {"report": _complete(system, [{"role": "user", "content": _SCENARIO_REPORT_PROMPT}], timeout=120)}


_RUN_REPORT_PROMPT = """\
请为这次 run 生成一份详细的分析报告，用 Markdown 输出，依次包含下面五个部分（用 ### 标题）：

### 总体结果
用一段话概括本次 run 的整体表现，然后用表格列出关键数字：场景总数、通过 / 未通过 / errored、\
primary score、talk-bench 各项聚合分（<<AGG_METRICS>>）。

### 失败原因分布
把所有未通过的场景按根因分组（例如：超时或 infra 错误、Agent 未正确完成业务操作导致 DB 校验失败、\
<<FAILURE_KINDS>>——以数据里实际出现的为准，不要生搬这个列表）。\
用表格呈现：原因 | 场景数 | 场景 ID 列表。每个未通过的场景都要归入且只归入一组：\
同一场景有多个因素时归入最直接导致失败的那组，可在备注里提次要因素；\
证据不足无法归因的场景放入"证据不足"组并说明缺少什么（例如 artifacts missing、transcript 摘录被省略）。\
各组场景数之和必须等于未通过场景总数。

表格之后，用一个 mermaid pie 图展示各组场景数。语法必须严格遵守：放在 ```mermaid 代码块中，\
第一行为 pie，第二行可以是 title 加简短标题，之后每行的格式严格为 "标签" : 数字\
（标签用双引号包裹，标签内不要出现引号、冒号或反引号）。

### 高频问题详解
挑出场景数最多的 3~5 组问题逐组展开：这类问题的典型表现、引用具体场景作为证据\
（场景 ID + 对话片段或 tool 错误原文）、它如何直接导致判分失败。

### 值得注意的其他发现
列出不属于上述主线但值得注意的现象（例如通过但分数或行为异常的场景、指标之间的矛盾、\
tool 使用精度异常低等）；没有就写"无"。

### 改进建议
按优先级列出可执行的改进建议，每条注明预计影响多少个场景（依据上面的分组计数）。

注意：上下文对未通过场景只提供部分 transcript 摘录（可能有省略），对被省略的场景不要编造细节；\
引用场景 ID 时保持原样。
"""


def _run_report_prompt(modality: str | None) -> str:
    if modality == "text":
        agg = (
            "final、resolution、experience、customer effort、conversation quality 等，只列上下文里有的；"
            "responsiveness 与 voice consistency 在 text run 中没有测量，不要列出，也不要当成扣分项"
        )
        kinds = (
            "tool 调用报错、错误转人工、Agent 提前结束或挂断对话、judge 或评分工件等"
        )
    else:
        agg = "final、resolution、experience、responsiveness 等，只列上下文里有的"
        kinds = "tool 调用报错、错误转人工、ASR/语音转写问题、judge 或评分工件等"
    return _RUN_REPORT_PROMPT.replace("<<AGG_METRICS>>", agg).replace("<<FAILURE_KINDS>>", kinds)


@app.post("/api/run_report")
def api_run_report(req: ReportRequest):
    run_dir = _data_dir_at_depth(req.path, 1, "run")
    modality = _run_dir_modality(run_dir)
    system = _analyst_prompt(modality, (
        "The user pressed a button "
        "asking for a detailed report on one complete talk-bench run in which an agent was "
        "tested on many scenarios. Ground every claim in the run data below and cite scenario "
        "ids as evidence. Transcript excerpts are included only for scenarios that did not "
        "pass, and may be capped; if something is not in the context, say so instead of "
        "guessing. "
    ), _run_context(run_dir, modality))
    prompt = [{"role": "user", "content": _run_report_prompt(modality)}]
    return {"report": _complete(system, prompt, timeout=300)}  # large runs: big context and a long structured answer


# ---- transcript translation: Simplified Chinese under each turn on the scenario page ----
# Saved beside conversation.json as a dotfile, so it stays out of the file list. It is
# keyed by a hash of the turn texts: a transcript that changed reads as not translated yet.

TRANSLATION_FILE = ".transcript_zh.json"
_TRANSLATE_BATCH_CHARS = 6000  # per LLM call; a long transcript's batches run in parallel

_TRANSLATE_SYSTEM = (
    "You translate the transcript of a customer-service conversation between an Agent and a "
    "user (played by a benchmark's User Simulator) into Simplified Chinese for an engineer "
    "reviewing it. The input is a JSON array of turns {\"i\": id, \"text\": original}. Reply with "
    "only a JSON array holding one {\"i\": id, \"zh\": translation} per input turn, with the same "
    "ids, and nothing else. Translate each turn faithfully and on its own, keeping its tone: "
    "never merge, summarize, answer or correct turns. Speech-recognition errors and odd wording "
    "are evidence, so translate what is written and keep a garbled word in the original, in "
    "brackets. Keep names of people, places, airlines and products, IDs, codes, booking and "
    "order numbers, email addresses and phone numbers exactly as written. Keep the terms Agent, "
    "User Simulator, tool call, ASR, LLM and API in English; never translate Agent as '代理'. "
    "Return a turn that is already in Chinese unchanged."
)


def _translate_batch(batch: list[tuple[int, str]], system: str = _TRANSLATE_SYSTEM) -> dict[int, str]:
    payload = json.dumps([{"i": i, "text": text} for i, text in batch], ensure_ascii=False)
    reply = _complete(system, [{"role": "user", "content": payload}], timeout=240)
    array = re.search(r"\[.*\]", reply, re.S)  # tolerate a ```json fence or a stray sentence
    try:
        items = json.loads(array.group(0) if array else reply)
    except json.JSONDecodeError:
        items = None
    if not isinstance(items, list):
        raise HTTPException(status_code=502, detail="LLM returned a translation that is not a JSON array")
    wanted = {i for i, _ in batch}
    return {
        it["i"]: it["zh"] for it in items
        if isinstance(it, dict) and it.get("i") in wanted and isinstance(it.get("zh"), str)
    }


def _translate_all(texts: list[str], system: str) -> dict[int, str]:
    """{index: translation} for the non-blank texts, in batches of ~_TRANSLATE_BATCH_CHARS run in
    parallel, plus a second pass for any the model skipped; a text still missing is left out."""
    batches: list[list[tuple[int, str]]] = [[]]
    size = 0
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        if batches[-1] and size + len(text) > _TRANSLATE_BATCH_CHARS:
            batches.append([])
            size = 0
        batches[-1].append((i, text))
        size += len(text)
    done: dict[int, str] = {}
    for _ in range(2):
        todo = [[(i, t) for i, t in b if i not in done] for b in batches]
        todo = [b for b in todo if b]
        if not todo:
            break
        with ThreadPoolExecutor(max_workers=4) as pool:
            for part in pool.map(lambda b: _translate_batch(b, system), todo):
                done.update(part)
    return done


def _texts_key(texts: list[str]) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()[:16]


def _cached_translation(sdir: Path, texts: list[str] | None) -> list[str | None] | None:
    cached = _load_json(sdir / TRANSLATION_FILE)
    if texts and isinstance(cached, dict) and cached.get("key") == _texts_key(texts):
        return cached.get("translations")
    return None


@app.post("/api/translate_transcript")
def api_translate_transcript(req: ReportRequest):
    sdir = _data_dir_at_depth(req.path, 2, "scenario")
    texts = _transcript_texts(_load_json(sdir / "conversation.json"),
                              _records_transcript(_load_json(sdir / "records.json"), None))
    if not texts:
        raise HTTPException(status_code=404, detail="no transcript to translate")
    if (cached := _cached_translation(sdir, texts)) is not None:
        return {"translations": cached}
    done = _translate_all(texts, _TRANSLATE_SYSTEM)
    translations = [done.get(i) for i in range(len(texts))]
    if all(t is not None or not text.strip() for t, text in zip(translations, texts)):
        try:  # an incomplete translation isn't kept, so the next click fills the gaps
            _write_json(sdir / TRANSLATION_FILE, {"key": _texts_key(texts), "model": LLM_MODEL,
                                                  "translations": translations})
        except OSError:
            pass  # read-only data dir: still shown, just not kept
    return {"translations": translations}


# ---- task panel translation: Simplified Chinese under the task and grading texts ----
# The scenario page's side panel shows scenario.json's task (goal, caller instructions, what the
# caller knows) and grading criteria with the judge's reasoning. Its prose is translated on request
# and kept as {original: translation} in a dotfile beside scenario.json, so a re-evaluated run (new
# judge reasoning) only translates what is new. IDs, tool calls and DB fields are not prose: left out.

TASK_TRANSLATION_FILE = ".task_zh.json"

_TASK_TRANSLATE_SYSTEM = (
    "You translate parts of a customer-service benchmark scenario into Simplified Chinese for an "
    "engineer reviewing it: the task given to the User Simulator that plays the caller, the facts "
    "it knows, the criteria an LLM judge grades the Agent on, and the judge's reasoning. The input "
    "is a JSON array {\"i\": id, \"text\": original}. Reply with only a JSON array holding one "
    "{\"i\": id, \"zh\": translation} per input item, with the same ids, and nothing else. "
    "Translate each item faithfully and on its own, keeping every condition, number and ordering "
    "word (before, after, only if, must not): never summarize, merge or judge them. Keep names of "
    "people, places, airlines and products, IDs, codes, dates, amounts, booking and order numbers, "
    "email addresses, phone numbers, tool and function names (snake_case) and quoted speech "
    "exactly as written. Keep the terms Agent, User Simulator, tool call, ASR, LLM and API in "
    "English; never translate Agent as '代理'. Return an item that is already in Chinese unchanged."
)


def _task_texts(scenario: Any, session: dict | None) -> list[str]:
    """The prose the task panel shows, deduplicated in panel order (index.html taskPanelHTML)."""
    sc = scenario if isinstance(scenario, dict) else {}
    us = sc.get("user_scenario") or {}
    ev = sc.get("evaluation") or {}
    tc = (session or {}).get("task_completion") or {}
    crit = lambda c: c if isinstance(c, str) else c.get("text") if isinstance(c, dict) else None  # noqa: E731
    texts = [(sc.get("goal") or {}).get("description"), us.get("reason_for_call"), us.get("task_instructions")]
    known = us.get("known_info")
    texts += list(known.values()) if isinstance(known, dict) else []
    texts += list(us.get("unknown_info") or [])
    for key, checks in (("communicate_info", "communicate_checks"), ("nl_assertions", "nl_assertion_checks")):
        texts += [crit(c) for c in ev.get(key) or []]
        texts += [c.get("justification") for c in tc.get(checks) or [] if isinstance(c, dict)]
    texts += [crit(c) for c in ev.get("prohibited_communicate") or []]
    return list(dict.fromkeys(t for t in texts if isinstance(t, str) and t.strip()))


def _cached_task_translation(sdir: Path) -> dict[str, str]:
    cached = _load_json(sdir / TASK_TRANSLATION_FILE)
    found = cached.get("map") if isinstance(cached, dict) else None
    return found if isinstance(found, dict) else {}


@app.post("/api/translate_task")
def api_translate_task(req: ReportRequest):
    sdir = _data_dir_at_depth(req.path, 2, "scenario")
    _, _, session = _scenario_results(sdir)
    texts = _task_texts(_load_json(sdir / "scenario.json"), session)
    if not texts:
        raise HTTPException(status_code=404, detail="no task text to translate")
    known = _cached_task_translation(sdir)
    todo = [t for t in texts if t not in known]
    if todo:
        done = _translate_all(todo, _TASK_TRANSLATE_SYSTEM)
        known.update({todo[i]: zh for i, zh in done.items()})
        try:  # kept even when incomplete: it is keyed by text, so the next click asks only for the gaps
            _write_json(sdir / TASK_TRANSLATION_FILE, {"model": LLM_MODEL, "map": known})
        except OSError:
            pass  # read-only data dir: still shown, just not kept
    return {"map": {t: known[t] for t in texts if t in known}, "missing": sum(t not in known for t in texts)}


# ---- run test: launch talk-bench runs with a user-configured agent ----
#
# A test is exactly
#     uv run talk-bench run --config <agent.yaml> --domain X --split Y --source S
#         [--scenario-ids a,b] [--speech-complexity <preset>]
# started as a child process inside one of the local ai-ds-research checkouts;
# every other CLI option keeps its default. The working directory matters: the
# CLI finds the checkout's talkbench.yaml from there and writes the run to its
# data/runs. Two .env files load: litellm's import-time load_dotenv() walks up from
# the .venv in use (so talk-bench-talkdesk/.env supplies the TALKDESK_* credentials
# there), then the CLI's own load_dotenv() adds talk-bench/.env, found next to its
# module from either cwd, without overriding. The agent YAML the user edited or imported is written
# verbatim under <data dir>/.test_jobs/<job>/ next to the log, and the run
# directory is cloned into the archive while it grows and when it ends, so the
# run shows up in the list like any other. `talk-bench run --resume <run dir>` (redo
# the errored / unfinished scenarios of a run, merged into its results.json) is
# offered for the tests started here and for any run a checkout holds.

JOBS_DIR = DATA_DIR / ".test_jobs"
MAX_RUNNING_TESTS = 4
_LOG_CHUNK = 512_000  # bytes of log returned per poll
_SNAPSHOT_INTERVAL_S = 90  # re-clone a running job's partial run dir into the archive this often
_LIVE_REFRESH_S = 10  # re-read a run talk-bench writes straight into the archive this often

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_start_lock = threading.Lock()  # held from the "may this test start?" checks until the job is in _jobs

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_AGENT_MODULE_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")
_MCP_TRANSPORTS = {"platform_interaction_id", "dynamic_variable_header", "direct_header"}
# top-level YAML blocks the runner reads next to `agent:` (talk-bench config_loader)
_EXTRA_BLOCKS = ("simulator", "evaluation", "simulator_gate", "mid_turn_speak_gate", "terminal_hangup_gate")
# console lines of `talk-bench run` worth turning into structured status
_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\] (\S+)(?: \(trial \d+/\d+\))?: ([^\[]+?) \[[\d.]+s\]")
_LOADED_RE = re.compile(r"Loaded (\d+) scenarios")
_SCORE_RE = re.compile(r"^Score:\s*([\d.]+)%\s*\((\d+)/(\d+)\s*passed\)")
_RESULTS_RE = re.compile(r"^Results:\s*(.+?)[/\\]results\.json\s*$")
_ERRORS_RE = re.compile(r"^Errors:\s*(\d+)\s*$")  # printed after Score: when some scenarios errored
# talk_bench = talk-bench's own simulator; tau2 = tau2-bench's native user simulator,
# swapped in by tau2_user_sim/launch.py (which runs the same CLI in the checkout's venv)
USER_SIMULATORS = ("talk_bench", "tau2")
TAU2_SIM_LAUNCHER = BASE_DIR / "tau2_user_sim" / "launch.py"
# tau2's SpeechComplexity presets (v1.0.1) for --speech-complexity on tau2 tasks: the
# caller's persona (verbosity, whether it interrupts), voice and audio effects
SPEECH_COMPLEXITIES = (
    "control", "regular", "control_audio", "control_accents", "control_behavior",
    "control_audio_accents", "control_audio_behavior", "control_accents_behavior",
)


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uv_bin() -> str | None:
    found = shutil.which("uv")
    candidates = ([found] if found else []) + [
        str(Path.home() / ".local" / "bin" / "uv"), "/opt/homebrew/bin/uv", "/usr/local/bin/uv",
    ]
    return next((c for c in candidates if Path(c).is_file()), None)


def _launcher(root: Path, user_simulator: str = "talk_bench") -> list[str] | None:
    """How to invoke the CLI in a checkout: `uv run talk-bench` (what the docs use),
    or the venv's shim when uv is not installed. With the tau2 user simulator it is the
    same CLI started through tau2_user_sim/launch.py in that checkout's venv."""
    uv = _uv_bin()
    if user_simulator == "tau2":
        if uv:
            return [uv, "run", "python", str(TAU2_SIM_LAUNCHER)]
        python = root / ".venv" / "bin" / "python"
        return [str(python), str(TAU2_SIM_LAUNCHER)] if python.is_file() else None
    if uv:
        return [uv, "run", "talk-bench"]
    shim = root / ".venv" / "bin" / "talk-bench"
    return [str(shim)] if shim.is_file() else None


def _talk_bench_lib_root(root: Path) -> Path | None:
    """The checkout holding the domain data: talk-bench-talkdesk depends on its sibling."""
    for cand in (root, root.parent / "talk-bench"):
        if (cand / "libs" / "talk_bench_domains").is_dir():
            return cand
    return None


# per --source: the checkout's domain data dir (under _talk_bench_lib_root) and its splits file
_DOMAIN_DATA = {
    "talk_bench": ("libs/talk_bench_domains/src/talk_bench_domains", "splits.json"),
    "tau2": ("vendor/tau2-bench/data/tau2/domains", "split_tasks.json"),
}


def _domain_dirs(lib: Path, source: str) -> list[Path]:
    base = lib / _DOMAIN_DATA[source][0]
    if not base.is_dir():
        return []
    return [d for d in sorted(base.iterdir()) if d.is_dir() and (source == "tau2" or not d.name.startswith(("_", ".")))]


def _scan_domains(root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """domain -> splits (with task counts) per dataset source, read from the checkout's
    own data files so the pickers offer exactly what the CLI accepts."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {"talk_bench": {}, "tau2": {}}
    lib = _talk_bench_lib_root(root)
    for source, domains in out.items():
        for d in _domain_dirs(lib, source) if lib else []:
            splits = _load_json(d / _DOMAIN_DATA[source][1])
            if isinstance(splits, dict) and splits:
                domains[d.name] = [
                    {"split": str(k), "count": len(v) if isinstance(v, (list, dict)) else None} for k, v in splits.items()
                ]
            elif source == "talk_bench" and (d / "tasks.json").is_file():
                tasks = _load_json(d / "tasks.json")
                domains[d.name] = [{"split": "base", "count": len(tasks) if isinstance(tasks, list) else None}]
    return out


# scenario ids as talk-bench's data files spell them: tau2 telecom ids look like
# "telecom-tau2-[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]"
_ID_RE = re.compile(r"^[\w.\-\[\]:|]+$")
MAX_SCENARIO_ID_LEN = 300
MAX_SCENARIO_IDS = 2000


def _scenario_catalog(root: Path, source: str, domain: str, split: str) -> list[dict[str, Any]]:
    """The scenario ids `--domain/--split/--source` select, read from the same data files
    the CLI uses, each with a one-line summary for the picker. Comma-joined splits are
    unioned and `all` covers every domain, mirroring talk-bench."""
    lib = _talk_bench_lib_root(root)
    if lib is None:
        return []
    tau2 = source == "tau2"
    base = lib / _DOMAIN_DATA[source][0]
    domains = [d.name for d in _domain_dirs(lib, source)] if domain == "all" else [domain]
    parts = [x.strip() for x in split.split(",") if x.strip()]
    out: dict[str, dict[str, Any]] = {}  # by id: the first domain listing it wins
    for dom in domains:
        splits = _load_json(base / dom / _DOMAIN_DATA[source][1])
        tasks = _load_json(base / dom / "tasks.json")
        by_id = {str(t.get("id")): t for t in tasks if isinstance(t, dict)} if isinstance(tasks, list) else {}
        if isinstance(splits, dict):
            ids = [str(sid) for part in parts for sid in (splits.get(part) or [])]
        elif by_id and parts == ["base"] and not tau2:  # talk_bench domains that ship tasks.json only
            ids = list(by_id)
        else:
            ids = []
        for tid in ids:
            task = by_id.get(tid, {})
            scenario = task.get("user_scenario") if isinstance(task.get("user_scenario"), dict) else {}
            if tau2:
                desc = task.get("description") if isinstance(task.get("description"), dict) else {}
                sid, extra = f"{dom}-tau2-{tid}", {}
                summary = desc.get("purpose") or (scenario.get("instructions") or {}).get("reason_for_call")
            else:
                sid, extra = tid, {k: task[k] for k in ("difficulty", "tags") if task.get(k)}
                summary = task.get("description") or scenario.get("reason_for_call")
            out.setdefault(sid, {"id": sid, "domain": dom, "summary": _clip(summary or "", 160), **extra})
    return list(out.values())


def _project_manifest(root: Path) -> dict[str, Any]:
    try:
        doc = yaml.safe_load((root / "talkbench.yaml").read_text())
    except Exception:
        return {}
    return doc if isinstance(doc, dict) else {}


def _sample_configs(root: Path) -> list[dict[str, Any]]:
    """Agent YAMLs shipped in the checkout's samples/ — importable as starting points."""
    out = []
    samples = root / "samples"
    if not samples.is_dir():
        return out
    for f in sorted(samples.iterdir()):
        if not f.is_file() or f.suffix not in (".yaml", ".yml"):
            continue
        try:
            doc = yaml.safe_load(f.read_text())
        except Exception:
            continue
        agent = doc.get("agent") if isinstance(doc, dict) else None
        if isinstance(agent, dict) and agent.get("module"):
            out.append({"file": f.name, "name": agent.get("name"), "module": agent.get("module")})
    return out


def _test_repo() -> Path:
    """The ai-ds-research checkout tests run in: the first one _source_repos() finds
    that has a talk-bench project."""
    for repo in _source_repos():
        if (repo / TEST_PROJECT / "pyproject.toml").is_file():
            return repo
    raise HTTPException(status_code=404, detail="no talk-bench checkout was found to run the test in")


def _project_packages(root: Path) -> set[str]:
    """Top-level Python packages a checkout's own uv workspace members provide
    (libs/*/src/<pkg>, apps/*/src/<pkg>) — the ones importable only from its venv."""
    return {
        p.name for p in (*root.glob("libs/*/src/*"), *root.glob("apps/*/src/*"))
        if p.is_dir() and (p / "__init__.py").is_file()
    }


def _runs_root(root: Path) -> Path:
    """Where talk-bench writes the runs of a checkout: talkbench.yaml's output_dir."""
    return root / str((_project_manifest(root).get("defaults") or {}).get("output_dir") or "data/runs")


def _test_project_root(module: str | None = None) -> Path:
    """The checkout a test runs in: the project whose workspace provides the agent
    module's top-level package (talk-bench-talkdesk for talkdesk_agent.*), else talk-bench."""
    repo = _test_repo()
    pkg = module.split(".", 1)[0] if module else ""
    if pkg:
        for name in SOURCE_PROJECTS:
            root = repo / name
            if (root / "pyproject.toml").is_file() and pkg in _project_packages(root):
                return root
    return repo / TEST_PROJECT


def _describe_project(root: Path) -> dict[str, Any]:
    """One checkout as the Run test page shows it. has_env is talk-bench/.env: the CLI
    refuses to start without that one wherever the test runs (the checkout's own .env
    loads on top of it, see the section comment above)."""
    launcher = _launcher(root)
    tau2_launcher = _launcher(root, "tau2")
    env_root = _talk_bench_lib_root(root) or root
    return {
        "id": str(root),
        "name": root.name,
        "packages": sorted(_project_packages(root)),
        "has_env": (env_root / ".env").is_file(),
        "launcher": " ".join([Path(launcher[0]).name, *launcher[1:]]) if launcher else None,
        "tau2_launcher": " ".join([Path(tau2_launcher[0]).name, *tau2_launcher[1:]]) if tau2_launcher else None,
    }


def _test_project() -> dict[str, Any] | None:
    """The checkouts described for the Run test page: the default one (talk-bench) plus
    every project a test can be routed to with the packages that send it there, the
    samples of all of them, and the domains / splits the data files define."""
    try:
        repo = _test_repo()
    except HTTPException:
        return None
    root = repo / TEST_PROJECT
    projects = [
        _describe_project(repo / name)
        for name in SOURCE_PROJECTS if (repo / name / "pyproject.toml").is_file()
    ]
    samples = [{**s, "project": p["name"]} for p in projects for s in _sample_configs(Path(p["id"]))]
    return {**_describe_project(root), "projects": projects, "samples": samples, "domains": _scan_domains(root)}


def _normalize_agent(raw: Any) -> dict[str, Any]:
    """Validate the agent block the way talk-bench's AgentConfig will, with errors the
    form can show. Unknown keys are dropped."""
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="agent config must be a mapping")
    name = str(raw.get("name") or "").strip()
    module = str(raw.get("module") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="agent.name is required")
    if len(name) > 80:
        raise HTTPException(status_code=400, detail="agent.name must be at most 80 characters")
    if not _AGENT_MODULE_RE.match(module):
        raise HTTPException(
            status_code=400,
            detail="agent.module must be a dotted Python path like voice_agent.openai_realtime.OpenAIRealtimeAgent",
        )
    params = raw.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="agent.params must be a mapping")
    agent: dict[str, Any] = {"name": name, "module": module, "params": params}
    conc = raw.get("agents_max_supported_concurrency")
    if conc not in (None, ""):
        try:
            conc = int(conc)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="agents_max_supported_concurrency must be an integer")
        if conc < 1:
            raise HTTPException(status_code=400, detail="agents_max_supported_concurrency must be at least 1")
        agent["agents_max_supported_concurrency"] = conc
    if isinstance(raw.get("cost"), dict):
        agent["cost"] = raw["cost"]
    mcp = raw.get("mcp")
    if isinstance(mcp, dict) and str(mcp.get("server_url") or "").strip():
        block: dict[str, Any] = {"server_url": str(mcp["server_url"]).strip()}
        transport = mcp.get("identity_transport")
        if isinstance(transport, dict) and transport.get("type"):
            ttype = str(transport["type"])
            if ttype not in _MCP_TRANSPORTS:
                raise HTTPException(
                    status_code=400,
                    detail=f"mcp.identity_transport.type must be one of {', '.join(sorted(_MCP_TRANSPORTS))}",
                )
            block["identity_transport"] = {"type": ttype}
            if transport.get("variable_name"):
                block["identity_transport"]["variable_name"] = str(transport["variable_name"])
        agent["mcp"] = block
    return agent


def _parse_config_text(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(agent block, other top-level blocks) of an agent YAML. talk-bench accepts
    both `agent: {…}` with siblings and a flat agent mapping."""
    try:
        doc = yaml.safe_load(text)
    except Exception as e:  # yaml.YAMLError carries the line/column
        raise HTTPException(status_code=400, detail=f"invalid YAML/JSON: {e}")
    if not isinstance(doc, dict):
        raise HTTPException(status_code=400, detail="config must be a YAML/JSON mapping")
    if isinstance(doc.get("agent"), dict):
        extra = {k: v for k, v in doc.items() if k in _EXTRA_BLOCKS and isinstance(v, dict)}
        return _normalize_agent(doc["agent"]), extra
    return _normalize_agent(doc), {}


def _dump_yaml(doc: dict[str, Any]) -> str:
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _archived_agents() -> list[dict[str, Any]]:
    """Distinct agent configs recorded in archived runs, newest first."""
    by_run = {r["path"]: r for r in build_index()}
    groups: dict[str, dict[str, Any]] = {}
    for run_name, ctx in list(_archived_run_context.items()):
        entry = by_run.get(run_name)
        if not entry:
            continue
        try:
            agent = _normalize_agent(ctx["agent"])  # drops the nulls results.json records
        except HTTPException:
            continue
        key = json.dumps(agent, sort_keys=True, default=str)
        stamp = entry.get("timestamp") or ""
        latest = {
            "timestamp": stamp, "run_path": run_name, "domain": ctx.get("domain"),
            "split": ctx.get("split"), "source": ctx.get("source"),
        }
        g = groups.get(key)
        if g is None:
            groups[key] = {"agent": agent, "yaml": _dump_yaml({"agent": agent}), "run_paths": [run_name], **latest}
        else:
            g["run_paths"].append(run_name)
            if stamp > g["timestamp"]:
                g.update(latest)
    return sorted(groups.values(), key=lambda g: g["timestamp"], reverse=True)


# ---- options, import helpers ----

@app.get("/api/test/options")
def api_test_options():
    return {"project": _test_project(), "archived": _archived_agents(), "data_dir": str(DATA_DIR)}


@app.get("/api/test/scenarios")
def api_test_scenarios(
    source: str = Query("talk_bench"), domain: str = Query(...), split: str = Query("base"),
):
    """What `--domain/--split/--source` resolve to, for the scenario picker (--scenario-ids)."""
    root = _test_project_root()
    domain, split = _check_selection(source, domain, split)
    return {"scenarios": _scenario_catalog(root, source, domain, split)}


@app.get("/api/test/sample")
def api_test_sample(file: str = Query(...), project: str = Query(TEST_PROJECT)):
    if project not in SOURCE_PROJECTS:
        raise HTTPException(status_code=400, detail="unknown project")
    root = _test_repo() / project
    if Path(file).name != file or Path(file).suffix not in (".yaml", ".yml"):
        raise HTTPException(status_code=400, detail="invalid sample file name")
    path = root / "samples" / file
    if not path.is_file():
        raise HTTPException(status_code=404, detail="sample not found")
    return {"text": path.read_text()}


class ConfigTextBody(BaseModel):
    text: str


@app.post("/api/test/parse_config")
def api_parse_config(body: ConfigTextBody):
    if len(body.text) > 200_000:
        raise HTTPException(status_code=413, detail="config text is too large")
    agent, extra = _parse_config_text(body.text)
    try:
        project = _describe_project(_test_project_root(agent["module"]))
    except HTTPException:
        project = None  # no checkout at all: the page already says tests cannot start
    return {"agent": agent, "extra_blocks": extra, "project": project}


# ---- jobs ----

def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _save_job(job: dict[str, Any]) -> None:
    d = _job_dir(job["id"])
    d.mkdir(parents=True, exist_ok=True)
    _write_json(d / "job.json", _public_job(job))


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _get_job(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="no such test")
    return job


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _child_env() -> dict[str, str]:
    """Environment for a test: the viewer's process environment from before its own
    .env was loaded (so the checkout's .env, which the CLI loads itself, is the only
    source of LLM/speech credentials), unbuffered output for live logs, uv on PATH."""
    env = dict(_BASE_ENV)
    for key in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)  # the viewer's own venv must not leak into the checkout's `uv run`
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("NO_COLOR", "1")
    env["TERM"] = "dumb"
    viewer_venv = str(BASE_DIR / ".venv")
    parts = [p for p in env.get("PATH", "").split(os.pathsep) if p and not p.startswith(viewer_venv)]
    uv = _uv_bin()
    for extra in ([str(Path(uv).parent)] if uv else []) + ["/opt/homebrew/bin", "/usr/local/bin"]:
        if extra not in parts and Path(extra).is_dir():
            parts.append(extra)
    env["PATH"] = os.pathsep.join(parts)
    return env


def _new_job(**fields: Any) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:12], "created": _now_iso(), "started": time.time(), "finished": None,
        "status": "running", "pid": None, "exit_code": None, "cmd": [], "cwd": None,
        "run_dir": None, "run_path": None, "synced_at": None, "in_archive": False,
        "progress": {"done": 0, "total": None}, "score": None, "errored": None, "error": None,
        "resume_of": None, "scenario_ids": [], "user_simulator": "talk_bench", "speech_complexity": None,
        "schedule_name": None, "scheduled_for": None,
        **fields,
    }


def _start_job(job: dict[str, Any], cmd: list[str], root: Path) -> None:
    log_path = _job_dir(job["id"]) / "log.txt"
    with log_path.open("ab") as log:
        header = f"$ cd {shlex.quote(str(root))} && {shlex.join(cmd)}\n# started {job['created']}\n\n"
        log.write(header.encode())
        log.flush()
        try:
            proc = subprocess.Popen(
                cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                env=_child_env(), start_new_session=True,  # own process group: cancel kills uv + python
            )
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"could not start talk-bench: {e}")
    job.update(pid=proc.pid, cmd=cmd, cwd=str(root))
    with _jobs_lock:
        _jobs[job["id"]] = job
    _save_job(job)
    threading.Thread(target=_watch_job, args=(job["id"], proc), daemon=True).start()


def _ingest_log(job: dict[str, Any], log_path: Path) -> None:
    """Turn the log lines written since the last poll into progress / score / run dir."""
    try:
        size = log_path.stat().st_size
    except OSError:
        return
    offset = job.get("_log_offset", 0)
    if size <= offset:
        return
    with log_path.open("rb") as f:
        f.seek(offset)
        data = f.read(size - offset)
    lines = (job.get("_log_rest", "") + data.decode("utf-8", errors="replace")).split("\n")
    job["_log_rest"] = lines.pop()  # trailing partial line waits for the next poll
    job["_log_offset"] = size
    for line in lines:
        line = line.rstrip("\r")
        m = _PROGRESS_RE.match(line)
        if m:
            job["progress"] = {"done": int(m.group(1)), "total": int(m.group(2))}
            continue
        m = _LOADED_RE.search(line)
        if m and not job["progress"].get("total"):
            job["progress"]["total"] = int(m.group(1))
            continue
        m = _SCORE_RE.match(line)
        if m:
            job["score"] = {"pct": float(m.group(1)), "passed": int(m.group(2)), "total": int(m.group(3))}
            continue
        m = _ERRORS_RE.match(line)
        if m:
            job["errored"] = int(m.group(1))
            continue
        m = _RESULTS_RE.match(line)
        if m:
            run_dir = Path(m.group(1))
            if not run_dir.is_absolute():
                run_dir = Path(job["cwd"]) / run_dir
            job["run_dir"] = str(run_dir)
            continue
        if line.startswith(("Error", "Configuration error", "Invalid YAML")):
            job["error"] = line[:400]


def _locate_run_dir(job: dict[str, Any]) -> None:
    """Find the run dir the CLI created for this job: it writes run_config.json first,
    long before the final `Results:` line. Match the agent/domain prefix and the start time."""
    if not (job.get("runs_root") or job.get("cwd")) or not job.get("agent_name") or not job.get("domain"):
        return
    runs_root = Path(job.get("runs_root") or Path(job["cwd"]) / "data" / "runs")
    if not runs_root.is_dir():
        return
    prefix = f"{_slug(job['agent_name'])}-{_slug(job['domain'])}-"
    claimed = {j.get("run_dir") for j in list(_jobs.values()) if j is not job}
    try:  # a dir born after the job's last log write belongs to a later run
        log_mtime = (_job_dir(job["id"]) / "log.txt").stat().st_mtime
    except OSError:
        log_mtime = float("inf")
    best: tuple[float, Path] | None = None
    for d in runs_root.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix) or str(d) in claimed:
            continue
        try:
            st = d.stat()
        except OSError:
            continue
        born = getattr(st, "st_birthtime", st.st_mtime)
        if born < job["started"] - 5 or born > log_mtime + 5 or not (d / "run_config.json").exists():
            continue
        if best is None or born < best[0]:
            best = (born, d)
    if best:
        job["run_dir"] = str(best[1])


def _in_archive(path: str | Path) -> bool:
    """Whether a run dir (or the folder runs are written into) sits in the archive itself."""
    p = Path(path).resolve()
    return DATA_DIR.resolve() in (p, p.parent)


def _sync_job_run(job: dict[str, Any]) -> bool:
    """Clone the job's run directory into the archive (a no-op when unchanged). A run
    talk-bench writes straight into the archive is only re-read: True = the index was refreshed."""
    run_dir = Path(job["run_dir"]) if job.get("run_dir") else None
    if not run_dir or not run_dir.is_dir():
        return False
    if not (run_dir / "results.json").is_file() and not _scenario_dirs(run_dir):
        return False  # nothing to browse yet (a config error leaves only run_config.json behind)
    if _in_archive(run_dir):
        job["run_path"] = run_dir.name
        job["synced_at"] = time.time()  # tells the page to reload the run list
        _invalidate_index()
        return True
    changed = _mirror_run(run_dir) is not None
    job["run_path"] = run_dir.name
    if changed:
        job["synced_at"] = time.time()
        _undelete(run_dir.name)
        _invalidate_index()
    return changed


def _last_log_lines(log_path: Path, n: int = 3) -> str:
    try:
        with log_path.open("rb") as f:
            f.seek(max(0, log_path.stat().st_size - 4000))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    return " | ".join(lines[-n:])[:400]


def _watch_job(job_id: str, proc: subprocess.Popen | None) -> None:
    """Follow one test until its process ends, then settle its status and archive the run.
    proc is None for a job re-attached after a viewer restart (poll the pid instead)."""
    job = _jobs[job_id]
    log_path = _job_dir(job_id) / "log.txt"
    last_snapshot = time.time()
    last_progress: tuple[int, int] | None = None
    while True:
        alive = (proc.poll() is None) if proc is not None else _pid_alive(job.get("pid"))
        _ingest_log(job, log_path)
        if not job.get("run_dir"):
            _locate_run_dir(job)
        # a run written straight into the archive costs nothing to re-read, so it is refreshed often
        interval = _LIVE_REFRESH_S if job.get("run_dir") and _in_archive(job["run_dir"]) else _SNAPSHOT_INTERVAL_S
        if alive and job.get("run_dir") and time.time() - last_snapshot >= interval:
            last_snapshot = time.time()
            run_dir = Path(job["run_dir"])
            try:
                progress = (len(_scenario_dirs(run_dir)), sum(1 for _ in run_dir.glob("*/run_meta.json")))
                if progress != last_progress:  # only when a scenario started or finished since
                    last_progress = progress
                    _sync_job_run(job)
            except Exception as e:
                job["error"] = f"archive snapshot failed: {e}"
        if not alive:
            break
        time.sleep(1.0)
    exit_code = proc.returncode if proc is not None else None
    with _jobs_lock:
        job["exit_code"] = exit_code
        job["finished"] = time.time()
        if job["status"] == "cancelling":
            job["status"] = "cancelled"
        elif exit_code in (0, 1) or (exit_code is None and job.get("score")):  # Score: is printed only once the run completes
            job["status"] = "done"  # 0 = every scenario passed, 1 = some did not; the run completed
        elif exit_code is None:  # followed by pid after a viewer restart: no exit code to read
            job["status"] = "interrupted"
            job["error"] = job.get("error") or "the viewer restarted while this test was running"
        else:
            job["status"] = "error"  # 2 = configuration, 3 = runtime
        if job["status"] == "done" and job.get("errored") is None:
            job["errored"] = 0  # no Errors: line after Score:
    if job["status"] == "error" and not job.get("error"):
        job["error"] = _last_log_lines(log_path)
    try:
        _sync_job_run(job)
        # archived in full: the checkout's copy goes too, unless --resume still has work in it
        if job["status"] == "done" and job.get("run_dir") and job.get("project"):
            if _release_source(Path(job["project"]), Path(job["run_dir"]), quiet_s=0):
                job["checkout_removed"] = True
    except Exception as e:
        job["error"] = ((job.get("error") or "") + f" · archive sync failed: {e}").strip(" ·")
    _save_job(job)


def _recover_jobs() -> None:
    """Reload job records at startup; re-attach to tests still running (they live in
    their own session, so a viewer restart does not stop them)."""
    if not JOBS_DIR.is_dir():
        return
    for d in sorted(JOBS_DIR.iterdir()):
        job = _load_json(d / "job.json") if d.is_dir() else None
        if not isinstance(job, dict) or not job.get("id"):
            continue
        job = _new_job(**{k: v for k, v in job.items() if not k.startswith("_")})
        if job["status"] == "done" and job["errored"] is None and job.get("run_dir"):
            # recorded before errored was tracked: the run's results.json tells
            res = _load_json(Path(job["run_dir"]) / "results.json")
            job["errored"] = res.get("scenarios_errored") if isinstance(res, dict) else None
        _jobs[job["id"]] = job
    # once every record is loaded, so _locate_run_dir skips the run dirs the others claim
    for job in list(_jobs.values()):
        if job.get("status") in ("running", "cancelling"):
            # still running (its own session survives a viewer restart): followed to its end;
            # ended while the viewer was down: the watcher's first pass reads the log, finds
            # the run dir and settles it (done / cancelled / interrupted), then archives it
            threading.Thread(target=_watch_job, args=(job["id"], None), daemon=True).start()


_recover_jobs()


def _check_token(value: str, label: str, pattern: str) -> str:
    value = value.strip()
    if not value or len(value) > 200 or not re.match(pattern, value):
        raise HTTPException(status_code=400, detail=f"invalid {label}: {value[:60]!r}")
    return value


def _check_selection(source: str, domain: str, split: str) -> tuple[str, str]:
    if source not in ("tau2", "talk_bench"):
        raise HTTPException(status_code=400, detail="source must be tau2 or talk_bench")
    return _check_token(domain, "domain", r"^[a-z0-9_\-]+$"), _check_token(split or "base", "split", r"^[A-Za-z0-9_:,\-]+$")


class TestJobBody(BaseModel):
    config_text: str  # the agent YAML, written verbatim and passed as --config
    source: str = "talk_bench"
    domain: str
    split: str = "base"
    scenario_ids: list[str] = []  # optional --scenario-ids; empty = the whole split
    user_simulator: str = "talk_bench"  # talk_bench | tau2 (voice agents only)
    speech_complexity: str | None = None  # --speech-complexity, tau2 source only


@app.get("/api/test/jobs")
def api_test_jobs():
    jobs = sorted(list(_jobs.values()), key=lambda j: j.get("started") or 0, reverse=True)
    return {"jobs": [_public_job(j) for j in jobs[:200]]}


def _prepare_test(body: TestJobBody) -> dict[str, Any]:
    """Check a test request the way talk-bench will and resolve the checkout it runs in.
    Start test and every scheduled run go through this."""
    if len(body.config_text) > 200_000:
        raise HTTPException(status_code=413, detail="agent config is too large")
    agent, _extra = _parse_config_text(body.config_text)  # same checks talk-bench's AgentConfig makes
    if body.user_simulator not in USER_SIMULATORS:
        raise HTTPException(status_code=400, detail=f"user_simulator must be one of {', '.join(USER_SIMULATORS)}")
    if body.user_simulator == "tau2" and _is_text_module(agent["module"]):
        raise HTTPException(
            status_code=400,
            detail="the tau2 user simulator option is for voice agents — text runs on tau2 tasks already use tau2's own simulator",
        )
    speech_complexity = body.speech_complexity or None
    if speech_complexity is not None:
        if body.source != "tau2":
            raise HTTPException(status_code=400, detail="--speech-complexity applies to tau2 tasks only")
        if speech_complexity not in SPEECH_COMPLEXITIES:
            raise HTTPException(status_code=400, detail=f"speech_complexity must be one of {', '.join(SPEECH_COMPLEXITIES)}")
    root = _test_project_root(agent["module"])
    launcher = _launcher(root, body.user_simulator)
    if launcher is None:
        raise HTTPException(status_code=500, detail="neither uv nor .venv/bin/talk-bench was found for this checkout")
    domain, split = _check_selection(body.source, body.domain, body.split)
    scenario_ids: list[str] = []
    for sid in body.scenario_ids:
        sid = sid.strip()
        if not sid or sid in scenario_ids:
            continue
        if len(sid) > MAX_SCENARIO_ID_LEN or not _ID_RE.match(sid):
            raise HTTPException(status_code=400, detail=f"invalid scenario id: {sid[:60]!r}")
        scenario_ids.append(sid)
    if len(scenario_ids) > MAX_SCENARIO_IDS:
        raise HTTPException(status_code=400, detail=f"at most {MAX_SCENARIO_IDS} scenario ids per test")
    return {
        "agent": agent, "root": root, "launcher": launcher, "config_text": body.config_text,
        "source": body.source, "domain": domain, "split": split, "scenario_ids": scenario_ids,
        "user_simulator": body.user_simulator, "speech_complexity": speech_complexity,
    }


def _running_tests() -> int:
    return sum(1 for j in list(_jobs.values()) if j["status"] in ("running", "cancelling"))


def _launch_test(t: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """Start a test _prepare_test checked; fields go into the job record."""
    with _start_lock:
        if _running_tests() >= MAX_RUNNING_TESTS:
            raise HTTPException(status_code=409, detail=f"{MAX_RUNNING_TESTS} tests are already running — wait for one to finish")
        root, agent = t["root"], t["agent"]
        job = _new_job(
            project=str(root), project_name=root.name, runs_root=str(DATA_DIR), in_archive=True,
            agent_name=agent["name"], module=agent["module"],
            source=t["source"], domain=t["domain"], split=t["split"], scenario_ids=t["scenario_ids"],
            user_simulator=t["user_simulator"], speech_complexity=t["speech_complexity"], **fields,
        )
        d = _job_dir(job["id"])
        d.mkdir(parents=True, exist_ok=True)
        config_path = d / "agent.yaml"
        config_path.write_text(t["config_text"].rstrip() + "\n")
        cmd = [
            *t["launcher"], "run", "--config", str(config_path),
            "--domain", t["domain"], "--split", t["split"], "--source", t["source"],
            "--output-dir", str(DATA_DIR),  # the run lands in the archive: nothing to copy
        ]
        if t["scenario_ids"]:
            cmd += ["--scenario-ids", ",".join(t["scenario_ids"])]
        if t["speech_complexity"]:
            cmd += ["--speech-complexity", t["speech_complexity"]]
        _start_job(job, cmd, root)
    return job


@app.post("/api/test/jobs")
def api_start_test(body: TestJobBody):
    """Start `talk-bench run --config <agent.yaml> --domain X --split Y --source S
    [--scenario-ids a,b] [--speech-complexity <preset>]` in the checkout whose workspace
    provides the agent module — through tau2_user_sim/launch.py when the tau2 user
    simulator is picked."""
    return {"job": _public_job(_launch_test(_prepare_test(body)))}


@app.get("/api/test/jobs/{job_id}")
def api_test_job(job_id: str, offset: int = 0):
    job = _get_job(job_id)
    log_path = _job_dir(job_id) / "log.txt"
    text, size, more = "", 0, False
    try:
        size = log_path.stat().st_size
        if offset < 0 or offset > size:
            offset = 0
        with log_path.open("rb") as f:
            f.seek(offset)
            data = f.read(_LOG_CHUNK)
        text = data.decode("utf-8", errors="replace")
        offset += len(data)
        more = offset < size
    except OSError:
        pass
    config_path = _job_dir(job_id) / "agent.yaml"
    return {
        "job": _public_job(job), "log": text, "next_offset": offset, "more": more,
        "config_yaml": config_path.read_text() if config_path.is_file() else None,
    }


def _log_export_header(job: dict[str, Any], config_yaml: str | None) -> str:
    """`#` lines that make an exported log readable on its own: what ran, how it ended, the agent.yaml."""
    test = [job.get("source") or "?", f"{job.get('domain') or '?'}/{job.get('split') or '?'}"]
    if job.get("scenario_ids"):
        test.append(f"--scenario-ids {','.join(job['scenario_ids'])}")
    if job.get("user_simulator") == "tau2":
        test.append("tau2 user simulator")
    if job.get("speech_complexity"):
        test.append(f"speech complexity {job['speech_complexity']}")
    status = [job.get("status") or "?"]
    if job.get("exit_code") is not None:
        status.append(f"exit {job['exit_code']}")
    if job.get("score"):
        s = job["score"]
        status.append(f"{s['passed']}/{s['total']} passed ({s['pct']:g}%)")
    elif (job.get("progress") or {}).get("total"):
        status.append(f"{job['progress']['done']}/{job['progress']['total']} scenarios")
    if job.get("errored"):
        status.append(f"{job['errored']} errored")
    if job.get("started"):
        secs = int((job.get("finished") or time.time()) - job["started"])
        status.append(f"{secs // 3600}h{secs // 60 % 60:02d}m{secs % 60:02d}s" if secs >= 3600 else f"{secs // 60}m{secs % 60:02d}s")
    lines = [
        f"talk-bench test {job['id']} · exported {_now_iso()} from the runs viewer",
        f"agent    {job.get('agent_name') or '?'} ({job.get('module') or '?'})",
        f"test     {' · '.join(test)}",
        f"status   {' · '.join(status)}",
    ]
    if job.get("run_dir"):
        lines.append(f"run dir  {job['run_dir']}")
    if job.get("error"):
        lines.append(f"error    {job['error']}")
    if config_yaml:
        lines += ["", "agent.yaml", *(f"  {ln}" for ln in config_yaml.rstrip().splitlines())]
    return "".join(f"# {ln}".rstrip() + "\n" for ln in lines) + "\n"


@app.get("/api/test/jobs/{job_id}/log")
def api_export_test_log(job_id: str):
    """The test's whole log as a download (up to now while it runs), after a header
    with its settings, outcome and agent.yaml."""
    job = _get_job(job_id)
    d = _job_dir(job_id)
    config_path = d / "agent.yaml"
    header = _log_export_header(job, config_path.read_text() if config_path.is_file() else None)

    def body():
        yield header.encode()
        try:
            with (d / "log.txt").open("rb") as f:
                while chunk := f.read(_LOG_CHUNK):
                    yield chunk
        except OSError:
            yield b"(no log was written)\n"

    started = (job.get("created") or "").replace("-", "").replace(":", "")
    name = "-".join(p for p in (_slug(job.get("agent_name") or ""), _slug(job.get("domain") or ""), _slug(job.get("split") or ""), started, job_id) if p)
    return StreamingResponse(body(), media_type="text/plain; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="{name}.log"'})


@app.post("/api/test/jobs/{job_id}/cancel")
def api_cancel_test(job_id: str):
    job = _get_job(job_id)
    if job["status"] not in ("running", "cancelling"):
        raise HTTPException(status_code=409, detail="this test is not running")
    job["status"] = "cancelling"
    pid = job.get("pid")

    def _signal(sig: int) -> None:
        try:
            os.killpg(pid, sig)  # started with start_new_session: the pid is its group's id
        except OSError:
            pass

    _signal(signal.SIGTERM)  # talk-bench has no graceful stop; the run can be resumed afterwards

    def _force() -> None:
        time.sleep(8)
        if _pid_alive(pid):
            _signal(signal.SIGKILL)

    threading.Thread(target=_force, daemon=True).start()
    _save_job(job)
    return {"job": _public_job(job)}


def _launch_resume(run_dir: str, root: Path, user_simulator: str, config_text: str | None, **fields: Any) -> dict[str, Any]:
    """Start `talk-bench run --resume <run dir>` in root; fields go into the job record.
    The run dir is absolute, so the cwd only has to be able to import the agent."""
    with _start_lock:
        if any(j["status"] in ("running", "cancelling") and j.get("run_dir") == run_dir for j in list(_jobs.values())):
            raise HTTPException(status_code=409, detail="a test is already running in this run directory")
        if _running_tests() >= MAX_RUNNING_TESTS:
            raise HTTPException(status_code=409, detail=f"{MAX_RUNNING_TESTS} tests are already running — wait for one to finish")
        if not (root / "pyproject.toml").is_file():
            raise HTTPException(status_code=404, detail="the checkout this test ran in is gone")
        # a resumed run keeps the user simulator its first scenarios ran with
        launcher = _launcher(root, user_simulator)
        if launcher is None:
            raise HTTPException(status_code=500, detail="neither uv nor .venv/bin/talk-bench was found for this checkout")
        job = _new_job(**fields, project=str(root), project_name=root.name, user_simulator=user_simulator,
                       run_dir=run_dir, in_archive=_in_archive(run_dir))
        _job_dir(job["id"]).mkdir(parents=True, exist_ok=True)
        if config_text:
            (_job_dir(job["id"]) / "agent.yaml").write_text(config_text)
        _start_job(job, [*launcher, "run", "--resume", run_dir], root)
    return job


@app.post("/api/test/jobs/{job_id}/resume")
def api_resume_test(job_id: str):
    """`talk-bench run --resume <run dir>`: redo errored / unfinished scenarios of a test
    and merge into the same results.json. Runs in the checkout the agent module belongs
    to, which may differ from where an old job was recorded."""
    parent = _get_job(job_id)
    if parent["status"] in ("running", "cancelling"):
        raise HTTPException(status_code=409, detail="this test is still running")
    run_dir = parent.get("run_dir")
    if not run_dir or not (Path(run_dir) / "run_config.json").is_file():
        raise HTTPException(status_code=409, detail="this test left no resumable run directory")
    try:
        root = _test_project_root(parent.get("module"))
    except HTTPException:
        root = Path(parent["project"])
    config = _job_dir(parent["id"]) / "agent.yaml"
    job = _launch_resume(
        run_dir, root, parent.get("user_simulator") or "talk_bench",
        config.read_text() if config.is_file() else None, resume_of=parent["id"],
        **{k: parent.get(k) for k in (
            "runs_root", "agent_name", "module", "source", "domain", "split", "scenario_ids", "speech_complexity",
        )},
    )
    return {"job": _public_job(job)}


# talk-bench's --resume runs a scenario again when its run_meta.json is missing or ends in
# one of these (runner/resume.py), and scores the finished ones again while results.json
# is missing or was written before scoring completed
_RESUME_REDO_END_REASONS = {"error", "error_agent"}
_RUN_NAME_RE = re.compile(r"^[\w][\w.\-]*$")


@lru_cache(maxsize=8192)
def _end_reason_at(path: str, _mtime_ns: int, _size: int) -> str | None:
    """end_reason of one run_meta.json, read once per version of the file."""
    meta = _load_json(Path(path))
    return str(meta["end_reason"]) if isinstance(meta, dict) and meta.get("end_reason") else None


def _archive_resume_project() -> Path | None:
    """The checkout an archived run is resumed from when its agent module routes nowhere."""
    try:
        return _test_repo() / TEST_PROJECT
    except HTTPException:
        return None


def _source_runs() -> list[tuple[Path, Path]]:
    """(project root, run dir) of every run --resume can pick up (it has run_config.json):
    runs talk-bench wrote into a checkout, then runs a test of this page wrote straight
    into the archive (other archived runs are not scanned: thousands of multi-MB
    run_meta.json). The checkout's copy wins a name: the archived one may be a snapshot."""
    out, seen = [], set()
    for repo in _source_repos():
        for name in SOURCE_PROJECTS:
            root = repo / name
            runs = _runs_root(root)
            if not (root / "pyproject.toml").is_file() or not runs.is_dir():
                continue
            for d in sorted(runs.iterdir()):
                if not d.name.startswith(".") and d.name not in seen and (d / "run_config.json").is_file():
                    seen.add(d.name)
                    out.append((root, d))
    project = _archive_resume_project()
    if project is not None:
        names = {Path(j["run_dir"]).name for j in list(_jobs.values()) if j.get("in_archive") and j.get("run_dir")}
        for name in sorted(names - seen):
            if (DATA_DIR / name / "run_config.json").is_file():
                out.append((project, DATA_DIR / name))
    return out


def _find_source_run(name: str) -> tuple[Path, Path] | None:
    if not _RUN_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="invalid run name")
    for repo in _source_repos():
        for proj in SOURCE_PROJECTS:
            root = repo / proj
            d = _runs_root(root) / name
            if (root / "pyproject.toml").is_file() and (d / "run_config.json").is_file():
                return root, d
    project = _archive_resume_project()
    if project is not None and (DATA_DIR / name / "run_config.json").is_file():
        return project, DATA_DIR / name  # resumed in place, in the archive
    return None


def _resume_state(project: Path, run_dir: Path) -> dict[str, Any] | None:
    """What `talk-bench run --resume <run dir>` would do, read from the files its own
    classification reads: redo = scenarios not finished (counted against the planned
    total when it is known), rescore = finished ones still unscored. None when
    run_config.json does not hold a usable config. _agent stays server-side (_public_job)."""
    doc = _load_json(run_dir / "run_config.json")
    rc = doc.get("run_config") if isinstance(doc, dict) else None
    agent = rc.get("agent_config") if isinstance(rc, dict) else None
    if not isinstance(agent, dict) or not agent.get("module"):
        return None
    trials = rc.get("trials") if isinstance(rc.get("trials"), int) and rc["trials"] > 0 else 1
    scenario_ids = [str(x) for x in rc.get("scenario_ids") or []]
    source, domain, split = rc.get("source") or "talk_bench", str(rc.get("domain") or ""), str(rc.get("split") or "base")
    results = _load_json(run_dir / "results.json")
    results = results if isinstance(results, dict) else None
    sessions = _scenario_dirs(run_dir)
    finished, last_write = 0, run_dir.stat().st_mtime
    for sdir in sessions:
        meta = sdir / "run_meta.json"
        try:
            st = meta.stat()
        except OSError:
            last_write = max(last_write, sdir.stat().st_mtime)
            continue
        last_write = max(last_write, st.st_mtime)
        reason = _end_reason_at(str(meta), st.st_mtime_ns, st.st_size)
        if reason and reason.lower() not in _RESUME_REDO_END_REASONS:
            finished += 1
    planned = None  # every (scenario, trial) the run covers
    if results and isinstance(results.get("scenarios_total"), int):
        planned = results["scenarios_total"]
    elif scenario_ids:
        planned = len(scenario_ids) * trials
    elif source in _DOMAIN_DATA and domain and not rc.get("tags") and not rc.get("jailbreak"):
        planned = len(_scenario_catalog(project, source, domain, split)) * trials or None
    redo = max(planned - finished, 0) if planned is not None else len(sessions) - finished
    scored = bool(results and results.get("talk_bench_evaluation_complete"))
    running = next((j["id"] for j in list(_jobs.values())
                    if j["status"] in ("running", "cancelling") and j.get("run_dir") == str(run_dir)), None)
    try:
        runs_in = _test_project_root(agent["module"])  # where --resume will run: the module's checkout
    except HTTPException:
        runs_in = project
    return {
        "name": run_dir.name, "run_dir": str(run_dir), "project": str(runs_in), "project_name": runs_in.name,
        "timestamp": _parse_timestamp(run_dir.name), "in_archive": (DATA_DIR / run_dir.name).is_dir(),
        "_agent": agent, "agent_name": agent.get("name") or run_dir.name, "module": agent["module"],
        "source": source, "domain": domain, "split": split, "scenario_ids": scenario_ids,
        "speech_complexity": rc.get("speech_complexity"), "trials": trials,
        "user_simulator": (_run_user_simulator(run_dir) or {}).get("engine") or "talk_bench",
        "scripted": (run_dir / "scripted_user_simulator.json").is_file(),
        "planned": planned, "started": len(sessions), "finished": finished, "redo": redo,
        "rescore": 0 if scored else finished, "has_results": results is not None,
        "last_write": last_write, "running_job": running,
    }


@app.get("/api/test/resumable")
def api_test_resumable(name: str | None = Query(None)):
    """Runs (_source_runs) that --resume has work for: some scenarios errored, never
    finished or were never scored. With name: that one run's state, whatever it is
    (null when neither a checkout nor the archive holds it with a run_config.json)."""
    if name is not None:
        found = _find_source_run(name)
        st = _resume_state(*found) if found else None
        return {"run": _public_job(st) if st else None}
    runs = []
    for project, run_dir in _source_runs():
        st = _resume_state(project, run_dir)
        if _has_resume_work(st):
            runs.append(_public_job(st))
    runs.sort(key=lambda r: r["timestamp"] or "", reverse=True)
    return {"runs": runs}


def _has_resume_work(st: dict[str, Any] | None) -> bool:
    # a run that left no scenario directory never got going: start the test again instead
    return bool(st and st["started"] and (st["redo"] or st["rescore"]))


def _resumable(project: Path, run_dir: Path) -> bool:
    """Whether the Run test page lists this checkout run as one to resume."""
    return _has_resume_work(_resume_state(project, run_dir))


class ResumeRunBody(BaseModel):
    name: str  # run directory name, the same in the checkout's data/runs and in the archive


@app.post("/api/test/resume_run")
def api_resume_run(body: ResumeRunBody):
    """`talk-bench run --resume <run dir>` for a run in a checkout or the archive, wherever
    it was started from: the run dir's run_config.json says what to run (agent, task set),
    so nothing else is passed. Runs in the checkout the agent module belongs to."""
    found = _find_source_run(body.name)
    if not found:
        raise HTTPException(
            status_code=404,
            detail="neither a checkout nor the archive holds this run with a run_config.json — runs made before --resume existed cannot be resumed",
        )
    project, run_dir = found
    st = _resume_state(project, run_dir)
    if st is None:
        raise HTTPException(status_code=409, detail="this run's run_config.json holds no usable agent config")
    if st["scripted"]:
        raise HTTPException(status_code=409, detail="this run used scripted_user_sim, which has been removed — it cannot be resumed")
    try:
        config_text = _dump_yaml({"agent": _normalize_agent(st["_agent"])})
    except HTTPException:
        config_text = _dump_yaml({"agent": st["_agent"]})
    # the newest test of this page that worked on the run, if any
    parent = max((j for j in list(_jobs.values()) if j.get("run_dir") == st["run_dir"]),
                 key=lambda j: j.get("started") or 0, default=None)
    job = _launch_resume(
        st["run_dir"], Path(st["project"]), st["user_simulator"], config_text,
        resume_of=parent["id"] if parent else None, runs_root=str(run_dir.parent),
        agent_name=st["agent_name"], module=st["module"], source=st["source"], domain=st["domain"],
        split=st["split"], scenario_ids=st["scenario_ids"], speech_complexity=st["speech_complexity"],
    )
    return {"job": _public_job(job)}


@app.post("/api/test/jobs/{job_id}/sync")
def api_sync_test(job_id: str):
    """Copy the run directory into the archive now (partial results of a running test)."""
    job = _get_job(job_id)
    if not job.get("run_dir"):
        _locate_run_dir(job)
    if not job.get("run_dir"):
        raise HTTPException(status_code=409, detail="talk-bench has not created a run directory yet")
    changed = _sync_job_run(job)
    _save_job(job)
    return {"job": _public_job(job), "changed": changed}


@app.delete("/api/test/jobs/{job_id}")
def api_delete_test(job_id: str):
    job = _get_job(job_id)
    if job["status"] in ("running", "cancelling"):
        raise HTTPException(status_code=409, detail="stop the test before removing it")
    with _jobs_lock:
        _jobs.pop(job_id, None)
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    return {"deleted": job_id}


# ---- scheduled tests ----
#
# A scheduled test is a saved Start-test request (agent YAML, source / domain / split,
# scenario ids, user simulator) plus the moment to start it, once (epoch seconds; the
# page picks it in the browser's time zone). A thread in the viewer starts it when it is
# due, through the same _prepare_test / _launch_test as the button; the job records the
# schedule's name and time, and the schedule is removed.
#   - missed while the viewer was stopped or the Mac asleep, it starts as soon as the
#     viewer is back
#   - while MAX_RUNNING_TESTS tests run, it waits for a free slot
#   - if it cannot start (the YAML no longer loads, the checkout is gone), it stays in the
#     list with the error until it is edited or deleted
# Schedules live in <data dir>/.test_schedules.json, changed under an flock so a second
# viewer process on the same data dir cannot start one twice.

SCHEDULES_FILE = DATA_DIR / ".test_schedules.json"
_SCHEDULES_LOCK_FILE = DATA_DIR / ".test_schedules.lock"
MAX_SCHEDULES = 50
MAX_SCHEDULE_AHEAD_S = 366 * 86400
_SCHEDULER_TICK_S = 20
_schedules_lock = threading.Lock()


@contextmanager
def _schedules_txn():
    """Yield the schedule list for reading or changing; changes are saved on exit.
    One thread, and one process on this data dir, at a time."""
    with _schedules_lock, _SCHEDULES_LOCK_FILE.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # released when the file closes
        data = _load_json(SCHEDULES_FILE) if SCHEDULES_FILE.exists() else []
        if not isinstance(data, list):
            raise HTTPException(status_code=500, detail=f"{SCHEDULES_FILE.name} is unreadable — fix or remove it")
        schedules = [s for s in data if isinstance(s, dict) and s.get("id") and isinstance(s.get("run_at"), (int, float))]
        before = json.dumps(schedules, sort_keys=True)
        yield schedules
        if json.dumps(schedules, sort_keys=True) != before:
            _write_json(SCHEDULES_FILE, schedules)


def _find_schedule(schedules: list[dict[str, Any]], sid: str) -> dict[str, Any]:
    s = next((s for s in schedules if s["id"] == sid), None)
    if s is None:
        raise HTTPException(status_code=404, detail="no such scheduled test")
    return s


def _public_schedule(s: dict[str, Any]) -> dict[str, Any]:
    """A schedule for the list: the test settings without the agent YAML."""
    return {**s, "test": {k: v for k, v in (s.get("test") or {}).items() if k != "config_text"}}


def _scheduler_tick() -> None:
    now = time.time()
    with _schedules_txn() as schedules:
        for s in list(schedules):
            if s.get("error") or now < s["run_at"]:
                continue
            if _running_tests() >= MAX_RUNNING_TESTS:
                s["waiting"] = f"{MAX_RUNNING_TESTS} tests are already running"
                continue  # starts as soon as a slot frees up
            try:
                _launch_test(_prepare_test(TestJobBody(**s["test"])), schedule_name=s["name"], scheduled_for=s["run_at"])
            except HTTPException as e:
                s["error"] = str(e.detail)[:400]
            except Exception as e:  # a broken schedule must not stop the others
                s["error"] = repr(e)[:400]
            else:
                schedules.remove(s)  # from here on it is a test in the jobs list
                continue
            s.pop("waiting", None)


def _scheduler_loop() -> None:
    while True:
        try:
            _scheduler_tick()
        except Exception as e:
            print(f"test scheduler: {e!r}", flush=True)
        time.sleep(_SCHEDULER_TICK_S)


class ScheduleBody(BaseModel):
    name: str = ""  # default: agent · domain/split
    run_at: float  # epoch seconds
    test: TestJobBody


def _schedule_fields(body: ScheduleBody) -> dict[str, Any]:
    now = time.time()
    if not now < body.run_at <= now + MAX_SCHEDULE_AHEAD_S:
        raise HTTPException(status_code=400, detail="pick a start time in the future, at most a year ahead")
    t = _prepare_test(body.test)  # rejected now, not at 02:00
    name = " ".join(body.name.split()) or f"{t['agent']['name']} · {t['domain']}/{t['split']}"[:120]
    if len(name) > 120:
        raise HTTPException(status_code=400, detail="the name must be at most 120 characters")
    return {
        "name": name, "run_at": body.run_at,
        "agent_name": t["agent"]["name"], "module": t["agent"]["module"],
        "test": {
            k: t[k] for k in
            ("config_text", "source", "domain", "split", "scenario_ids", "user_simulator", "speech_complexity")
        },
    }


@app.get("/api/test/schedules")
def api_test_schedules():
    with _schedules_txn() as schedules:
        out = [_public_schedule(s) for s in sorted(schedules, key=lambda s: s["run_at"])]
    return {"schedules": out}


@app.post("/api/test/schedules")
def api_create_schedule(body: ScheduleBody):
    fields = _schedule_fields(body)
    with _schedules_txn() as schedules:
        if len(schedules) >= MAX_SCHEDULES:
            raise HTTPException(status_code=409, detail=f"at most {MAX_SCHEDULES} scheduled tests")
        s = {"id": uuid.uuid4().hex[:12], "created": _now_iso(), **fields}
        schedules.append(s)
    return {"schedule": _public_schedule(s)}


@app.get("/api/test/schedules/{sid}")
def api_test_schedule(sid: str):
    with _schedules_txn() as schedules:
        s = _find_schedule(schedules, sid)
        return {"schedule": {**_public_schedule(s), "test": s["test"]}}


@app.put("/api/test/schedules/{sid}")
def api_update_schedule(sid: str, body: ScheduleBody):
    """Replace the test and the start time; a schedule that failed to start is pending again."""
    fields = _schedule_fields(body)
    with _schedules_txn() as schedules:
        s = _find_schedule(schedules, sid)
        s.update(fields, updated=_now_iso())
        s.pop("error", None)
        s.pop("waiting", None)
    return {"schedule": _public_schedule(s)}


@app.delete("/api/test/schedules/{sid}")
def api_delete_schedule(sid: str):
    with _schedules_txn() as schedules:
        schedules.remove(_find_schedule(schedules, sid))
    return {"deleted": sid}



# ---- settings: the checkout's config files and this viewer's .env, edited from the page ----
#
# The Settings page shows this viewer's .env and, for each project of the checkout tests run in
# (talk-bench, talk-bench-talkdesk), its talkbench.yaml, its .env and its samples/ agent YAMLs; the
# Provision page adds talk-bench-talkdesk's samples/provision/ YAMLs. Secret values (names with KEY,
# TOKEN, SECRET, PASSWORD or CREDENTIAL) never leave the server: the page gets their last four
# characters and sends a value only to replace one. Writing, provisioning and the MCP server controls
# answer only requests made on this Mac: the viewer is shared through a public tunnel, and whoever can
# point OPENAI_BASE_URL or TALKDESK_TOKEN_URL at a host of their own collects the keys sent there, so
# write access is as good as read access to the secrets. Each save keeps the file it replaced under
# <data dir>/.settings_history (the last 10 per file); talkbench.yaml and agent YAMLs are rewritten
# with ruamel.yaml so their comments survive.

VIEWER_PROJECT = "viewer"
TALKDESK_PROJECT = "talk-bench-talkdesk"
SETTINGS_HISTORY_DIR = DATA_DIR / ".settings_history"
_SETTINGS_KEEP = 10
_settings_lock = threading.Lock()
_LOCAL_NAMES = {"localhost", "127.0.0.1", "::1"}
_SECRET_RE = re.compile(r"KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|TOKEN$", re.I)
_ENV_LINE_RE = re.compile(
    r"^(?P<lead>\s*)(?P<hash>#\s*)?(?P<export>export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)(?P<sep>[ \t]*=)(?P<rest>.*)$"
)
_ENV_HEADING_RE = re.compile(r"^#\s*-{2,}\s*(?P<title>.*?)\s*-*\s*$")  # "# -- Azure Speech ---------"
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_YAML_FILE_RE = re.compile(r"^\w[\w.\-]*\.ya?ml$")
# viewer settings read once at start-up: a change to them needs a restart (the rest apply at once)
_VIEWER_RESTART_KEYS = {"VIEWER_DATA_DIR"}


def _is_local_request(request: Request) -> bool:
    """Made on this Mac to the viewer itself: not forwarded by the ngrok tunnel (which adds
    X-Forwarded-For and keeps the public Host) and not posted here by a page of another site."""
    if (request.client.host if request.client else "") not in _LOCAL_NAMES:
        return False
    if any(h in request.headers for h in ("x-forwarded-for", "x-forwarded-host", "forwarded", "x-real-ip")):
        return False
    if (urlsplit("//" + request.headers.get("host", "")).hostname or "") not in _LOCAL_NAMES:
        return False
    origin = request.headers.get("origin")
    return not origin or (urlsplit(origin).hostname or "") in _LOCAL_NAMES


def _require_local(request: Request) -> None:
    if not _is_local_request(request):
        raise HTTPException(
            status_code=403,
            detail="Settings and provisioning can only be changed on the Mac that runs the viewer (open it at localhost).",
        )


def _settings_projects() -> dict[str, Path]:
    """Project name -> root: this viewer, then each project of the checkout tests run in."""
    out = {VIEWER_PROJECT: BASE_DIR}
    try:
        repo = _test_repo()
    except HTTPException:
        return out
    for name in SOURCE_PROJECTS:
        if (repo / name / "pyproject.toml").is_file():
            out[name] = repo / name
    return out


def _settings_path(project: str, file: str) -> tuple[Path, str]:
    """The file (project, file) names and its kind: env, project (talkbench.yaml), agent
    (samples/*.yaml) or provision (samples/provision/*.yaml). No other file can be named."""
    root = _settings_projects().get(project)
    if root is None:
        raise HTTPException(status_code=404, detail=f"no project {project!r}")
    parts = file.split("/")
    if file == ".env":
        return root / file, "env"
    if project != VIEWER_PROJECT:
        if file == "talkbench.yaml":
            return root / file, "project"
        if len(parts) == 2 and parts[0] == "samples" and _YAML_FILE_RE.match(parts[1]):
            return root / file, "agent"
        if (project == TALKDESK_PROJECT and len(parts) == 3 and parts[:2] == ["samples", "provision"]
                and _YAML_FILE_RE.match(parts[2])):
            return root / file, "provision"
    raise HTTPException(status_code=404, detail=f"{file} is not a config file the viewer edits")


def _file_version(data: bytes | None) -> str:
    return hashlib.sha256(data).hexdigest()[:16] if data is not None else "missing"


def _read_config(path: Path) -> tuple[str, str]:
    """(text, version) of a config file; a missing file reads as empty."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return "", _file_version(None)
    return data.decode("utf-8", errors="replace"), _file_version(data)


def _write_config(project: str, file: str, path: Path, text: str, version: str | None) -> None:
    """Replace the file atomically when it still is the version the page loaded, keeping its
    permissions (a new .env is 0600) and the old text in the settings history."""
    old, current = _read_config(path)
    if version is not None and version != current:
        raise HTTPException(status_code=409, detail=f"{file} changed on disk since it was loaded — reload it first")
    target = path.resolve() if path.exists() else path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.viewer-tmp")
    tmp.write_text(text)
    if target.exists():
        shutil.copymode(target, tmp)
    else:
        os.chmod(tmp, 0o600 if target.name == ".env" else 0o644)
    if current != "missing":
        hist = SETTINGS_HISTORY_DIR / project
        hist.mkdir(parents=True, exist_ok=True)
        stem = file.replace("/", "__")
        snap = hist / f"{stem}.{time.strftime('%Y%m%dT%H%M%S')}"
        snap.write_text(old)
        os.chmod(snap, 0o600)
        for stale in sorted(hist.glob(f"{stem}.*"))[:-_SETTINGS_KEEP]:
            stale.unlink(missing_ok=True)
    os.replace(tmp, target)


# -- .env files --

def _env_value(rest: str) -> tuple[str, int, int]:
    """The value of the text after '=' the way python-dotenv reads it, and the (start, end)
    of its source in rest. What follows end (an inline # comment) survives a rewrite."""
    start = len(rest) - len(rest.lstrip())
    s = rest[start:]
    if s[:1] in ("'", '"'):
        quote, i, out = s[0], 1, []
        while i < len(s) and s[i] != quote:
            if quote == '"' and s[i] == "\\" and i + 1 < len(s):
                out.append({"n": "\n", "t": "\t", "r": "\r"}.get(s[i + 1], s[i + 1]))
                i += 2
                continue
            out.append(s[i])
            i += 1
        return "".join(out), start, start + min(i + 1, len(s))
    m = re.search(r"(?:^|\s)#", s)
    value = (s[:m.start()] if m else s).rstrip()
    return value, start, start + len(value)


def _env_quote(value: str) -> str:
    if re.fullmatch(r"[\w@%+=:,./~\-{}?&!*\[\]]*", value):
        return value
    if "'" not in value:
        return f"'{value}'"  # single quotes: no escapes, no ${VAR} expansion
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _mask(value: str) -> str:
    return "" if not value else "…" + value[-4:] if len(value) >= 12 else "•" * 6


def _env_items(text: str) -> list[dict[str, Any]]:
    """The page's view of a .env, in file order: headings ("# -- Title ---"), notes (a comment
    block on its own) and entries (KEY=value, or "# KEY=value" commented out) carrying the comment
    lines right above them as help. Secret values are masked."""
    items: list[dict[str, Any]] = []
    block: list[str] = []

    def flush() -> None:
        if block:
            items.append({"type": "note", "text": "\n".join(block)})
            block.clear()

    for n, line in enumerate(text.split("\n")):
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        m = _ENV_LINE_RE.match(line)
        if m:
            value, _, end = _env_value(m["rest"])
            secret = bool(_SECRET_RE.search(m["key"]))
            items.append({
                "type": "entry", "line": n, "key": m["key"], "active": not m["hash"], "secret": secret,
                "value": None if secret else value, "hint": _mask(value) if secret else None, "set": bool(value),
                "help": "\n".join(block), "inline": m["rest"][end:].strip().lstrip("#").strip(),
            })
            block.clear()
        elif (h := _ENV_HEADING_RE.match(stripped)):
            flush()
            if h["title"]:
                items.append({"type": "heading", "text": h["title"]})
        elif stripped.startswith("#"):
            block.append(re.sub(r"^#\s?", "", stripped))
        else:
            flush()
            items.append({"type": "note", "text": line, "raw": True})  # a line python-dotenv skips
    flush()
    return items


def _env_example_keys(root: Path) -> list[dict[str, str]]:
    """Variables the project's .env.example documents, for the page's "add a variable" list."""
    text, _ = _read_config(root / ".env.example")
    seen: dict[str, str] = {}
    for item in _env_items(text):
        if item["type"] == "entry" and item["key"] not in seen:
            seen[item["key"]] = item["help"] or item["inline"]
    return [{"key": k, "help": h} for k, h in seen.items()]


class EnvEdit(BaseModel):
    line: int
    key: str
    value: str | None = None  # None: keep the value
    active: bool | None = None  # False: comment the line out
    delete: bool = False


class EnvAdd(BaseModel):
    key: str
    value: str = ""
    active: bool = True


def _check_env_value(key: str, value: str) -> None:
    if not _ENV_KEY_RE.match(key):
        raise HTTPException(status_code=400, detail=f"{key!r} is not a valid variable name")
    if "\n" in value or "\r" in value or "\0" in value:
        raise HTTPException(status_code=400, detail=f"{key}: a value must be a single line")


def _env_apply(text: str, edits: list[EnvEdit], adds: list[EnvAdd]) -> str:
    lines = text.split("\n")
    drop: set[int] = set()
    for e in edits:
        m = _ENV_LINE_RE.match(lines[e.line]) if 0 <= e.line < len(lines) else None
        if not m or m["key"] != e.key:
            raise HTTPException(status_code=409, detail=f"line {e.line + 1} is no longer {e.key} — reload the file")
        if e.delete:
            drop.add(e.line)
            continue
        rest = m["rest"]
        if e.value is not None:
            _check_env_value(e.key, e.value)
            _, start, end = _env_value(rest)
            tail = rest[end:]
            rest = rest[:start] + _env_quote(e.value) + (" " + tail if tail.startswith("#") else tail)
        active = not m["hash"] if e.active is None else e.active
        lines[e.line] = f"{m['lead']}{'' if active else '# '}{m['export'] or ''}{e.key}{m['sep']}{rest}"
    kept = [line for n, line in enumerate(lines) if n not in drop]
    out = "\n".join(kept)
    if adds:
        set_keys = {m["key"]: n for n, line in enumerate(kept) if (m := _ENV_LINE_RE.match(line)) and not m["hash"]}
        new = []
        for a in adds:
            _check_env_value(a.key, a.value)
            if a.active and a.key in set_keys:
                raise HTTPException(status_code=400, detail=f"{a.key} is already set on line {set_keys[a.key] + 1}")
            new.append(f"{'' if a.active else '# '}{a.key}={_env_quote(a.value)}")
        out = (out.rstrip("\n") + "\n" if out.strip() else "") + "\n".join(new) + "\n"
    return out


def _apply_viewer_env(old: dict[str, str | None], path: Path) -> list[str]:
    """Bring this process in line with the viewer .env just saved, the way load_dotenv would
    have at start-up (a variable the shell set wins). Returns the changed keys that only a
    restart applies."""
    global LLM_MODEL
    new = dotenv_values(path)
    restart = []
    for key in set(old) | set(new):
        if key in _BASE_ENV or old.get(key) == new.get(key):
            continue
        if new.get(key) is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = new[key]
        if key in _VIEWER_RESTART_KEYS:
            restart.append(key)
    LLM_MODEL = os.environ.get("VIEWER_LLM_MODEL", "azure/gpt-4.1")
    return restart


# -- YAML files: talkbench.yaml through talk-bench's own ProjectConfig, agent and provision YAMLs --

@cache
def _tb_project_config() -> Any | None:
    _tb_matching()  # puts the checkout's talk_bench_core on sys.path
    try:
        from talk_bench_core.project_config import ProjectConfig
    except Exception:
        return None
    return ProjectConfig


def _project_yaml_check(text: str) -> list[str]:
    """Validate talkbench.yaml with talk-bench's ProjectConfig (unknown fields inside a block are
    errors, unknown top-level keys are dropped with a warning). Returns the warnings."""
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"not valid YAML: {e}")
    if doc is None:
        doc = {}
    if not isinstance(doc, dict):
        raise HTTPException(status_code=400, detail="talkbench.yaml must be a mapping")
    pc = _tb_project_config()
    if pc is None:
        return ["talk-bench's ProjectConfig could not be loaded, so this was only checked as YAML"]
    known = {f.alias or name for name, f in pc.model_fields.items()}
    warnings = [f"unknown top-level key {k!r}: talk-bench ignores it" for k in sorted(set(doc) - known, key=str)]
    try:
        pc.model_validate(doc)
    except Exception as e:
        errors = e.errors() if hasattr(e, "errors") else []
        lines = [f"{'.'.join(str(p) for p in err['loc']) or '(file)'}: {err['msg']}" for err in errors] or [str(e)]
        raise HTTPException(status_code=400, detail="talkbench.yaml is not valid:\n" + "\n".join(lines))
    return warnings


def _provision_fields(root: Path) -> list[str]:
    """ProvisionConfig's fields, read from talkdesk_agent's provisioner.py without importing it."""
    src = root / "libs" / "talkdesk_agent" / "src" / "talkdesk_agent" / "provisioner.py"
    try:
        tree = ast.parse(src.read_text())
    except (OSError, SyntaxError):
        return []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ProvisionConfig":
            return [s.target.id for s in node.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
    return []


def _provision_yaml(text: str) -> dict[str, Any]:
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"not valid YAML: {e}")
    if not isinstance(doc, dict):
        raise HTTPException(status_code=400, detail="a provisioning config must be a YAML mapping")
    missing = [k for k in ("agent_name", "api_base_url") if not doc.get(k)]
    if missing:
        raise HTTPException(status_code=400, detail=f"missing required field(s): {', '.join(missing)}")
    return doc


def _yaml_text_check(kind: str, text: str) -> list[str]:
    if len(text) > 400_000:
        raise HTTPException(status_code=413, detail="file is too large")
    if kind == "project":
        return _project_yaml_check(text)
    if kind == "agent":
        _parse_config_text(text)
        return []
    doc = _provision_yaml(text)
    fields = _provision_fields(_settings_projects().get(TALKDESK_PROJECT, BASE_DIR))
    return [f"unknown field {k!r}: talkdesk-agent ignores it" for k in doc if fields and k not in fields]


def _yaml_rt():
    from ruamel.yaml import YAML

    y = YAML()  # round-trip mode: comments, key order and quoting survive
    y.preserve_quotes = True
    y.width = 4096
    y.indent(mapping=2, sequence=2, offset=0)
    return y


def _rt_load(text: str):
    from ruamel.yaml.comments import CommentedMap

    try:
        doc = _yaml_rt().load(text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"not valid YAML: {e}")
    if doc is None:
        doc = CommentedMap()
    if not isinstance(doc, CommentedMap):
        raise HTTPException(status_code=400, detail="the file must be a YAML mapping")
    return doc


def _rt_dump(doc, original: str) -> str:
    buf = io.StringIO()
    _yaml_rt().dump(doc, buf)
    out = buf.getvalue()
    if not doc:
        out = ""
    trail = original[len(original.rstrip("\n")):] or "\n"  # ruamel drops trailing blank lines
    return out.rstrip("\n") + trail


def _post_comment(m, key):
    item = m.ca.items.get(key)
    return item[2] if item and len(item) > 2 else None


def _last_leaf(m, key):
    """For a mapping value the comment lines below an entry sit on its deepest last key."""
    value = m.get(key)
    while hasattr(value, "ca") and isinstance(value, dict) and len(value):
        m, key = value, next(reversed(value))
        value = m[key]
    return m, key


def _append_post(m, key, lines: str) -> None:
    from ruamel.yaml.error import CommentMark
    from ruamel.yaml.tokens import CommentToken

    tok = _post_comment(m, key)
    if tok is None:
        m.ca.items.setdefault(key, [None, None, None, None])[2] = CommentToken("\n" + lines, CommentMark(0), None)
    else:
        tok.value = tok.value + lines


def _yaml_put(m, key, value) -> None:
    """Add a new key at the end of mapping m: the comment lines that followed the old last
    entry (usually the next block's header) move below the new one."""
    last = next(reversed(m), None) if len(m) else None
    m[key] = value
    if last is None:
        return
    lm, lk = _last_leaf(m, last)
    tok = _post_comment(lm, lk)
    if tok is None:
        return
    head, _, tail = tok.value.partition("\n")
    if tail.strip():
        tok.value = head + "\n"
        _append_post(*_last_leaf(m, key), tail)


def _yaml_pop(m, key, carry: str = "") -> str:
    """Delete m[key], keeping the comment lines below it (its own end-of-line comment goes with
    it): they move below the entry before, or above the new first one. Returns them when m is
    left empty, for the caller to place."""
    keys = list(m)
    i = keys.index(key)
    lm, lk = _last_leaf(m, key)
    tok = _post_comment(lm, lk)
    tail = (tok.value.partition("\n")[2] if tok else "") + carry
    m.ca.items.pop(key, None)
    del m[key]
    if not tail.strip():
        return ""
    if i > 0:
        _append_post(*_last_leaf(m, keys[i - 1]), tail)
        return ""
    if len(m):
        lines = [re.sub(r"^#\s?", "", ln.strip()) for ln in tail.strip("\n").split("\n")]
        m.yaml_set_comment_before_after_key(next(iter(m)), before="\n".join(lines), indent=getattr(m.lc, "col", 0) or 0)
        return ""
    return tail


def _yaml_set_path(doc, path: list[str], value: Any) -> None:
    from ruamel.yaml.comments import CommentedMap
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString, SingleQuotedScalarString

    node = doc
    for i, k in enumerate(path[:-1]):
        child = node.get(k)
        if not isinstance(child, CommentedMap):
            nested = value
            for kk in reversed(path[i + 1:]):
                nested = CommentedMap([(kk, nested)])
            if k in node:
                node[k] = nested  # a scalar or null block, e.g. `evaluation:` with nothing under it
            else:
                _yaml_put(node, k, nested)
            return
        node = child
    key = path[-1]
    if key not in node:
        _yaml_put(node, key, value)
        return
    old = node[key]
    if isinstance(value, str) and isinstance(old, (DoubleQuotedScalarString, SingleQuotedScalarString)):
        value = type(old)(value)  # keep the quoting the file used
    node[key] = value


def _yaml_unset_path(doc, path: list[str]) -> None:
    chain, node = [doc], doc
    for k in path[:-1]:
        node = node.get(k) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            return
        chain.append(node)
    if path[-1] not in node:
        return
    carry = _yaml_pop(node, path[-1])
    # a block the unset left empty goes too ("user_persona: {}" would read as a choice)
    for i in range(len(chain) - 1, 0, -1):
        if len(chain[i]):
            if carry:
                _append_post(*_last_leaf(chain[i], next(reversed(chain[i]))), carry)
            return
        carry = _yaml_pop(chain[i - 1], path[i - 1], carry)


def _scalar_end(line: str, col: int) -> int | None:
    """Where the scalar token starting at col ends on its line (None: it is not a one-line scalar)."""
    q = line[col:col + 1]
    if q in ("|", ">", "[", "{", "&", "*", "!", ""):
        return None
    if q in ("'", '"'):
        i = col + 1
        while i < len(line):
            if q == '"' and line[i] == "\\":
                i += 2
                continue
            if line[i] == q:
                if q == "'" and line[i + 1:i + 2] == "'":
                    i += 2
                    continue
                return i + 1
            i += 1
        return None
    m = re.compile(r"\s#").search(line, col)
    return len((line[:m.start()] if m else line).rstrip())


def _yaml_scalar(value: Any, quote: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    value = str(value)
    if quote == "'":
        return "'" + value.replace("'", "''") + "'"
    plain = quote != '"' and re.fullmatch(r"[\w./@~+-][\w./@~+:,=-]*", value) and yaml.safe_load(value) == value
    return value if plain else json.dumps(value, ensure_ascii=False)


def _yaml_edit(text: str, changes: list[tuple[list[str], Any]]) -> str:
    """Apply (path, value) changes (value None: remove the key) to a YAML text. A new value for an
    existing one-line scalar is written over the old one on its line, so the rest of the line (an
    aligned # comment) stays as it was; adding and removing keys goes through ruamel.yaml."""
    from ruamel.yaml.comments import CommentedMap

    doc, lines, rest = _rt_load(text), text.split("\n"), []
    for path, value in changes:
        node = doc
        for k in path[:-1]:
            node = node.get(k) if isinstance(node, CommentedMap) else None
        key, end = path[-1], None
        if (value is not None and not isinstance(value, (dict, list)) and isinstance(node, CommentedMap)
                and key in node and not isinstance(node[key], (dict, list))):
            row, col = node.lc.value(key)
            end = _scalar_end(lines[row], col)
        if end is None:
            rest.append((path, value))
            continue
        lines[row] = lines[row][:col] + _yaml_scalar(value, lines[row][col]) + lines[row][end:]
    text = "\n".join(lines)
    if not rest:
        return text
    doc = _rt_load(text)
    for path, value in rest:
        _yaml_unset_path(doc, path) if value is None else _yaml_set_path(doc, path, value)
    return _rt_dump(doc, text)


class YamlChange(BaseModel):
    path: list[str]
    value: Any = None  # None: remove the key (talk-bench's default applies)


# -- endpoints --

def _settings_file_payload(project: str, file: str) -> dict[str, Any]:
    path, kind = _settings_path(project, file)
    text, version = _read_config(path)
    out: dict[str, Any] = {
        "project": project, "file": file, "kind": kind, "path": str(path),
        "exists": version != "missing", "version": version,
    }
    if kind == "env":
        root = path.parent
        out.update(items=_env_items(text), example_keys=_env_example_keys(root))
        if project == VIEWER_PROJECT:
            out["applies"] = "Applies at once; VIEWER_DATA_DIR needs a viewer restart. A variable set in the shell that started the viewer wins over this file."
        else:
            out["applies"] = "Read by every test or provisioning started after the save."
        return out
    out["text"] = text
    try:
        out["warnings"] = _yaml_text_check(kind, text) if text.strip() else []
        out["error"] = None
    except HTTPException as e:
        out["warnings"], out["error"] = [], e.detail
    if kind == "project":
        pc = _tb_project_config()
        out["schema"] = pc.model_json_schema(by_alias=True) if pc else None
        try:
            doc = yaml.safe_load(text) or {}
        except yaml.YAMLError:
            doc = None
        out["doc"] = json.loads(json.dumps(doc, default=str)) if isinstance(doc, dict) else None
        out["suggest"] = _project_suggestions()
    return out


def _project_suggestions() -> dict[str, list[str]]:
    """Values written for each key in the projects' talkbench.yaml files, commented-out
    alternatives included — the form's suggestions for its text fields."""
    out: dict[str, set[str]] = {}
    for name, root in _settings_projects().items():
        if name == VIEWER_PROJECT:
            continue
        text, _ = _read_config(root / "talkbench.yaml")
        for m in re.finditer(r"^\s*#?\s*([a-z_]+):[ \t]+([^\s#'\"]\S*|'[^'\n]*'|\"[^\"\n]*\")[ \t]*(?:#.*)?$", text, re.M):
            out.setdefault(m[1], set()).add(m[2].strip("'\""))
    return {k: sorted(v) for k, v in out.items()}


@app.get("/api/settings")
def api_settings(request: Request):
    projects = []
    for name, root in _settings_projects().items():
        files = [{"file": ".env", "kind": "env", "exists": (root / ".env").is_file()}]
        if name != VIEWER_PROJECT:
            files.insert(0, {"file": "talkbench.yaml", "kind": "project", "exists": (root / "talkbench.yaml").is_file()})
            for s in _sample_configs(root):
                files.append({"file": f"samples/{s['file']}", "kind": "agent", "exists": True,
                              "name": s.get("name"), "module": s.get("module")})
        projects.append({"name": name, "root": str(root), "files": files})
    return {"local": _is_local_request(request), "projects": projects}


@app.get("/api/settings/file")
def api_settings_file(project: str = Query(...), file: str = Query(...)):
    return _settings_file_payload(project, file)


class SettingsSaveBody(BaseModel):
    project: str
    file: str
    version: str | None = None  # None: create a file that must not exist yet
    text: str | None = None  # YAML files: the whole new text
    changes: list[YamlChange] | None = None  # talkbench.yaml: form edits applied to the file as it is
    env_edits: list[EnvEdit] = []
    env_adds: list[EnvAdd] = []


@app.post("/api/settings/file")
def api_settings_save(body: SettingsSaveBody, request: Request):
    _require_local(request)
    path, kind = _settings_path(body.project, body.file)
    with _settings_lock:
        text, current = _read_config(path)
        if body.version is None and current != "missing":
            raise HTTPException(status_code=409, detail=f"{body.file} already exists")
        if kind == "env":
            old = dotenv_values(path) if path.is_file() else {}
            new_text = _env_apply(text, body.env_edits, body.env_adds)
        elif body.changes is not None and kind == "project":
            if not all(ch.path and all(p for p in ch.path) for ch in body.changes):
                raise HTTPException(status_code=400, detail="invalid field path")
            new_text = _yaml_edit(text, [(ch.path, ch.value) for ch in body.changes])
        elif body.text is not None:
            new_text = body.text
        else:
            raise HTTPException(status_code=400, detail="nothing to save")
        warnings = [] if kind == "env" else _yaml_text_check(kind, new_text)
        _write_config(body.project, body.file, path, new_text, body.version)
        restart = _apply_viewer_env(old, path) if kind == "env" and body.project == VIEWER_PROJECT else []
    return {**_settings_file_payload(body.project, body.file), "saved_warnings": warnings, "restart": restart}


# ---- Talkdesk provisioning and the MCP server, run in talk-bench-talkdesk ----
#
# The steps of talk-bench-talkdesk's README, Step 3: an MCP server for the domain, reachable from
# Talkdesk (`talkdesk-mcp-server --source S --domain D --ngrok`), then
# `talkdesk-agent provision --config samples/provision/<x>.yaml`, which creates the MCP server record
# and the agent system and prints its agent_system_id; that id then goes into the agent YAML a test
# runs (agent.params.agent_system_id). Like `talkdesk-agent generate`, the preview patches the template
# without any API call, but into a copy (see _TD_PREVIEW); provision, like the CLI, writes the patched
# template back to libs/talkdesk_agent/templates. Both load talk-bench-talkdesk/.env (TALKDESK_ACCOUNT_ID /
# OAUTH_PATH / TOKEN_URL / MCP_URL, NGROK_AUTHTOKEN). Processes started here get their own process
# group and survive a viewer restart; their logs live under <data dir>/.talkdesk_jobs/<id>/.

TD_JOBS_DIR = DATA_DIR / ".talkdesk_jobs"
_td_jobs: dict[str, dict[str, Any]] = {}
_td_lock = threading.Lock()
_td_loaded = False
_TD_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_TD_RESULT_RES = {
    "agent_system_id": re.compile(r"Agent System ID:\s*(\S+)"),
    "mcp_server_id": re.compile(r"MCP server created\. ID:\s*(\S+)"),
    "agent_name": re.compile(r"Agent Name:\s*(.+?)\s*$", re.M),
    "url": re.compile(r"ngrok URL:\s*(\S+)"),
    "local_url": re.compile(r"Starting MCP Streamable HTTP server on (\S+)"),
}


def _td_root() -> Path:
    root = _settings_projects().get(TALKDESK_PROJECT)
    if root is None:
        raise HTTPException(status_code=404, detail="no talk-bench-talkdesk project in the checkout")
    return root


def _td_cli(root: Path, script: str) -> list[str]:
    uv = _uv_bin()
    if uv:
        return [uv, "run", script]
    shim = root / ".venv" / "bin" / script
    if shim.is_file():
        return [str(shim)]
    raise HTTPException(status_code=500, detail=f"neither uv nor {shim} was found")


def _td_env_values(root: Path) -> dict[str, str]:
    return {k: v for k, v in dotenv_values(root / ".env").items() if v is not None} if (root / ".env").is_file() else {}


def _td_mcp_domains(root: Path) -> dict[str, list[str]]:
    """The --domain choices of talkdesk-mcp-server per --source, read from its backends.py."""
    src = root / "apps" / "talkdesk_mcp_server" / "src" / "talkdesk_mcp_server" / "backends.py"
    out = {"talk_bench": [], "tau2": []}
    try:
        tree = ast.parse(src.read_text())
    except (OSError, SyntaxError):
        return out
    names = {"_TALK_BENCH_DOMAINS": "talk_bench", "_TAU2_DOMAINS": "tau2"}
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0] if isinstance(node, ast.Assign) else None
        if isinstance(target, ast.Name) and target.id in names and isinstance(node.value, ast.Call) and node.value.args:
            try:
                out[names[target.id]] = sorted(ast.literal_eval(node.value.args[0]))
            except ValueError:
                pass
    return out


def _td_load() -> None:
    global _td_loaded
    if _td_loaded:
        return
    _td_loaded = True
    if TD_JOBS_DIR.is_dir():
        for d in TD_JOBS_DIR.iterdir():
            job = _load_json(d / "job.json")
            if isinstance(job, dict) and job.get("id") == d.name:
                _td_jobs[d.name] = job


def _td_save(job: dict[str, Any]) -> None:
    d = TD_JOBS_DIR / job["id"]
    d.mkdir(parents=True, exist_ok=True)
    _write_json(d / "job.json", {k: v for k, v in job.items() if not k.startswith("_")})


def _td_log(job_id: str) -> Path:
    return TD_JOBS_DIR / job_id / "log.txt"


def _td_scan(job: dict[str, Any]) -> None:
    """Pick what the log says so far: ids printed by provision, the URLs an MCP server serves on
    (printed at its start: its log grows with every tool call, so only the head is read)."""
    try:
        with _td_log(job["id"]).open("rb") as f:
            text = f.read(65536 if job["kind"] == "mcp" else -1).decode("utf-8", errors="replace")
    except OSError:
        return
    for key, rx in _TD_RESULT_RES.items():
        if (m := rx.search(text)):
            job.setdefault("result", {})[key] = m[1]
    if job["kind"] == "provision" and job["status"] in ("error", "interrupted"):
        errs = re.findall(r"^Error: (.+)$", text, re.M)
        job["error"] = errs[-1][:500] if errs else job.get("error")


def _td_settle(job: dict[str, Any], code: int | None) -> None:
    job["finished"] = time.time()
    job["exit_code"] = code
    if job["kind"] == "mcp":
        job["status"] = "stopped"
    elif code == 0:
        job["status"] = "done"
    else:
        job["status"] = "error" if code is not None else "interrupted"
    _td_scan(job)
    if job["kind"] == "provision" and job["status"] != "done" and not job.get("error"):
        job["error"] = _last_log_lines(_td_log(job["id"]))
    _td_save(job)


def _td_watch(job_id: str, proc: subprocess.Popen) -> None:
    code = proc.wait()
    with _td_lock:
        _td_settle(_td_jobs[job_id], code)


def _td_refresh() -> None:
    """Settle jobs whose process ended while no watcher was waiting (a viewer restart)."""
    with _td_lock:
        _td_load()
        for job in _td_jobs.values():
            if job["status"] == "running":
                if not job.get("_watched") and not _pid_alive(job.get("pid")):
                    _td_settle(job, None)
                else:
                    _td_scan(job)


def _td_start(kind: str, cmd: list[str], root: Path, config_text: str | None = None, **fields: Any) -> dict[str, Any]:
    job = {
        "id": uuid.uuid4().hex[:12], "kind": kind, "status": "running", "started": time.time(),
        "finished": None, "exit_code": None, "pid": None, "cmd": cmd, "cwd": str(root), "result": {},
        "error": None, **fields,
    }
    d = TD_JOBS_DIR / job["id"]
    d.mkdir(parents=True, exist_ok=True)
    if config_text is not None:
        (d / "config.yaml").write_text(config_text)
        cmd[cmd.index("@config")] = str(d / "config.yaml")
    with _td_log(job["id"]).open("ab") as log:
        log.write(f"$ cd {shlex.quote(str(root))} && {shlex.join(cmd)}\n\n".encode())
        log.flush()
        try:
            proc = subprocess.Popen(
                cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                env=_child_env(), start_new_session=True,
            )
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"could not start {cmd[0]}: {e}")
    job["pid"] = proc.pid
    with _td_lock:
        _td_jobs[job["id"]] = job
        _td_save(job)
        job["_watched"] = True  # not saved: after a restart the pid is polled instead
    threading.Thread(target=_td_watch, args=(job["id"], proc), daemon=True).start()
    return job


def _td_public(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _etime_start(etime: str) -> float | None:
    """When a process started, from ps's elapsed time ([[dd-]hh:]mm:ss)."""
    days, _, clock = etime.rpartition("-")
    try:
        parts = [int(x) for x in clock.split(":")]
        secs = sum(v * 60 ** i for i, v in enumerate(reversed(parts))) + int(days or 0) * 86400
    except ValueError:
        return None
    return time.time() - secs


def _argv_opt(argv: list[str], name: str, default: str) -> str:
    return argv[argv.index(name) + 1] if name in argv[:-1] else default  # talkdesk-mcp-server's own defaults


def _mcp_processes() -> list[dict[str, Any]]:
    """talkdesk-mcp-server processes on this Mac, one per process group (uv run + its python),
    with the options they were started with."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,pgid=,etime=,command="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    groups: dict[int, dict[str, Any]] = {}
    members: dict[int, list[str]] = {}
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, pgid, etime, command = int(parts[0]), int(parts[1]), parts[2], parts[3]
        members.setdefault(pgid, []).append(command)
        if "talkdesk-mcp-server" not in command or pid == os.getpid():
            continue
        argv = command.split()
        port = _argv_opt(argv, "--port", "3030")
        g = groups.setdefault(pgid, {"pgid": pgid, "pids": [], "started": _etime_start(etime)})
        g["pids"].append(pid)
        g.update(source=_argv_opt(argv, "--source", "talk_bench"), domain=_argv_opt(argv, "--domain", "airline"),
                 port=int(port) if port.isdigit() else 3030, ngrok="--ngrok" in argv)
    for pgid, g in groups.items():  # a group holding anything else (a shell) is stopped pid by pid
        g["own_group"] = all("talkdesk-mcp-server" in c for c in members.get(pgid, []))
    return list(groups.values())


def _mcp_probe(url: str, timeout: float) -> dict[str, Any]:
    """GET <url>/ of an MCP server: it answers with its name, talkdesk-mcp-<source>-<domain>."""
    req = urllib.request.Request(url.rstrip("/") + "/", headers={"Accept": "application/json", "ngrok-skip-browser-warning": "1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            info = json.loads(resp.read(65536) or b"{}")
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"ok": False, "error": str(getattr(e, "reason", e))[:200]}
    name = str(info.get("name") or "") if isinstance(info, dict) else ""
    m = re.match(r"^talkdesk-mcp-(talk_bench|tau2)-(.+)$", name)
    return {"ok": True, "name": name, "source": m[1] if m else None, "domain": m[2] if m else None}


def _mcp_servers() -> list[dict[str, Any]]:
    _td_refresh()
    by_pid = {j["pid"]: j for j in _td_jobs.values() if j["kind"] == "mcp" and j["status"] == "running"}
    servers = []
    for g in _mcp_processes():
        job = by_pid.get(g["pgid"])
        local = _mcp_probe(f"http://127.0.0.1:{g['port']}", 0.8)
        servers.append({
            **g, "job_id": job["id"] if job else None, "url": (job or {}).get("result", {}).get("url"),
            "serving": local.get("ok", False), "name": local.get("name"),
        })
    return servers


def _td_resolve(root: Path, doc: dict[str, Any]) -> list[dict[str, Any]]:
    """What talkdesk-agent will use for each setting: the YAML's value, else the .env fallback."""
    env = _td_env_values(root)
    rows = []
    for key, env_key in (("account_id", "TALKDESK_ACCOUNT_ID"), ("oauth_path", "TALKDESK_OAUTH_PATH"),
                         ("token_url", "TALKDESK_TOKEN_URL"), ("mcp_url", "TALKDESK_MCP_URL")):
        value, source = (doc.get(key), "yaml") if doc.get(key) else (env.get(env_key) or None, f".env {env_key}")
        row = {"key": key, "value": value, "source": source if value else None, "ok": bool(value) or key == "token_url"}
        if key == "oauth_path" and value:
            row["ok"] = (root / Path(str(value)).expanduser()).is_file()
            row["note"] = None if row["ok"] else "file not found next to talk-bench-talkdesk"
        if not value:
            row["note"] = "not set (YAML or .env)" if key != "token_url" else "not set: talkdesk_agent's default"
        rows.append(row)
    template = str(doc.get("template_name") or "airline-agent-template.json")
    found = (root / "libs" / "talkdesk_agent" / "templates" / template).is_file()
    rows.append({"key": "template_name", "value": template, "source": "yaml" if doc.get("template_name") else "default",
                 "ok": found, "note": None if found else "no such template in libs/talkdesk_agent/templates"})
    return rows


def _td_platform(url: str | None) -> str | None:
    host = urlsplit(str(url or "")).hostname or ""
    for tag, label in (("talkdeskqa", "QA"), ("talkdeskstg", "STG"), ("talkdeskapp", "PRD")):
        if tag in host:
            return label
    return None


@app.get("/api/talkdesk")
def api_talkdesk(request: Request):
    root = _td_root()
    _td_refresh()
    prov = root / "samples" / "provision"
    configs = []
    for f in sorted(prov.glob("*.y*ml")) if prov.is_dir() else []:
        try:
            doc = yaml.safe_load(f.read_text())
        except yaml.YAMLError:
            doc = None
        doc = doc if isinstance(doc, dict) else {}
        configs.append({"file": f"samples/provision/{f.name}", "agent_name": doc.get("agent_name"),
                        "platform": _td_platform(doc.get("api_base_url")), "template_name": doc.get("template_name")})
    templates = sorted(p.name for p in (root / "libs" / "talkdesk_agent" / "templates").glob("*.json"))
    agents = [{"file": f"samples/{s['file']}", "name": s.get("name"), "module": s.get("module")}
              for s in _sample_configs(root) if str(s.get("module") or "").startswith("talkdesk_agent.")]
    env = _td_env_values(root)
    jobs = sorted((_td_public(j) for j in _td_jobs.values()), key=lambda j: j["started"], reverse=True)
    return {
        "local": _is_local_request(request), "root": str(root), "configs": configs, "templates": templates,
        "agent_samples": agents, "domains": _td_mcp_domains(root), "mcp_url": env.get("TALKDESK_MCP_URL"),
        "has_ngrok_token": bool(env.get("NGROK_AUTHTOKEN")), "servers": _mcp_servers(), "jobs": jobs,
    }


@app.post("/api/talkdesk/check")
def api_talkdesk_check(body: ConfigTextBody):
    root = _td_root()
    doc = _provision_yaml(body.text)
    fields = _provision_fields(root)
    return {
        "agent_name": doc.get("agent_name"), "api_base_url": doc.get("api_base_url"),
        "platform": _td_platform(doc.get("api_base_url")), "resolved": _td_resolve(root, doc),
        "warnings": [f"unknown field {k!r}: talkdesk-agent ignores it" for k in doc if fields and k not in fields],
    }


# What `talkdesk-agent generate` prints, patched into a copy of the template: talkdesk_agent's
# _patch_template writes its result back to the template it read, and the bundled one is in git.
_TD_PREVIEW = """\
import json, os, shutil, sys, tempfile
from pathlib import Path
from dotenv import load_dotenv
from talkdesk_agent.provisioner import _TEMPLATES_DIR, ProvisionConfig, generate_template
load_dotenv(".env")
config = ProvisionConfig.from_yaml_and_env(yaml_path=Path(sys.argv[1]), env=dict(os.environ))
with tempfile.TemporaryDirectory() as tmp:
    copy = Path(tmp) / config.template_name
    shutil.copyfile(_TEMPLATES_DIR / config.template_name, copy)
    print(json.dumps(generate_template(config, template_path=copy), indent=2))
"""


@app.post("/api/talkdesk/generate")
def api_talkdesk_generate(body: ConfigTextBody, request: Request):
    """The template provision would deploy, with no API call and no file changed."""
    _require_local(request)
    root = _td_root()
    _provision_yaml(body.text)
    uv, python = _uv_bin(), root / ".venv" / "bin" / "python"
    launcher = [uv, "run", "python"] if uv else [str(python)] if python.is_file() else None
    if launcher is None:
        raise HTTPException(status_code=500, detail=f"neither uv nor {python} was found")
    TD_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", dir=TD_JOBS_DIR, delete=False) as f:
        f.write(body.text)
    try:
        proc = subprocess.run([*launcher, "-c", _TD_PREVIEW, f.name], cwd=root,
                              capture_output=True, text=True, timeout=180, env=_child_env())
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="the template preview did not finish in 3 minutes")
    finally:
        Path(f.name).unlink(missing_ok=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-12:])
        raise HTTPException(status_code=400, detail=tail or f"the template preview exited with {proc.returncode}")
    try:
        template = json.loads(proc.stdout)
    except ValueError:
        raise HTTPException(status_code=500, detail="the template preview printed no JSON")
    agents = template.get("agents") if isinstance(template, dict) else None
    tools = sum(len(a.get("tools") or []) for a in agents or [] if isinstance(a, dict))
    return {"json": json.dumps(template, indent=2, ensure_ascii=False), "agents": len(agents or []), "tools": tools}


class ProvisionBody(BaseModel):
    text: str
    file: str | None = None  # the samples/provision file the text came from, for the history


@app.post("/api/talkdesk/provision")
def api_talkdesk_provision(body: ProvisionBody, request: Request):
    _require_local(request)
    root = _td_root()
    doc = _provision_yaml(body.text)
    _td_refresh()
    with _td_lock:
        busy = [j for j in _td_jobs.values() if j["kind"] == "provision" and j["status"] == "running"]
    if busy:
        raise HTTPException(status_code=409, detail="a provisioning run is still going — wait for it to finish")
    job = _td_start(
        "provision", [*_td_cli(root, "talkdesk-agent"), "provision", "--config", "@config"], root,
        config_text=body.text, file=body.file, agent_name=str(doc.get("agent_name")),
        api_base_url=str(doc.get("api_base_url")), platform=_td_platform(doc.get("api_base_url")),
        mcp_url=doc.get("mcp_url") or _td_env_values(root).get("TALKDESK_MCP_URL"),  # for the agent YAML afterwards
    )
    return {"job": _td_public(job)}


class McpStartBody(BaseModel):
    source: str = "talk_bench"
    domain: str
    port: int = 3030
    ngrok: bool = True


@app.post("/api/talkdesk/mcp")
def api_talkdesk_mcp_start(body: McpStartBody, request: Request):
    _require_local(request)
    root = _td_root()
    domains = _td_mcp_domains(root)
    if body.source not in domains:
        raise HTTPException(status_code=400, detail="source must be talk_bench or tau2")
    if domains[body.source] and body.domain not in domains[body.source]:
        raise HTTPException(status_code=400, detail=f"talkdesk-mcp-server serves no {body.source} domain {body.domain!r}")
    if not 1024 <= body.port <= 65535:
        raise HTTPException(status_code=400, detail="port must be between 1024 and 65535")
    running = _mcp_processes()
    if any(s["port"] == body.port for s in running):
        raise HTTPException(status_code=409, detail=f"an MCP server already listens on port {body.port} — stop it first")
    if body.ngrok and any(s["ngrok"] for s in running):
        raise HTTPException(status_code=409, detail="an MCP server with an ngrok tunnel is running — stop it first")
    cmd = [*_td_cli(root, "talkdesk-mcp-server"), "--source", body.source, "--domain", body.domain, "--port", str(body.port)]
    job = _td_start("mcp", cmd + (["--ngrok"] if body.ngrok else []), root,
                    source=body.source, domain=body.domain, port=body.port, ngrok=body.ngrok)
    return {"job": _td_public(job)}


class McpStopBody(BaseModel):
    pgid: int


@app.post("/api/talkdesk/mcp/stop")
def api_talkdesk_mcp_stop(body: McpStopBody, request: Request):
    _require_local(request)
    server = next((s for s in _mcp_processes() if s["pgid"] == body.pgid), None)
    if server is None:
        raise HTTPException(status_code=404, detail="that MCP server is not running")
    if body.pgid == os.getpgrp():
        raise HTTPException(status_code=400, detail="refusing to stop the viewer's own process group")

    def _signal(sig: int) -> None:
        if server["own_group"]:
            with contextlib.suppress(OSError):
                os.killpg(body.pgid, sig)
        for pid in server["pids"]:
            with contextlib.suppress(OSError):
                os.kill(pid, sig)

    _signal(signal.SIGTERM)

    def _force() -> None:
        time.sleep(5)
        if any(_pid_alive(p) for p in server["pids"]):
            _signal(signal.SIGKILL)

    threading.Thread(target=_force, daemon=True).start()
    return {"stopped": body.pgid}


class McpCheckBody(BaseModel):
    url: str


@app.post("/api/talkdesk/mcp/check")
def api_talkdesk_mcp_check(body: McpCheckBody, request: Request):
    _require_local(request)  # the server fetches the URL: not for anyone on the tunnel
    if urlsplit(body.url).scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="the MCP URL must start with http:// or https://")
    return _mcp_probe(body.url, 8)


@app.get("/api/talkdesk/jobs/{job_id}")
def api_talkdesk_job(job_id: str, offset: int = 0):
    if not _TD_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="no such job")
    _td_refresh()
    job = _td_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="no such job")
    log, next_offset = "", offset
    try:
        with _td_log(job_id).open("rb") as f:
            size = f.seek(0, 2)
            if offset > size:
                offset = 0
            f.seek(offset)
            chunk = f.read(_LOG_CHUNK)
            log, next_offset = chunk.decode("utf-8", errors="replace"), offset + len(chunk)
    except OSError:
        pass
    config = (TD_JOBS_DIR / job_id / "config.yaml")
    return {"job": _td_public(job), "log": log, "next_offset": next_offset,
            "config_text": config.read_text() if config.is_file() else None}


@app.delete("/api/talkdesk/jobs/{job_id}")
def api_talkdesk_job_delete(job_id: str, request: Request):
    _require_local(request)
    if not _TD_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="no such job")
    _td_refresh()
    with _td_lock:
        job = _td_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="no such job")
        if job["status"] == "running":
            raise HTTPException(status_code=409, detail="still running")
        _td_jobs.pop(job_id)
    shutil.rmtree(TD_JOBS_DIR / job_id, ignore_errors=True)
    return {"deleted": job_id}


class ApplyBody(BaseModel):
    file: str  # samples/<agent>.yaml of talk-bench-talkdesk
    agent_system_id: str
    mcp_url: str | None = None  # also set agent.mcp.server_url
    version: str | None = None


@app.post("/api/talkdesk/apply")
def api_talkdesk_apply(body: ApplyBody, request: Request):
    """Put a provisioned agent_system_id (and the MCP URL it was provisioned with) into an agent YAML."""
    _require_local(request)
    path, kind = _settings_path(TALKDESK_PROJECT, body.file)
    if kind != "agent":
        raise HTTPException(status_code=400, detail="pick an agent YAML from samples/")
    if not re.fullmatch(r"[\w\-]{3,100}", body.agent_system_id):
        raise HTTPException(status_code=400, detail="that does not look like an agent_system_id")
    with _settings_lock:
        text, version = _read_config(path)
        doc = yaml.safe_load(text)
        at = ["agent"] if isinstance(doc, dict) and isinstance(doc.get("agent"), dict) else []
        changes = [(at + ["params", "agent_system_id"], body.agent_system_id)]
        if body.mcp_url:
            changes.append((at + ["mcp", "server_url"], body.mcp_url))
        new_text = _yaml_edit(text, changes)
        _parse_config_text(new_text)
        _write_config(TALKDESK_PROJECT, body.file, path, new_text, body.version or version)
    return _settings_file_payload(TALKDESK_PROJECT, body.file)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


# Raw file access for audio playback and artifact inspection; supports Range requests.
app.mount("/data", PublicDataFiles(directory=DATA_DIR), name="data")
