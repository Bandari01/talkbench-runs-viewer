"""FastAPI backend for browsing archived talk-bench run data.

Data layout (created by the backup step):
    data/<source-repo>/<project>/<run-dir>/
        results.json                        (may be missing for errored runs)
        talk_bench_evaluation_result.json   (may be missing)
        <scenario-id>/
            scenario.json, conversation.json, run_meta.json, tool_log.json,
            conversation.wav, user.wav, agent.wav, *.ndjson, ...
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
# VIEWER_DATA_DIR points the archive at shared storage (e.g. a mounted Azure
# Files share) so several hosts can serve and upload to the same runs.
DATA_DIR = Path(os.environ.get("VIEWER_DATA_DIR") or BASE_DIR / "data").expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)  # fresh clone, or a share with no runs yet
STATIC_DIR = Path(__file__).resolve().parent / "static"

# LLM credentials and model come from this project's own .env (see .env.example).
# Tests launched from the Run test page get the pre-.env environment, so the
# checkout's own .env decides their credentials (see _child_env).
_BASE_ENV = dict(os.environ)
load_dotenv(BASE_DIR / ".env")
LLM_MODEL = os.environ.get("VIEWER_LLM_MODEL", "azure/gpt-4.1")
READ_ONLY = os.environ.get("VIEWER_READ_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
CHAT_ENABLED = os.environ.get("VIEWER_DISABLE_CHAT", "").strip().lower() not in {
    "1", "true", "yes", "on",
}
SYNC_ENABLED = not READ_ONLY or os.environ.get("VIEWER_ALLOW_SYNC", "").strip().lower() in {
    "1", "true", "yes", "on",
}

TS_RE = re.compile(r"(\d{8}T\d{6}Z)")

app = FastAPI(
    title="talk-bench runs viewer",
    docs_url=None if READ_ONLY else "/docs",
    redoc_url=None if READ_ONLY else "/redoc",
    openapi_url=None if READ_ONLY else "/openapi.json",
)



class ApiGZipMiddleware(GZipMiddleware):
    """Compress only /api/ JSON. Static/audio responses are left alone so
    Range requests (audio seeking) keep working and wav bytes aren't re-crunched."""

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").startswith("/api/"):
            await super().__call__(scope, receive, send)
        else:
            await self.app(scope, receive, send)


app.add_middleware(ApiGZipMiddleware, minimum_size=1024)

_index_cache: list[dict[str, Any]] | None = None
# agent / simulator config recorded in each archived run's results.json, keyed by run
# name — feeds the "from archived run" picker on the Run test page
_archived_run_context: dict[str, dict[str, Any]] = {}
# Run detail payloads are costly to build; cache a few, cleared on data changes.
_run_cache: dict[str, dict[str, Any]] = {}
_RUN_CACHE_MAX = 16


@app.middleware("http")
async def shared_view_guard(request: Request, call_next):
    # Reject public writes before FastAPI parses a potentially large upload body.
    chat_request = (
        request.method == "POST"
        and request.url.path in {"/api/chat", "/api/scenario_report", "/api/run_report"}
        and CHAT_ENABLED
    )
    sync_request = request.method == "POST" and request.url.path == "/api/sync_sources" and SYNC_ENABLED
    if (
        READ_ONLY
        and request.method not in {"GET", "HEAD", "OPTIONS"}
        and not chat_request
        and not sync_request
    ):
        return JSONResponse(status_code=403, content={"detail": "this shared viewer is read-only"})
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


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


def _safe_path(rel: str) -> Path:
    p = (DATA_DIR / rel).resolve()
    if not p.is_relative_to(DATA_DIR.resolve()):
        raise HTTPException(status_code=400, detail="path escapes data dir")
    return p


def _data_dir_at_depth(rel: str, depth: int, label: str) -> Path:
    parts = Path(rel).parts
    if len(parts) != depth or any(part.startswith(".") or part in {"..", ""} for part in parts):
        raise HTTPException(status_code=400, detail=f"invalid {label} path")
    p = _safe_path(rel)
    if not p.is_dir():
        raise HTTPException(status_code=404, detail=f"{label} not found")
    return p


def _ensure_writable() -> None:
    if READ_ONLY:
        raise HTTPException(status_code=403, detail="this shared viewer is read-only")


def _ensure_chat_enabled() -> None:
    if not CHAT_ENABLED:
        raise HTTPException(status_code=403, detail="AI chat is disabled on this shared viewer")


def _ensure_sync_enabled() -> None:
    if not SYNC_ENABLED:
        raise HTTPException(status_code=403, detail="source sync is disabled on this shared viewer")


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


def _run_modality(results: dict[str, Any] | None, scenarios: list[Path]) -> str | None:
    """"text" or "voice" (None when a run has no scenarios to judge by).

    The agent module is authoritative; runs from older harnesses carry no
    run_context, so fall back to whether the first scenarios recorded audio.
    """
    module = (((results or {}).get("run_context") or {}).get("agent_config") or {}).get("module") or ""
    # text_bridge drives a voice agent class over text, so it is a text run despite the package name
    if "text_agent" in module or "TextAgent" in module or "text_bridge" in module or "TextBridge" in module:
        return "text"
    if "voice_agent" in module or "TalkdeskAgent" in module:
        return "voice"
    for sdir in scenarios[:3]:
        if any((sdir / f"{name}.wav").exists() for name in ("conversation", "user", "agent")):
            return "voice"
    return "text" if scenarios else None


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
        }
        if results:
            entry.update(
                agent_name=results.get("agent_name"),
                run_id=results.get("run_id"),
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
                    "simulator": run_context.get("simulator_config"),
                    "domain": run_context.get("domain"),
                    "split": run_context.get("split"),
                    # tau2-sourced scenarios are named <domain>-tau2-<n>
                    "source": "tau2" if any("-tau2-" in s.name for s in scenarios[:5]) else "talk_bench",
                }
        else:
            # fall back to parsing the agent name out of the directory name
            entry["agent_name"] = run_dir.name.split("-1trials-")[0].rsplit("-", 0)[0]
        if eval_res:
            entry["eval_final"] = eval_res.get("final")
            entry["eval_resolution"] = eval_res.get("resolution")
            entry["eval_experience"] = eval_res.get("experience")
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


def _save_stars(stars: set[str]) -> None:
    # write + replace, like the notes file: a torn write on shared storage would
    # fail to parse and silently read back as "no stars"
    temp_file = STARS_FILE.with_name(f"{STARS_FILE.name}.tmp")
    temp_file.write_text(json.dumps(sorted(stars), indent=1))
    temp_file.replace(STARS_FILE)


def _load_run_notes() -> dict[str, str]:
    notes = _load_json(RUN_NOTES_FILE)
    if not isinstance(notes, dict):
        return {}
    return {
        path: note
        for path, note in notes.items()
        if isinstance(path, str) and isinstance(note, str) and note
    }


def _save_run_notes(notes: dict[str, str]) -> None:
    temp_file = RUN_NOTES_FILE.with_name(f"{RUN_NOTES_FILE.name}.tmp")
    temp_file.write_text(
        json.dumps(dict(sorted(notes.items())), ensure_ascii=False, indent=2) + "\n"
    )
    temp_file.replace(RUN_NOTES_FILE)


@app.get("/api/runs")
def api_runs(refresh: bool = False):
    global _index_cache
    if _index_cache is None or refresh:
        _index_cache = build_index()
    payload = {
        "runs": _index_cache,
        "stars": sorted(_load_stars()),
        "notes": _load_run_notes(),
    }
    if not READ_ONLY:
        payload["data_dir"] = str(DATA_DIR)
    return payload


@app.get("/api/config")
def api_config():
    return {
        "read_only": READ_ONLY,
        "chat_enabled": CHAT_ENABLED,
        "sync_enabled": SYNC_ENABLED,
        "test_runs_enabled": TEST_RUNS_ENABLED,
        "llm_model": LLM_MODEL,
    }


@app.post("/api/star")
def api_star(path: str = Form(...), starred: bool = Form(...)):
    _ensure_writable()
    _data_dir_at_depth(path, 1, "run")
    with _stars_lock:
        stars = _load_stars()
        if starred:
            stars.add(path)
        else:
            stars.discard(path)
        _save_stars(stars)
    return {"path": path, "starred": starred}


@app.post("/api/note")
def api_note(path: str = Form(...), note: str = Form("")):
    _ensure_writable()
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
        _save_run_notes(notes)
    return {"path": path, "note": note}


@app.get("/api/run")
def api_run(path: str = Query(...)):
    run_dir = _data_dir_at_depth(path, 1, "run")
    cached = _run_cache.get(path)
    if cached is not None:
        return cached
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
            meta = _load_json(sdir / "run_meta.json") or {}
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
            "domain": summary.get("domain"),
        })

    top_files = sorted(
        f.name for f in run_dir.iterdir() if f.is_file() and not f.name.startswith(".")
    )
    payload = {
        "path": path,
        "results": results,
        "evaluation": eval_res,
        "scenarios": scenarios,
        "top_files": top_files,
    }
    if len(_run_cache) >= _RUN_CACHE_MAX:
        _run_cache.pop(next(iter(_run_cache)))
    _run_cache[path] = payload
    return payload


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
        "timestamp": _parse_timestamp(path),
        "domains": sorted((results.get("per_domain") or {}).keys()),
        "primary_score": results.get("primary_score"),
        "scenarios_passed": results.get("scenarios_passed"),
        "scenarios_total": results.get("scenarios_total"),
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
    rows = [per.get(i) or {} for i in ids]
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


@app.get("/api/compare")
def api_compare(a: str = Query(...), b: str = Query(...)):
    """Align two runs by scenario id — the point of comparison for text vs voice."""
    if a == b:
        raise HTTPException(status_code=400, detail="pick two different runs")
    side_a, per_a = _compare_side(a)
    side_b, per_b = _compare_side(b)
    shared = sorted(set(per_a) & set(per_b), key=_natural_key)
    scenarios = []
    for sid in shared:
        sa, sb = per_a.get(sid) or {}, per_b.get(sid) or {}
        scenarios.append({
            "id": sid,
            "a": {
                "passed": sa.get("passed"), "goal_score": sa.get("goal_score"),
                "end_reason": sa.get("end_reason"), "turn_count": sa.get("turn_count"),
                "duration_ms": sa.get("total_duration_ms"),
            },
            "b": {
                "passed": sb.get("passed"), "goal_score": sb.get("goal_score"),
                "end_reason": sb.get("end_reason"), "turn_count": sb.get("turn_count"),
                "duration_ms": sb.get("total_duration_ms"),
            },
        })
    counts = Counter()
    for row in scenarios:
        pa, pb = bool(row["a"]["passed"]), bool(row["b"]["passed"])
        counts["both_pass" if pa and pb else "a_only" if pa else "b_only" if pb else "both_fail"] += 1
    side_a["stats"] = _compare_stats(per_a, shared)
    side_b["stats"] = _compare_stats(per_b, shared)
    return {
        "a": side_a,
        "b": side_b,
        "scenarios": scenarios,
        "counts": {k: counts.get(k, 0) for k in ("both_pass", "a_only", "b_only", "both_fail")},
        "only_in_a": sorted(set(per_a) - set(per_b), key=_natural_key),
        "only_in_b": sorted(set(per_b) - set(per_a), key=_natural_key),
    }


@app.get("/api/scenario")
def api_scenario(path: str = Query(...)):
    sdir = _data_dir_at_depth(path, 2, "scenario")
    meta = _load_json(sdir / "run_meta.json") or {}
    # db_state is huge and rarely needed inline; drop it from the payload
    meta.pop("db_state", None)

    # per-scenario scores live in the parent run's result files
    run_dir = sdir.parent
    results = _load_json(run_dir / "results.json") or {}
    scenario_summary = next(
        (s for s in results.get("per_scenario") or [] if s.get("scenario_id") == sdir.name), None
    )
    eval_res = _load_json(run_dir / "talk_bench_evaluation_result.json") or {}
    session_result = next(
        (s for s in eval_res.get("session_results") or [] if s.get("session_id") == sdir.name), None
    )
    if session_result:
        # turns duplicate conversation.json; keep the payload lean
        session_result = {k: v for k, v in session_result.items() if k != "turns"}
    files = []
    for f in sorted(sdir.rglob("*")):
        relative = f.relative_to(sdir)
        if f.is_file() and not any(part.startswith(".") for part in relative.parts):
            files.append({
                "name": str(relative),
                "size": f.stat().st_size,
            })
    return {
        "path": path,
        "scenario": _load_json(sdir / "scenario.json"),
        "conversation": _load_json(sdir / "conversation.json"),
        "tool_log": _load_json(sdir / "tool_log.json"),
        "run_meta": meta,
        "scenario_summary": scenario_summary,
        "session_result": session_result,
        "files": files,
        "audio": {
            name: (sdir / f"{name}.wav").exists()
            for name in ("conversation", "user", "agent")
        },
    }


def _invalidate_index() -> None:
    global _index_cache
    _index_cache = None
    _run_cache.clear()


@app.post("/api/upload_run")
async def upload_run(
    file: UploadFile = File(...),
    overwrite: bool = Form(False),
):
    """Import run directories from a .zip into data/."""
    _ensure_writable()
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="expected a .zip file")

    with tempfile.TemporaryDirectory(dir=_staging_dir()) as tmp:
        zip_path = Path(tmp) / "upload.zip"
        with zip_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        try:
            zf = zipfile.ZipFile(zip_path)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="not a valid zip file")
        with zf:
            entries = [n for n in zf.namelist() if n.strip("/") and "__MACOSX" not in n]
            for n in entries:
                if n.startswith("/") or ".." in Path(n).parts:
                    raise HTTPException(status_code=400, detail=f"unsafe path in zip: {n}")
            root_files = [n for n in entries if "/" not in n.strip("/") and not n.endswith("/")]
            top_dirs = sorted({n.split("/")[0] for n in entries if "/" in n})

            extract_root = Path(tmp) / "extracted"
            zf.extractall(extract_root)

        # zip of loose files (no directories) -> treat as one run named after the zip
        if root_files and not top_dirs:
            run_name = Path(file.filename).stem
            src_runs = {run_name: extract_root}
        else:
            # each top-level directory in the zip is treated as one run
            src_runs = {d: extract_root / d for d in top_dirs}

        imported, skipped = [], []
        for name, src in src_runs.items():
            target = DATA_DIR / name
            if target.exists():
                if not overwrite:
                    skipped.append(name)
                    continue
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(target))
            imported.append(name)

    _invalidate_index()
    return {"imported": imported, "skipped_existing": skipped}


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


def _get_staged(upload_id: str) -> dict[str, Any]:
    st = _staged.get(upload_id)
    if not st:
        raise HTTPException(status_code=404, detail="unknown or expired upload_id")
    return st


@app.post("/api/upload_dir_begin")
def upload_dir_begin(
    run_name: str = Form(...),
    overwrite: bool = Form(False),
):
    _ensure_writable()
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
async def upload_dir_chunk(
    upload_id: str = Form(...),
    paths: list[str] = Form(...),
    files: list[UploadFile] = File(...),
):
    _ensure_writable()
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
    _ensure_writable()
    st = _staged.pop(upload_id, None)
    if not st:
        raise HTTPException(status_code=404, detail="unknown or expired upload_id")
    target = DATA_DIR / st["run_name"]
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(st["dir"]), str(target))
    _invalidate_index()
    return {"run": st["run_name"], "files": st["files"]}


@app.post("/api/upload_dir_abort")
def upload_dir_abort(upload_id: str = Form(...)):
    _ensure_writable()
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
# Tests started from the Run test page always run in this project, so its .env and
# talkbench.yaml apply; the first checkout in _source_repos() order is used.
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
    sig = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            sig.append((str(p.relative_to(root)), st.st_size, int(st.st_mtime)))
    return sig


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
    _ensure_sync_enabled()
    if not _sync_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a sync is already running")
    try:
        repos = _source_repos()
        if not repos:
            looked = ", ".join(str(p) for p in SOURCE_REPO_CANDIDATES)
            raise HTTPException(status_code=404, detail=f"no source checkout found (looked in: {looked})")
        added, updated, unchanged = [], [], 0
        synced: set[str] = set()  # run names are globally unique: first checkout wins
        for repo in repos:
            for proj in SOURCE_PROJECTS:
                src_root = repo / proj / "data" / "runs"
                if not src_root.is_dir():
                    continue
                for run in sorted(src_root.iterdir()):
                    if not run.is_dir() or run.name in synced:
                        continue
                    synced.add(run.name)
                    target = DATA_DIR / run.name
                    exists = target.exists()
                    if exists and _dir_signature(run) == _dir_signature(target):
                        unchanged += 1
                        continue
                    # stage under a temp name so an interrupted copy is never
                    # mistaken for a complete run on the next sync
                    tmp = DATA_DIR / f".sync-tmp-{run.name}"
                    if tmp.exists():
                        shutil.rmtree(tmp)
                    _clone_dir(run, tmp)
                    if exists:
                        shutil.rmtree(target)
                    tmp.rename(target)
                    (updated if exists else added).append(run.name)
        _invalidate_index()
        return {
            "added": added,
            "updated": updated,
            "unchanged": unchanged,
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


def _modality_brief(modality: str | None) -> str:
    brief = _MODALITY_BRIEF.get(modality or "")
    return brief + "\n\n" if brief else ""


def _analyst_role(modality: str | None) -> str:
    return {"text": "text-agent", "voice": "voice-agent"}.get(modality or "", "conversational-agent")


def _asr_cause(modality: str | None, phrase: str) -> str:
    """The ASR item belongs in a root-cause list only when speech was involved."""
    return "" if modality == "text" else phrase


def _scenario_context(sdir: Path, modality: str | None = None) -> str:
    """Assemble the artifacts an analyst would read to diagnose this scenario."""
    run_dir = sdir.parent
    results = _load_json(run_dir / "results.json") or {}
    summary = next(
        (s for s in results.get("per_scenario") or [] if s.get("scenario_id") == sdir.name), None
    )
    eval_res = _load_json(run_dir / "talk_bench_evaluation_result.json") or {}
    session = next(
        (s for s in eval_res.get("session_results") or [] if s.get("session_id") == sdir.name), None
    )
    if session:
        drop = {"turns"}
        if modality == "text":
            # zero-filled placeholders an analyst would otherwise read as a real failure
            drop |= {"responsiveness", "latency"}
        session = {k: v for k, v in session.items() if k not in drop}
    scenario = _load_json(sdir / "scenario.json")
    conversation = _load_json(sdir / "conversation.json")
    tool_log = _load_json(sdir / "tool_log.json")
    meta = _load_json(sdir / "run_meta.json") or {}

    parts = [
        f"Run: {run_dir.name}",
        f"Scenario: {sdir.name}",
        f"Agent: {results.get('agent_name', 'unknown')}",
        f"Modality: {modality or 'unknown'}",
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
        keys = (
            "final", "resolution", "experience", "pass_at1", "mean_pass_rate",
            "responsiveness", "customer_effort", "conversation_quality",
            "voice_consistency", "judge_parse_failure_rate",
        )
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
    not_passed_total = 0
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
            not_passed_total += 1
            end_reasons[str(summary.get("end_reason"))] += 1
            if sdir is not None:
                not_passed_dirs.append((sdir, summary))
    if lines:
        parts.append("## Per-scenario results (one line each)\n" + "\n".join(lines))

    agg_lines = [f"{not_passed_total}/{len(entries)} scenarios did not pass."]
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


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    _ensure_chat_enabled()
    is_run = len(Path(req.path).parts) == 1
    if is_run:
        run_dir = _data_dir_at_depth(req.path, 1, "run")
        modality = _run_dir_modality(run_dir)
        system = (
            f"You are a {_analyst_role(modality)} benchmark analyst. The user is reviewing one "
            "complete talk-bench run in which an agent was tested on many scenarios. Using the "
            "run data below, answer questions about the overall outcome. When asked for a "
            "summary, cover: (1) headline numbers (pass rate, key scores); (2) the dominant "
            "failure patterns, grouped by root cause (agent behavior, tool errors, "
            + _asr_cause(modality, "ASR/transcription issues, ")
            + "user-simulator behavior, judge/scoring artifacts, timeouts or infra "
            "errors), each with a count and example scenario ids; (3) anything else notable. "
            "Ground every claim in the data and cite scenario ids as evidence. Transcript "
            "excerpts are included only for scenarios that did not pass, and may be capped; "
            "if something is not in the context, say so instead of guessing — the user can "
            "open a scenario page and its per-scenario AI chat for a deep dive. "
            + _CHAT_STYLE_RULES + "\n\n" + _modality_brief(modality) + _run_context(run_dir, modality)
        )
    else:
        sdir = _data_dir_at_depth(req.path, 2, "scenario")
        modality = _run_dir_modality(sdir.parent)
        system = (
            f"You are a {_analyst_role(modality)} benchmark analyst. The user is investigating "
            "one talk-bench scenario run. Using the artifacts below, answer questions about what "
            "happened and, when the scenario failed, diagnose the concrete root cause (agent "
            "behavior, tool errors, "
            + _asr_cause(modality, "ASR issues, ")
            + "judge/scoring details, user-simulator behavior, timeouts). "
            "Quote specific turns or tool calls as evidence. "
            + _CHAT_STYLE_RULES + "\n\n" + _modality_brief(modality) + _scenario_context(sdir, modality)
        )
    import litellm  # imported lazily: heavy module, only needed for chat

    try:
        resp = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in req.messages[-20:]],
            timeout=180 if is_run else 90,  # run summaries read far more context
        )
        return {"reply": resp.choices[0].message.content, "model": LLM_MODEL}
    except Exception as e:  # surface provider errors to the UI
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")


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
    _ensure_chat_enabled()
    sdir = _data_dir_at_depth(req.path, 2, "scenario")
    modality = _run_dir_modality(sdir.parent)
    system = (
        f"You are a {_analyst_role(modality)} benchmark analyst. The user pressed a button "
        "asking for a structured report on one talk-bench scenario run. Ground every claim in "
        "the artifacts below and cite specific turns or tool calls as evidence; if something "
        "is not in the context, say so instead of guessing. "
        + _CHAT_STYLE_RULES + "\n\n" + _modality_brief(modality) + _scenario_context(sdir, modality)
    )
    import litellm  # imported lazily: heavy module, only needed for LLM features

    try:
        resp = litellm.completion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": _SCENARIO_REPORT_PROMPT},
            ],
            timeout=120,
        )
        return {"report": resp.choices[0].message.content, "model": LLM_MODEL}
    except Exception as e:  # surface provider errors to the UI
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")


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
    _ensure_chat_enabled()
    run_dir = _data_dir_at_depth(req.path, 1, "run")
    modality = _run_dir_modality(run_dir)
    system = (
        f"You are a {_analyst_role(modality)} benchmark analyst. The user pressed a button "
        "asking for a detailed report on one complete talk-bench run in which an agent was "
        "tested on many scenarios. Ground every claim in the run data below and cite scenario "
        "ids as evidence. Transcript excerpts are included only for scenarios that did not "
        "pass, and may be capped; if something is not in the context, say so instead of "
        "guessing. "
        + _CHAT_STYLE_RULES + "\n\n" + _modality_brief(modality) + _run_context(run_dir, modality)
    )
    import litellm  # imported lazily: heavy module, only needed for LLM features

    try:
        resp = litellm.completion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": _run_report_prompt(modality)},
            ],
            timeout=300,  # large runs: big context and a long structured answer
        )
        return {"report": resp.choices[0].message.content, "model": LLM_MODEL}
    except Exception as e:  # surface provider errors to the UI
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")


# ---- run test: launch talk-bench runs with a user-configured agent ----
#
# A test is exactly
#     uv run talk-bench run --config <agent.yaml> --domain X --split Y --source S [--scenario-ids a,b]
# started as a child process inside one of the local ai-ds-research checkouts;
# every other CLI option keeps its default. The working directory matters: the
# CLI loads the checkout's .env and talkbench.yaml from there and writes the run
# to its data/runs. The agent YAML the user edited or imported is written
# verbatim under <data dir>/.test_jobs/<job>/ next to the log, and the run
# directory is cloned into the archive while it grows and when it ends, so the
# run shows up in the list like any other.

TEST_RUNS_ENABLED = not READ_ONLY and os.environ.get(
    "VIEWER_DISABLE_TEST_RUNS", ""
).strip().lower() not in {"1", "true", "yes", "on"}
JOBS_DIR = DATA_DIR / ".test_jobs"
AGENT_CONFIGS_FILE = DATA_DIR / ".agent_configs.json"
MAX_RUNNING_TESTS = 4
_LOG_CHUNK = 512_000  # bytes of log returned per poll
_SNAPSHOT_INTERVAL_S = 90  # re-clone a running job's partial run dir into the archive this often

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_job_procs: dict[str, subprocess.Popen] = {}
_agent_configs_lock = threading.Lock()
_job_sync_lock = threading.Lock()

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
_TOKEN_RE = re.compile(r"^[\w.:,/=+@%\-]+$")


def _ensure_test_runs_enabled() -> None:
    if not TEST_RUNS_ENABLED:
        raise HTTPException(status_code=403, detail="running tests is disabled on this viewer")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _yaml():
    import yaml  # PyYAML, imported lazily like litellm

    return yaml


def _uv_bin() -> str | None:
    found = shutil.which("uv")
    candidates = ([found] if found else []) + [
        str(Path.home() / ".local" / "bin" / "uv"), "/opt/homebrew/bin/uv", "/usr/local/bin/uv",
    ]
    return next((c for c in candidates if Path(c).is_file()), None)


def _launcher(root: Path) -> list[str] | None:
    """How to invoke the CLI in a checkout: `uv run talk-bench` (what the docs use),
    or the venv's shim when uv is not installed."""
    uv = _uv_bin()
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


def _split_entries(splits: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"split": str(k), "count": len(v) if isinstance(v, (list, dict)) else None}
        for k, v in splits.items()
    ]


def _scan_domains(root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """domain -> splits (with task counts) per dataset source, read from the checkout's
    own data files so the pickers offer exactly what the CLI accepts."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {"talk_bench": {}, "tau2": {}}
    lib = _talk_bench_lib_root(root)
    if lib is None:
        return out
    native = lib / "libs" / "talk_bench_domains" / "src" / "talk_bench_domains"
    if native.is_dir():
        for d in sorted(native.iterdir()):
            if not d.is_dir() or d.name.startswith(("_", ".")):
                continue
            splits = _load_json(d / "splits.json")
            if isinstance(splits, dict) and splits:
                out["talk_bench"][d.name] = _split_entries(splits)
            elif (d / "tasks.json").is_file():
                tasks = _load_json(d / "tasks.json")
                out["talk_bench"][d.name] = [
                    {"split": "base", "count": len(tasks) if isinstance(tasks, list) else None}
                ]
    tau2 = lib / "vendor" / "tau2-bench" / "data" / "tau2" / "domains"
    if tau2.is_dir():
        for d in sorted(tau2.iterdir()):
            splits = _load_json(d / "split_tasks.json") if d.is_dir() else None
            if isinstance(splits, dict) and splits:
                out["tau2"][d.name] = _split_entries(splits)
    return out


# scenario ids as talk-bench's data files spell them: tau2 telecom ids look like
# "telecom-tau2-[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]"
_ID_RE = re.compile(r"^[\w.\-\[\]:|]+$")
MAX_SCENARIO_ID_LEN = 300
MAX_SCENARIO_IDS = 2000


def _clip_summary(text: Any, limit: int = 160) -> str:
    return _clip(text or "", limit)


def _scenario_catalog(root: Path, source: str, domain: str, split: str) -> list[dict[str, Any]]:
    """The scenario ids `--domain/--split/--source` select, read from the same data files
    the CLI uses, each with a one-line summary for the picker. Comma-joined splits are
    unioned and `all` covers every domain, mirroring talk-bench."""
    lib = _talk_bench_lib_root(root)
    if lib is None:
        return []
    parts = [x.strip() for x in split.split(",") if x.strip()]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(sid: str, dom: str, summary: str, extra: dict[str, Any]) -> None:
        if sid not in seen:
            seen.add(sid)
            out.append({"id": sid, "domain": dom, "summary": summary, **extra})

    if source == "tau2":
        base = lib / "vendor" / "tau2-bench" / "data" / "tau2" / "domains"
        domains = sorted(d.name for d in base.iterdir() if d.is_dir()) if domain == "all" and base.is_dir() else [domain]
        for dom in domains:
            splits = _load_json(base / dom / "split_tasks.json")
            tasks = _load_json(base / dom / "tasks.json")
            if not isinstance(splits, dict):
                continue
            by_id = {str(t.get("id")): t for t in tasks if isinstance(t, dict)} if isinstance(tasks, list) else {}
            for part in parts:
                for tid in splits.get(part) or []:
                    task = by_id.get(str(tid), {})
                    desc = task.get("description") if isinstance(task.get("description"), dict) else {}
                    instr = ((task.get("user_scenario") or {}).get("instructions") or {}) if isinstance(task.get("user_scenario"), dict) else {}
                    summary = desc.get("purpose") or instr.get("reason_for_call") or ""
                    add(f"{dom}-tau2-{tid}", dom, _clip_summary(summary), {})
        return out

    base = lib / "libs" / "talk_bench_domains" / "src" / "talk_bench_domains"
    domains = (
        sorted(d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith(("_", ".")))
        if domain == "all" and base.is_dir() else [domain]
    )
    for dom in domains:
        splits = _load_json(base / dom / "splits.json")
        tasks = _load_json(base / dom / "tasks.json")
        by_id = {str(t.get("id")): t for t in tasks if isinstance(t, dict)} if isinstance(tasks, list) else {}
        if isinstance(splits, dict):
            ids = [sid for part in parts for sid in (splits.get(part) or [])]
        elif by_id and parts == ["base"]:  # domains that ship tasks.json only
            ids = list(by_id)
        else:
            ids = []
        for sid in ids:
            task = by_id.get(str(sid), {})
            scenario = task.get("user_scenario") if isinstance(task.get("user_scenario"), dict) else {}
            summary = task.get("description") or scenario.get("reason_for_call") or ""
            extra = {k: task[k] for k in ("difficulty", "tags") if task.get(k)}
            add(str(sid), dom, _clip_summary(summary), extra)
    return out


def _project_manifest(root: Path) -> dict[str, Any]:
    manifest = root / "talkbench.yaml"
    if not manifest.is_file():
        return {}
    try:
        doc = _yaml().safe_load(manifest.read_text())
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
            doc = _yaml().safe_load(f.read_text())
        except Exception:
            continue
        agent = doc.get("agent") if isinstance(doc, dict) else None
        if isinstance(agent, dict) and agent.get("module"):
            out.append({"file": f.name, "name": agent.get("name"), "module": agent.get("module")})
    return out


def _test_project_root() -> Path:
    """The talk-bench checkout every test runs in: the first one _source_repos() finds."""
    for repo in _source_repos():
        root = repo / TEST_PROJECT
        if (root / "pyproject.toml").is_file():
            return root
    raise HTTPException(status_code=404, detail="no talk-bench checkout was found to run the test in")


def _test_project() -> dict[str, Any] | None:
    """That checkout described for the Run test page: launcher, .env, samples, and the
    domains / splits its data files define."""
    try:
        root = _test_project_root()
    except HTTPException:
        return None
    launcher = _launcher(root)
    manifest = _project_manifest(root)
    simulator = manifest.get("simulator")
    return {
        "id": str(root),
        "name": TEST_PROJECT,
        "repo": str(root.parent),
        "has_env": (root / ".env").is_file(),
        "launcher": " ".join([Path(launcher[0]).name, *launcher[1:]]) if launcher else None,
        "output_dir": str((manifest.get("defaults") or {}).get("output_dir") or "data/runs"),
        "samples": _sample_configs(root),
        "defaults": simulator if isinstance(simulator, dict) else {},
        "domains": _scan_domains(root),
    }


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
            detail="agent.module must be a dotted Python path like voice_agent.cascaded.CascadedAgent",
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


def _split_config_doc(doc: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """(agent block, other top-level blocks) of a parsed agent YAML. talk-bench accepts
    both `agent: {…}` with siblings and a flat agent mapping."""
    if not isinstance(doc, dict):
        raise HTTPException(status_code=400, detail="config must be a YAML/JSON mapping")
    if isinstance(doc.get("agent"), dict):
        extra = {k: v for k, v in doc.items() if k in _EXTRA_BLOCKS and isinstance(v, dict)}
        return _normalize_agent(doc["agent"]), extra
    return _normalize_agent(doc), {}


def _parse_config_text(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        doc = _yaml().safe_load(text)
    except Exception as e:  # yaml.YAMLError carries the line/column
        raise HTTPException(status_code=400, detail=f"invalid YAML/JSON: {e}")
    return _split_config_doc(doc)


def _dump_yaml(doc: dict[str, Any]) -> str:
    return _yaml().safe_dump(doc, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _archived_agents() -> list[dict[str, Any]]:
    """Distinct agent configs recorded in archived runs, newest first."""
    global _index_cache
    if _index_cache is None:
        _index_cache = build_index()
    by_run = {r["path"]: r for r in _index_cache}
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
            "split": ctx.get("split"), "source": ctx.get("source"), "modality": entry.get("modality"),
        }
        g = groups.get(key)
        if g is None:
            groups[key] = {"agent": agent, "yaml": _dump_yaml({"agent": agent}), "runs": 1, **latest}
        else:
            g["runs"] += 1
            if stamp > g["timestamp"]:
                g.update(latest)
    return sorted(groups.values(), key=lambda g: g["timestamp"], reverse=True)


# ---- saved agent configs ----

def _load_agent_configs() -> list[dict[str, Any]]:
    data = _load_json(AGENT_CONFIGS_FILE)
    if not isinstance(data, list):
        return []
    return [c for c in data if isinstance(c, dict) and c.get("id") and isinstance(c.get("text"), str)]


def _save_agent_configs(configs: list[dict[str, Any]]) -> None:
    tmp = AGENT_CONFIGS_FILE.with_name(AGENT_CONFIGS_FILE.name + ".tmp")
    tmp.write_text(json.dumps(configs, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(AGENT_CONFIGS_FILE)


class AgentConfigBody(BaseModel):
    id: str | None = None
    label: str | None = None
    text: str  # the agent YAML as typed


@app.get("/api/test/agent_configs")
def api_agent_configs():
    _ensure_test_runs_enabled()
    return {"configs": _load_agent_configs()}


@app.post("/api/test/agent_configs")
def api_save_agent_config(body: AgentConfigBody):
    _ensure_test_runs_enabled()
    if len(body.text) > 200_000:
        raise HTTPException(status_code=413, detail="agent config is too large")
    agent, _extra = _parse_config_text(body.text)
    label = (body.label or agent["name"]).strip()[:80] or agent["name"]
    with _agent_configs_lock:
        configs = _load_agent_configs()
        entry = next((c for c in configs if c["id"] == body.id), None) if body.id else None
        if entry is None:  # same label → update in place rather than piling up copies
            entry = next((c for c in configs if c.get("label") == label), None)
        if entry is None:
            entry = {"id": uuid.uuid4().hex[:12], "created": _now_iso()}
            configs.append(entry)
        entry.update(
            label=label, text=body.text.rstrip() + "\n", name=agent["name"], module=agent["module"],
            updated=_now_iso(),
        )
        _save_agent_configs(configs)
    return {"config": entry, "configs": configs}


@app.delete("/api/test/agent_configs/{config_id}")
def api_delete_agent_config(config_id: str):
    _ensure_test_runs_enabled()
    with _agent_configs_lock:
        configs = _load_agent_configs()
        kept = [c for c in configs if c["id"] != config_id]
        if len(kept) == len(configs):
            raise HTTPException(status_code=404, detail="no such saved config")
        _save_agent_configs(kept)
    return {"configs": kept}


# ---- options, import helpers ----

@app.get("/api/test/options")
def api_test_options():
    _ensure_test_runs_enabled()
    return {
        "project": _test_project(),
        "archived": _archived_agents(),
        "saved": _load_agent_configs(),
        "uv": _uv_bin() is not None,
        "limits": {"max_running": MAX_RUNNING_TESTS},
    }


@app.get("/api/test/scenarios")
def api_test_scenarios(
    source: str = Query("talk_bench"), domain: str = Query(...), split: str = Query("base"),
):
    """What `--domain/--split/--source` resolve to, for the scenario picker (--scenario-ids)."""
    _ensure_test_runs_enabled()
    root = _test_project_root()
    if source not in ("tau2", "talk_bench"):
        raise HTTPException(status_code=400, detail="source must be tau2 or talk_bench")
    domain = _check_token(domain, "domain", re.compile(r"^[a-z0-9_\-]+$"))
    split = _check_token(split or "base", "split", re.compile(r"^[A-Za-z0-9_:,\-]+$"))
    scenarios = _scenario_catalog(root, source, domain, split)
    return {"scenarios": scenarios, "count": len(scenarios)}


@app.get("/api/test/sample")
def api_test_sample(file: str = Query(...)):
    _ensure_test_runs_enabled()
    root = _test_project_root()
    if Path(file).name != file or Path(file).suffix not in (".yaml", ".yml"):
        raise HTTPException(status_code=400, detail="invalid sample file name")
    path = root / "samples" / file
    if not path.is_file():
        raise HTTPException(status_code=404, detail="sample not found")
    text = path.read_text()
    agent, extra = _parse_config_text(text)
    return {"agent": agent, "extra_blocks": extra, "text": text, "file": file}


class ConfigTextBody(BaseModel):
    text: str


@app.post("/api/test/parse_config")
def api_parse_config(body: ConfigTextBody):
    _ensure_test_runs_enabled()
    if len(body.text) > 200_000:
        raise HTTPException(status_code=413, detail="config text is too large")
    agent, extra = _parse_config_text(body.text)
    return {"agent": agent, "extra_blocks": extra}


@app.post("/api/test/parse_yaml")
def api_parse_yaml(body: ConfigTextBody):
    """Any YAML fragment -> its value. The form's free-form fields (nested mappings it has no
    typed control for) are typed as YAML and parsed here, so they read exactly as talk-bench
    will read them."""
    _ensure_test_runs_enabled()
    if len(body.text) > 50_000:
        raise HTTPException(status_code=413, detail="fragment is too large")
    try:
        value = _yaml().safe_load(body.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid YAML: {e}")
    return {"value": value}


class RenderConfigBody(BaseModel):
    agent: dict[str, Any]


@app.post("/api/test/render_config")
def api_render_config(body: RenderConfigBody):
    """The agent_config recorded in a run's results.json -> the agent.yaml text talk-bench reads."""
    _ensure_test_runs_enabled()
    agent = _normalize_agent(body.agent)
    return {"yaml": _dump_yaml({"agent": agent}), "agent": agent}


# ---- jobs ----

def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _save_job(job: dict[str, Any]) -> None:
    d = _job_dir(job["id"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "job.json.tmp"
    tmp.write_text(json.dumps(_public_job(job), ensure_ascii=False, indent=1))
    tmp.replace(d / "job.json")


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
        "run_dir": None, "run_name": None, "run_path": None, "synced_at": None,
        "progress": {"done": 0, "total": None}, "score": None, "error": None, "log_size": 0,
        "resume_of": None, "scenario_ids": [],
        **fields,
    }


def _start_job(job: dict[str, Any], cmd: list[str], root: Path) -> None:
    d = _job_dir(job["id"])
    d.mkdir(parents=True, exist_ok=True)
    log_path = d / "log.txt"
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
        _job_procs[job["id"]] = proc
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
    job["log_size"] = size
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
        m = _RESULTS_RE.match(line)
        if m:
            run_dir = Path(m.group(1))
            if not run_dir.is_absolute():
                run_dir = Path(job["cwd"]) / run_dir
            job["run_dir"] = str(run_dir)
            job["run_name"] = run_dir.name
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
    best: tuple[float, Path] | None = None
    for d in runs_root.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix) or str(d) in claimed:
            continue
        try:
            st = d.stat()
        except OSError:
            continue
        born = getattr(st, "st_birthtime", st.st_mtime)
        if born < job["started"] - 5 or not (d / "run_config.json").exists():
            continue
        if best is None or born < best[0]:
            best = (born, d)
    if best:
        job["run_dir"] = str(best[1])
        job["run_name"] = best[1].name


def _sync_job_run(job: dict[str, Any]) -> bool:
    """Clone the job's run directory into the archive (a no-op when unchanged)."""
    run_dir = Path(job["run_dir"]) if job.get("run_dir") else None
    if not run_dir or not run_dir.is_dir():
        return False
    if not (run_dir / "results.json").is_file() and not _scenario_dirs(run_dir):
        return False  # nothing to browse yet (a config error leaves only run_config.json behind)
    target = DATA_DIR / run_dir.name
    if run_dir.resolve() == target.resolve():  # the checkout writes straight into the archive
        job["run_path"] = run_dir.name
        return False
    with _job_sync_lock:
        if target.exists() and _dir_signature(run_dir) == _dir_signature(target):
            job["run_path"] = run_dir.name
            return False
        tmp = DATA_DIR / f".sync-tmp-{run_dir.name}"
        if tmp.exists():
            shutil.rmtree(tmp)
        _clone_dir(run_dir, tmp)
        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
    job["run_path"] = run_dir.name
    job["synced_at"] = time.time()
    _invalidate_index()
    return True


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
    last_meta_count = -1
    while True:
        alive = (proc.poll() is None) if proc is not None else _pid_alive(job.get("pid"))
        _ingest_log(job, log_path)
        if not job.get("run_dir"):
            _locate_run_dir(job)
        if alive and job.get("run_dir") and time.time() - last_snapshot >= _SNAPSHOT_INTERVAL_S:
            last_snapshot = time.time()
            done = sum(1 for _ in Path(job["run_dir"]).glob("*/run_meta.json"))
            if done != last_meta_count:  # only re-clone when a scenario finished since
                last_meta_count = done
                try:
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
        has_results = bool(job.get("run_dir")) and (Path(job["run_dir"]) / "results.json").is_file()
        if job["status"] == "cancelling":
            job["status"] = "cancelled"
        elif exit_code in (0, 1) or (exit_code is None and has_results):
            job["status"] = "done"  # 0 = every scenario passed, 1 = some did not; the run completed
        elif exit_code is None:
            job["status"] = "interrupted"
        else:
            job["status"] = "error"  # 2 = configuration, 3 = runtime
        _job_procs.pop(job_id, None)
    if job["status"] == "error" and not job.get("error"):
        job["error"] = _last_log_lines(log_path)
    try:
        _sync_job_run(job)
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
        _jobs[job["id"]] = job
        if job.get("status") not in ("running", "cancelling"):
            continue
        if _pid_alive(job.get("pid")):
            threading.Thread(target=_watch_job, args=(job["id"], None), daemon=True).start()
        else:
            job["status"] = "interrupted"
            job["finished"] = job.get("finished") or time.time()
            job["error"] = job.get("error") or "the viewer restarted while this test was running"
            _save_job(job)
            threading.Thread(target=_sync_job_run, args=(job,), daemon=True).start()


if TEST_RUNS_ENABLED:
    _recover_jobs()


def _check_token(value: str, label: str, pattern: re.Pattern[str] = _TOKEN_RE, max_len: int = 200) -> str:
    value = value.strip()
    if not value or len(value) > max_len or not pattern.match(value):
        raise HTTPException(status_code=400, detail=f"invalid {label}: {value[:60]!r}")
    return value


class TestJobBody(BaseModel):
    config_text: str  # the agent YAML, written verbatim and passed as --config
    source: str = "talk_bench"
    domain: str
    split: str = "base"
    scenario_ids: list[str] = []  # optional --scenario-ids; empty = the whole split


@app.get("/api/test/jobs")
def api_test_jobs():
    _ensure_test_runs_enabled()
    jobs = sorted(list(_jobs.values()), key=lambda j: j.get("started") or 0, reverse=True)
    return {"jobs": [_public_job(j) for j in jobs[:200]]}


@app.post("/api/test/jobs")
def api_start_test(body: TestJobBody):
    """Start `talk-bench run --config <agent.yaml> --domain X --split Y --source S
    [--scenario-ids a,b]` in the talk-bench checkout."""
    _ensure_test_runs_enabled()
    root = _test_project_root()
    launcher = _launcher(root)
    if launcher is None:
        raise HTTPException(status_code=500, detail="neither uv nor .venv/bin/talk-bench was found for this checkout")
    if len(body.config_text) > 200_000:
        raise HTTPException(status_code=413, detail="agent config is too large")
    agent, _extra = _parse_config_text(body.config_text)  # same checks talk-bench's AgentConfig makes
    if body.source not in ("tau2", "talk_bench"):
        raise HTTPException(status_code=400, detail="source must be tau2 or talk_bench")
    domain = _check_token(body.domain, "domain", re.compile(r"^[a-z0-9_\-]+$"))
    split = _check_token(body.split or "base", "split", re.compile(r"^[A-Za-z0-9_:,\-]+$"))
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
    running = sum(1 for j in list(_jobs.values()) if j["status"] in ("running", "cancelling"))
    if running >= MAX_RUNNING_TESTS:
        raise HTTPException(status_code=409, detail=f"{MAX_RUNNING_TESTS} tests are already running — wait for one to finish")
    manifest = _project_manifest(root)
    output_dir = str((manifest.get("defaults") or {}).get("output_dir") or "data/runs")
    job = _new_job(
        project=str(root), project_name=root.name, runs_root=str(root / output_dir),
        agent_name=agent["name"], module=agent["module"],
        source=body.source, domain=domain, split=split, scenario_ids=scenario_ids,
    )
    d = _job_dir(job["id"])
    d.mkdir(parents=True, exist_ok=True)
    config_path = d / "agent.yaml"
    config_path.write_text(body.config_text.rstrip() + "\n")
    cmd = [
        *launcher, "run", "--config", str(config_path),
        "--domain", domain, "--split", split, "--source", body.source,
    ]
    if scenario_ids:
        cmd += ["--scenario-ids", ",".join(scenario_ids)]
    _start_job(job, cmd, root)
    return {"job": _public_job(job)}


@app.get("/api/test/jobs/{job_id}")
def api_test_job(job_id: str, offset: int = 0):
    _ensure_test_runs_enabled()
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
        "job": _public_job(job), "log": text, "next_offset": offset, "log_size": size, "more": more,
        "config_yaml": config_path.read_text() if config_path.is_file() else None,
    }


@app.post("/api/test/jobs/{job_id}/cancel")
def api_cancel_test(job_id: str):
    _ensure_test_runs_enabled()
    job = _get_job(job_id)
    if job["status"] not in ("running", "cancelling"):
        raise HTTPException(status_code=409, detail="this test is not running")
    job["status"] = "cancelling"
    pid = job.get("pid")

    def _signal(sig: int) -> None:
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    _signal(signal.SIGTERM)  # talk-bench has no graceful stop; the run can be resumed afterwards

    def _force() -> None:
        time.sleep(8)
        if _pid_alive(pid):
            _signal(signal.SIGKILL)

    threading.Thread(target=_force, daemon=True).start()
    _save_job(job)
    return {"job": _public_job(job)}


@app.post("/api/test/jobs/{job_id}/resume")
def api_resume_test(job_id: str):
    """`talk-bench run --resume <run dir>`: redo errored / unfinished scenarios of a
    cancelled or crashed test and merge into the same results.json."""
    _ensure_test_runs_enabled()
    parent = _get_job(job_id)
    if parent["status"] in ("running", "cancelling"):
        raise HTTPException(status_code=409, detail="this test is still running")
    run_dir = parent.get("run_dir")
    if not run_dir or not (Path(run_dir) / "run_config.json").is_file():
        raise HTTPException(status_code=409, detail="this test left no resumable run directory")
    root = Path(parent["project"])
    if not (root / "pyproject.toml").is_file():
        raise HTTPException(status_code=404, detail="the checkout this test ran in is gone")
    launcher = _launcher(root)
    if launcher is None:
        raise HTTPException(status_code=500, detail="neither uv nor .venv/bin/talk-bench was found for this checkout")
    inherited = {
        k: parent.get(k) for k in (
            "project", "project_name", "runs_root", "agent_name", "module", "source", "domain", "split",
            "scenario_ids",
        )
    }
    job = _new_job(**inherited, resume_of=parent["id"], run_dir=run_dir, run_name=Path(run_dir).name)
    _job_dir(job["id"]).mkdir(parents=True, exist_ok=True)
    src = _job_dir(parent["id"]) / "agent.yaml"
    if src.is_file():
        shutil.copyfile(src, _job_dir(job["id"]) / "agent.yaml")
    _start_job(job, [*launcher, "run", "--resume", run_dir], root)
    return {"job": _public_job(job)}


@app.post("/api/test/jobs/{job_id}/sync")
def api_sync_test(job_id: str):
    """Copy the run directory into the archive now (partial results of a running test)."""
    _ensure_test_runs_enabled()
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
    _ensure_test_runs_enabled()
    job = _get_job(job_id)
    if job["status"] in ("running", "cancelling"):
        raise HTTPException(status_code=409, detail="stop the test before removing it")
    with _jobs_lock:
        _jobs.pop(job_id, None)
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    return {"deleted": job_id}



@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


# Raw file access for audio playback and artifact inspection; supports Range requests.
app.mount("/data", PublicDataFiles(directory=DATA_DIR), name="data")
