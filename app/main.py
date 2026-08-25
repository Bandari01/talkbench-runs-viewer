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
import shutil
import subprocess
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# LLM credentials and model come from this project's own .env (see .env.example)
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
# Run detail payloads are costly to build; cache a few, cleared on data changes.
_run_cache: dict[str, dict[str, Any]] = {}
_RUN_CACHE_MAX = 16


@app.middleware("http")
async def shared_view_guard(request: Request, call_next):
    # Reject public writes before FastAPI parses a potentially large upload body.
    chat_request = request.method == "POST" and request.url.path == "/api/chat" and CHAT_ENABLED
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


def build_index() -> list[dict[str, Any]]:
    runs = []
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
        else:
            # fall back to parsing the agent name out of the directory name
            entry["agent_name"] = run_dir.name.split("-1trials-")[0].rsplit("-", 0)[0]
        if eval_res:
            entry["eval_final"] = eval_res.get("final")
            entry["eval_resolution"] = eval_res.get("resolution")
            entry["eval_experience"] = eval_res.get("experience")
        runs.append(entry)
    runs.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
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
    STARS_FILE.write_text(json.dumps(sorted(stars), indent=1))


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

    with tempfile.TemporaryDirectory() as tmp:
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
        "dir": Path(tempfile.mkdtemp(prefix="talkbench_upload_")),
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

SOURCE_REPOS = [Path.home() / "GitHub" / f"ai-ds-research{suffix}" for suffix in ("", "-1", "-2", "-3", "-4")]
SOURCE_PROJECTS = ("talk-bench", "talk-bench-talkdesk")
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


@app.post("/api/sync_sources")
def sync_sources():
    _ensure_sync_enabled()
    if not _sync_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a sync is already running")
    try:
        added, updated, unchanged = [], [], 0
        for repo in SOURCE_REPOS:
            for proj in SOURCE_PROJECTS:
                src_root = repo / proj / "data" / "runs"
                if not src_root.is_dir():
                    continue
                for run in sorted(src_root.iterdir()):
                    if not run.is_dir():
                        continue
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
        return {"added": added, "updated": updated, "unchanged": unchanged}
    finally:
        _sync_lock.release()


# ---- scenario Q&A chat ----

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    path: str  # scenario path: <run>/<scenario>
    messages: list[ChatMessage]


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n…[truncated, {len(text)} chars total]"


def _scenario_context(sdir: Path) -> str:
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
        session = {k: v for k, v in session.items() if k != "turns"}
    scenario = _load_json(sdir / "scenario.json")
    conversation = _load_json(sdir / "conversation.json")
    tool_log = _load_json(sdir / "tool_log.json")
    meta = _load_json(sdir / "run_meta.json") or {}

    parts = [
        f"Run: {run_dir.name}",
        f"Scenario: {sdir.name}",
        f"Agent: {results.get('agent_name', 'unknown')}",
        f"End reason: {meta.get('end_reason')} · duration: {round((meta.get('total_duration_ms') or 0) / 1000)}s",
    ]
    if scenario:
        parts.append("## Scenario definition (goal, persona, tasks)\n" + _truncate(json.dumps(scenario, indent=1), 8000))
    if summary:
        parts.append("## Scenario result (results.json per_scenario entry)\n" + _truncate(json.dumps(summary, indent=1), 4000))
    if session:
        parts.append("## Evaluation session result (task_completion judge output)\n" + _truncate(json.dumps(session, indent=1), 6000))
    if conversation:
        lines = [f"[turn {t.get('turn_index')}] {t.get('speaker')}: {t.get('text')}" for t in conversation]
        parts.append("## Transcript\n" + _truncate("\n".join(lines), 20000))
    if tool_log:
        calls = [
            {"tool": c.get("tool_name"), "arguments": c.get("arguments"), "result": c.get("result")}
            for c in tool_log
        ]
        parts.append("## Tool calls\n" + _truncate(json.dumps(calls, indent=1), 15000))
    return "\n\n".join(parts)


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    _ensure_chat_enabled()
    sdir = _data_dir_at_depth(req.path, 2, "scenario")
    system = (
        "You are a voice-agent benchmark analyst. The user is investigating one talk-bench "
        "scenario run. Using the artifacts below, answer questions about what happened and, "
        "when the scenario failed, diagnose the concrete root cause (agent behavior, tool "
        "errors, ASR issues, judge/scoring details, user-simulator behavior, timeouts). "
        "Quote specific turns or tool calls as evidence. Be direct and concise. "
        "Answer in the same language the user writes in. When answering in Chinese, keep "
        "benchmark and technical terms such as Agent, User Simulator, tool call, ASR, LLM, "
        "and API in English; in particular, never translate Agent as '代理'. Preserve product "
        "names, model names, function names, and field names exactly as written in the artifacts.\n\n"
        + _scenario_context(sdir)
    )
    import litellm  # imported lazily: heavy module, only needed for chat

    try:
        resp = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in req.messages[-20:]],
            timeout=90,
        )
        return {"reply": resp.choices[0].message.content, "model": LLM_MODEL}
    except Exception as e:  # surface provider errors to the UI
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


# Raw file access for audio playback and artifact inspection; supports Range requests.
app.mount("/data", PublicDataFiles(directory=DATA_DIR), name="data")
