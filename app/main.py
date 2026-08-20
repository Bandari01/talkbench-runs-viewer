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
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# LLM credentials and model come from this project's own .env (see .env.example)
load_dotenv(BASE_DIR / ".env")
LLM_MODEL = os.environ.get("VIEWER_LLM_MODEL", "azure/gpt-4.1")

TS_RE = re.compile(r"(\d{8}T\d{6}Z)")

app = FastAPI(title="talk-bench runs viewer")

_index_cache: list[dict[str, Any]] | None = None


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
        if not run_dir.is_dir():
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


def _load_stars() -> set[str]:
    stars = _load_json(STARS_FILE)
    return set(stars) if isinstance(stars, list) else set()


def _save_stars(stars: set[str]) -> None:
    STARS_FILE.write_text(json.dumps(sorted(stars), indent=1))


@app.get("/api/runs")
def api_runs(refresh: bool = False):
    global _index_cache
    if _index_cache is None or refresh:
        _index_cache = build_index()
    return {"runs": _index_cache, "stars": sorted(_load_stars()), "data_dir": str(DATA_DIR)}


@app.post("/api/star")
def api_star(path: str = Form(...), starred: bool = Form(...)):
    if not (_safe_path(path)).is_dir():
        raise HTTPException(status_code=404, detail="run not found")
    with _stars_lock:
        stars = _load_stars()
        if starred:
            stars.add(path)
        else:
            stars.discard(path)
        _save_stars(stars)
    return {"path": path, "starred": starred}


@app.get("/api/run")
def api_run(path: str = Query(...)):
    run_dir = _safe_path(path)
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="run not found")
    results = _load_json(run_dir / "results.json")
    eval_res = _load_json(run_dir / "talk_bench_evaluation_result.json")

    per_scenario = {s.get("scenario_id"): s for s in (results or {}).get("per_scenario") or []}
    scenarios = []
    for sdir in _scenario_dirs(run_dir):
        meta = _load_json(sdir / "run_meta.json") or {}
        summary = per_scenario.get(sdir.name) or {}
        scenarios.append({
            "id": sdir.name,
            "passed": summary.get("passed"),
            "goal_score": summary.get("goal_score"),
            "end_reason": summary.get("end_reason") or meta.get("end_reason"),
            "turn_count": summary.get("turn_count"),
            "duration_ms": summary.get("total_duration_ms") or meta.get("total_duration_ms"),
            "domain": summary.get("domain"),
        })

    top_files = sorted(
        f.name for f in run_dir.iterdir() if f.is_file() and not f.name.startswith(".")
    )
    return {
        "path": path,
        "results": results,
        "evaluation": eval_res,
        "scenarios": scenarios,
        "top_files": top_files,
    }


@app.get("/api/scenario")
def api_scenario(path: str = Query(...)):
    sdir = _safe_path(path)
    if not sdir.is_dir():
        raise HTTPException(status_code=404, detail="scenario not found")
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
        if f.is_file() and not f.name.startswith("."):
            files.append({
                "name": str(f.relative_to(sdir)),
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


@app.post("/api/upload_run")
async def upload_run(
    file: UploadFile = File(...),
    overwrite: bool = Form(False),
):
    """Import run directories from a .zip into data/."""
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
    st = _staged.pop(upload_id, None)
    if st:
        shutil.rmtree(st["dir"], ignore_errors=True)
    return {"aborted": bool(st)}


@app.post("/api/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    target_dir: str = Form(...),
    overwrite: bool = Form(False),
):
    """Add or replace a single file inside an existing run/scenario directory."""
    dirp = _safe_path(target_dir)
    if not dirp.is_dir():
        raise HTTPException(status_code=404, detail="target directory not found")
    name = Path(file.filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="missing filename")
    dest = dirp / name
    if dest.exists() and not overwrite:
        raise HTTPException(status_code=409, detail=f"{name} already exists (set overwrite)")
    replaced = dest.exists()
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    _invalidate_index()
    return {"path": f"{target_dir}/{name}", "replaced": replaced}


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
    sdir = _safe_path(req.path)
    if not sdir.is_dir():
        raise HTTPException(status_code=404, detail="scenario not found")
    system = (
        "You are a voice-agent benchmark analyst. The user is investigating one talk-bench "
        "scenario run. Using the artifacts below, answer questions about what happened and, "
        "when the scenario failed, diagnose the concrete root cause (agent behavior, tool "
        "errors, ASR issues, judge/scoring details, user-simulator behavior, timeouts). "
        "Quote specific turns or tool calls as evidence. Be direct and concise. "
        "Answer in the same language the user writes in.\n\n"
        + _scenario_context(sdir)
    )
    import litellm  # imported lazily: heavy module, only needed for chat

    try:
        resp = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in req.messages[-20:]],
            temperature=0.2,
            timeout=90,
        )
        return {"reply": resp.choices[0].message.content, "model": LLM_MODEL}
    except Exception as e:  # surface provider errors to the UI
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


# raw file access (audio playback, ndjson/json download) — supports Range requests
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
