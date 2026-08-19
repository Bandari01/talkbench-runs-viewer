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
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"

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
    for project in sorted(DATA_DIR.iterdir()):
        if not project.is_dir():
            continue
        for run_dir in sorted(project.iterdir()):
            if not run_dir.is_dir():
                continue
            results = _load_json(run_dir / "results.json")
            eval_res = _load_json(run_dir / "talk_bench_evaluation_result.json")
            scenarios = _scenario_dirs(run_dir)
            entry: dict[str, Any] = {
                "path": f"{project.name}/{run_dir.name}",
                "project": project.name,
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


@app.get("/api/runs")
def api_runs(refresh: bool = False):
    global _index_cache
    if _index_cache is None or refresh:
        _index_cache = build_index()
    return {"runs": _index_cache, "data_dir": str(DATA_DIR)}


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


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


# raw file access (audio playback, ndjson/json download) — supports Range requests
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
