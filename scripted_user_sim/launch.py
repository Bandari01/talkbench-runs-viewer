"""`talk-bench` where tasks that carry ``metadata.user_script`` get a scripted caller.

    cd <ai-ds-research>/talk-bench
    uv run python <viewer>/scripted_user_sim/launch.py run --config agent.yaml \
        --domain airline --scenario-ids airline-scripted-001

Same arguments, output, run directory and exit codes as ``uv run talk-bench …``. In
this process only (nothing in the checkout changes):

- every ``tasks/<domain>/`` directory next to this file is registered as extra scenario
  data for that built-in domain (tasks are added, splits are unioned), so
  ``airline-scripted-001`` exists alongside the built-in airline tasks;
- a scenario whose metadata has ``user_script`` is driven by ``ScriptedUserSimulator``
  (no LLM decides what the caller says); every other scenario keeps talk-bench's own
  simulator;
- talk-bench's simulator / mid-turn-speak / terminal-hangup gates are turned off if
  talkbench.yaml enables them — they are LLM checkers that would rewrite or veto the
  scripted lines;
- the run directory gets ``scripted_user_simulator.json`` (which scenarios were
  scripted) and ``scripted_user_sim/<scenario>.attempt<N>.ndjson`` (one line per wake:
  event, the agent text the caller heard, the rule that fired, the action taken).
"""

from __future__ import annotations

import inspect
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"
MARKER_FILE = "scripted_user_simulator.json"
TRACE_DIR = "scripted_user_sim"
_GATE_FIELDS = (
    "simulator_gate_enabled",
    "mid_turn_speak_gate_enabled",
    "terminal_hangup_gate_enabled",
)

_attempts: Counter[tuple[str, str]] = Counter()
_marker: dict[str, Any] = {}


def _say(message: str) -> None:
    print(f"scripted user simulator: {message}", file=sys.stderr, flush=True)


def register_task_dirs(tasks_dir: Path = TASKS_DIR) -> list[str]:
    """Register ``tasks_dir/<domain>/`` as extra data for each built-in domain."""
    from talk_bench_domains.data_root import DataRoot
    from talk_bench_domains.environment import register_custom_data

    registered = []
    for domain_dir in sorted(p for p in tasks_dir.iterdir() if (p / "tasks.json").is_file()):
        register_custom_data(
            domain_dir.name,
            DataRoot(
                kind="filesystem",
                location=str(domain_dir),
                mode="domain_direct",
                bound_domain=domain_dir.name,
                tasks_policy="add",
                splits_policy="add",
            ),
        )
        registered.append(domain_dir.name)
    return registered


def _record(runner: Any, scenario_id: str, simulator: Any) -> Path | None:
    out = getattr(runner, "_run_output_dir", None)
    if out is None:
        return None
    out = Path(out)
    key = (str(out), scenario_id)
    _attempts[key] += 1
    _marker.setdefault(
        str(out),
        {
            "engine": "scripted",
            "launcher": str(Path(__file__).resolve()),
            "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scenarios": {},
        },
    )["scenarios"][scenario_id] = simulator.describe()
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER_FILE).write_text(json.dumps(_marker[str(out)], indent=1) + "\n")
    return out / TRACE_DIR / f"{scenario_id}.attempt{_attempts[key]}.ndjson"


def install() -> None:
    """Patch talk-bench's runner in this process."""
    from scripted_user_simulator import SCRIPT_KEY, ScriptedUserSimulator, parse_script
    from talk_bench.runner.runner import BenchmarkRunner

    original_init = BenchmarkRunner.__init__
    original_create = BenchmarkRunner._create_components
    params = inspect.signature(original_create).parameters
    if not {"scenario", "seed", "is_text"} <= set(params):
        _say(
            "this checkout's BenchmarkRunner._create_components no longer takes "
            f"(scenario, seed, is_text) — it takes {list(params)[1:]}; update {Path(__file__).name}"
        )
        sys.exit(2)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        enabled = [f for f in _GATE_FIELDS if getattr(self.config, f, False)]
        if enabled:
            _say(f"turning off {', '.join(enabled)} — they would rewrite the scripted lines")
            self.config = self.config.model_copy(update={f: False for f in enabled})

    async def _create_components(
        self: Any, scenario: Any, seed: int | None = None, *, is_text: bool = False
    ) -> tuple[Any, Any]:
        env, simulator = await original_create(self, scenario, seed, is_text=is_text)
        raw = (scenario.metadata or {}).get(SCRIPT_KEY)
        if raw is None:
            return env, simulator
        script = parse_script(raw)  # ConfigurationError: fails fast, never retried
        scripted = ScriptedUserSimulator(script, scenario_id=scenario.id)
        scripted._trace_path = _record(self, scenario.id, scripted)
        return env, scripted

    BenchmarkRunner.__init__ = __init__  # type: ignore[method-assign]
    BenchmarkRunner._create_components = _create_components  # type: ignore[method-assign]


def main() -> None:
    sys.path.insert(0, str(HERE))
    # importing the CLI loads .env and the built-in domains exactly as `talk-bench` does
    from talk_bench_cli.main import cli

    domains = register_task_dirs()
    install()
    _say(
        f"extra tasks registered for {', '.join(domains) or 'no domain'}; scenarios with "
        "metadata.user_script use the scripted caller"
    )
    cli(prog_name="talk-bench")


if __name__ == "__main__":
    main()
