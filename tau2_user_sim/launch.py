"""`talk-bench` with tau2-bench's native user simulator in voice runs.

    cd <ai-ds-research>/talk-bench        # or talk-bench-talkdesk for talkdesk_agent.*
    uv run python <viewer>/tau2_user_sim/launch.py run --config agent.yaml --domain airline --source tau2

Same arguments, output, run directory and exit codes as ``uv run talk-bench …``. The
one difference: every voice scenario's user simulator is
``Tau2NativeVoiceUserSimulator`` (see tau2_native_user_simulator.py) instead of
talk-bench's ``EventDrivenSimulator``. Text runs are left alone — on tau2 tasks
talk-bench already uses tau2's own simulator for them. Nothing in the checkout
changes: ``BenchmarkRunner`` is patched in this process only.

Also, in this process:
- talk-bench's simulator / mid-turn-speak / terminal-hangup gates are turned off
  if talkbench.yaml enables them — they would override tau2's own decisions;
- the run directory gets ``user_simulator.json`` recording which simulator and
  models made the run (the viewer tags those runs);
- ``TAU2_USER_SIM_DECISION_LLM`` picks the model for tau2's interruption decision
  (default: the simulator's ``llm_model`` from talkbench.yaml).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ENGINE = "tau2"
MARKER_FILE = "user_simulator.json"
# talk-bench's own LLM gates, which would second-guess the simulator's turns
_GATE_FIELDS = ("simulator_gate_enabled", "mid_turn_speak_gate_enabled", "terminal_hangup_gate_enabled")

logger = logging.getLogger("tau2_native_user_simulator")
_markers_written: set[str] = set()


def _say(message: str) -> None:
    print(f"tau2 user simulator: {message}", file=sys.stderr, flush=True)


def _tau2_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("tau2")
    except PackageNotFoundError:
        return None


def _write_marker(runner: Any, info: dict[str, Any]) -> None:
    out = getattr(runner, "_run_output_dir", None)
    if out is None or str(out) in _markers_written:
        return
    _markers_written.add(str(out))
    marker = {
        "engine": ENGINE,
        **info,
        "speech_complexity": runner.config.speech_complexity,
        "guidelines": "data/tau2/user_simulator/simulation_guidelines_voice.md",
        "tau2_version": _tau2_version(),
        "launcher": str(Path(__file__).resolve()),
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = Path(out) / MARKER_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker, indent=1) + "\n")


def _warn_user_tools(env: Any, scenario_id: str) -> None:
    from talk_bench_core.protocols import UserToolCapableEnvironment

    if isinstance(env, UserToolCapableEnvironment) and list(env.user_tools.definitions):
        logger.warning(
            "[%s] this task gives the caller user-side tools; the tau2 user simulator runs "
            "without them (like talk-bench's text-channel Tau2FaithfulUserSimulator), so steps "
            "the caller must do on their own device cannot happen",
            scenario_id,
        )


def install() -> None:
    """Patch talk-bench's runner so voice scenarios get the tau2 user simulator."""
    from talk_bench.runner.runner import BenchmarkRunner
    from tau2_native_user_simulator import Tau2NativeVoiceUserSimulator, tau2_persona_config

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
            _say(f"turning off {', '.join(enabled)} — those gates would override tau2's decisions")
            self.config = self.config.model_copy(update={f: False for f in enabled})

    async def _create_components(
        self: Any, scenario: Any, seed: int | None = None, *, is_text: bool = False
    ) -> tuple[Any, Any]:
        env, simulator = await original_create(self, scenario, seed, is_text=is_text)
        if is_text:
            return env, simulator
        llm = self.config.llm_config
        simulator = Tau2NativeVoiceUserSimulator(
            scenario,
            llm=llm.model,
            llm_args={"temperature": llm.temperature},
            seed=seed,
            decision_llm=os.environ.get("TAU2_USER_SIM_DECISION_LLM") or None,
            persona_config=tau2_persona_config(scenario, self.config.speech_complexity),
        )
        _warn_user_tools(env, scenario.id)
        _write_marker(self, simulator.describe())
        return env, simulator

    BenchmarkRunner.__init__ = __init__  # type: ignore[method-assign]
    BenchmarkRunner._create_components = _create_components  # type: ignore[method-assign]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # importing the CLI loads .env exactly as the `talk-bench` entry point does, before
    # anything imports tau2 (which reads TAU2_DATA_DIR at import time)
    from talk_bench_cli.main import cli

    install()
    _say("voice scenarios use tau2-bench's native user simulator (text runs are unchanged)")
    cli(prog_name="talk-bench")


if __name__ == "__main__":
    main()
