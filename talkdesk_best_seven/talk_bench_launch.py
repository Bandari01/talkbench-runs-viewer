"""`talk-bench` with the judge ``reasoning_effort`` of ai-ds-research PR #473.

    cd <ai-ds-research>/talk-bench-talkdesk        # or talk-bench
    uv run python <viewer>/talkdesk_best_seven/talk_bench_launch.py run --config agent.yaml \
        --project-config <viewer>/talkdesk_best_seven/seven_domain_talkdesk_project.yaml \
        --source talk_bench --domain meridian

Same arguments, output, run directory and exit codes as ``uv run talk-bench …``. In
this process only (nothing in the checkout changes), ``evaluation.reasoning_effort``
(none | minimal | low | medium | high | xhigh) is accepted in talkbench.yaml, the agent
YAML or a --simulator-config file, and when set the task-completion, db-semantic,
conversation-quality and customer-effort judges send it instead of a temperature
(reasoning deployments reject the two together). It is recorded in run_config.json,
the judge cache key and judge_requests. A checkout that already has the feature runs
unpatched.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # importing the CLI loads .env exactly as the `talk-bench` entry point does
    from talk_bench_cli.main import cli

    from patches import _say, install_judge_reasoning_effort

    if install_judge_reasoning_effort():
        _say("evaluation.reasoning_effort enabled (patched in this process)")
    else:
        _say("this checkout already supports evaluation.reasoning_effort; running unpatched")
    cli(prog_name="talk-bench")


if __name__ == "__main__":
    main()
