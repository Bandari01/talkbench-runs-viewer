"""`talkdesk-agent` with the provisioning changes of ai-ds-research PR #473.

    cd <ai-ds-research>/talk-bench-talkdesk
    uv run python <viewer>/talkdesk_best_seven/talkdesk_agent_launch.py generate \
        --config <viewer>/talkdesk_best_seven/provision/meridian_text_best.yaml
    … provision --config <same>
    … update --config <same> --agent-system-id <uuid> --mcp-server-id <id>

Same commands and output as ``uv run talkdesk-agent …``, plus, in this process only:

- ``external_tool_count_max`` in the provisioning YAML raises the 5-tools-per-agent
  check (the seven checkpoints use 30);
- variable IDs that are not UUIDs get a stable uuid5 per agent system;
- ``update`` re-deploys a template onto an existing agent system (one PUT, nothing
  created);
- ``template_name`` is looked up in ./templates first, then in the checkout's
  talkdesk_agent/templates. The patched JSON (IDs, account, MCP URL filled in) is
  written to ./.patched/<template_name>; the source template is never rewritten.

Pieces the checkout already has (the PR merged) run unpatched.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # importing the CLI loads talk-bench-talkdesk/.env and registers the stock commands
    from talkdesk_agent_cli.main import cli

    from patches import _say, install_provisioner

    patched = install_provisioner(cli)
    _say(
        f"patched {', '.join(patched)}"
        if patched
        else "this checkout already has PR #473's provisioning"
    )
    cli(prog_name="talkdesk-agent")


if __name__ == "__main__":
    main()
