# talkdesk_best_seven

The seven retained Talkdesk text-agent checkpoints from
[ai-ds-research#473](https://github.com/Talkdesk/ai-ds-research/pull/473) (cedarway_specialty_pharmacy,
everfield_utilities, griddlepoint, harborstone_credit_union, meridian, northline_insurance,
pinehaven_childrens_health) and the talk-bench / talkdesk-agent changes they need, usable from an
`ai-ds-research` checkout that does not have the PR. Nothing in the checkout is modified: the
launchers patch it in their own process, and skip any piece the checkout already has. Upstream
commits, scores and the change log are in [VERSION.md](VERSION.md).

Run everything from `<ai-ds-research>/talk-bench-talkdesk` (its venv has `talkdesk_agent` and the
MCP server, its `.env` holds `TALKDESK_ACCOUNT_ID` / `TALKDESK_OAUTH_PATH` / `TALKDESK_TOKEN_URL`).
`<viewer>` below is this repository.

## 1. MCP host for the seven domains

```bash
uv run python <viewer>/talkdesk_best_seven/seven_domain_server.py --port 3030 --ngrok
```

One process, one tunnel; each domain lives at `<base>/<domain>/` with its own conversation
registry (`GET /health` lists them).

## 2. Provision or update an agent system

```bash
TALKDESK_MCP_URL=<base>/meridian/ uv run python <viewer>/talkdesk_best_seven/talkdesk_agent_launch.py \
    provision --config <viewer>/talkdesk_best_seven/provision/meridian_text_best.yaml
```

`generate` prints the patched JSON without calling Talkdesk; `provision` creates an MCP server and
an agent system on QA (`api_base_url` in the YAML) and deploys the template; `update
--agent-system-id <uuid> --mcp-server-id <id>` re-deploys onto an existing system with a single PUT.
The provisioning YAMLs set `external_tool_count_max: 30` (the checkpoints have up to 30 tools per
agent; talkdesk-agent's default check allows 5). `template_name` resolves in `templates/` here
first; the patched copy that was sent goes to `.patched/` (gitignored).

## 3. Run the benchmark with the PR's judge profile

```bash
uv run python <viewer>/talkdesk_best_seven/talk_bench_launch.py run --config <talkdesk agent yaml> \
    --project-config <viewer>/talkdesk_best_seven/seven_domain_talkdesk_project.yaml \
    --source talk_bench --domain meridian
```

`seven_domain_talkdesk_project.yaml` is the PR's profile: `azure/gpt-5.6-luna` simulates the caller,
`azure/gpt-5.6-sol` judges with `evaluation.reasoning_effort: xhigh`, experience and audio-MOS
metrics off. The stock `talk-bench` rejects `reasoning_effort` (unknown field) on a checkout without
the PR, and so does the viewer's **Run test** page, which starts the stock CLI.
