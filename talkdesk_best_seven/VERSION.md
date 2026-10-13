# talkdesk_best_seven — version

**Current: 1.0.0** (2026-10-12)

Port of [Talkdesk/ai-ds-research#473](https://github.com/Talkdesk/ai-ds-research/pull/473)
"Add seven retained Talkdesk checkpoints and TalkBench integration" (open when ported).

| | commit |
|---|---|
| PR head ported | `35ec94f72445b17fa5c228e76c6a1877b95d34ba` |
| PR base (main) | `4fee53d5190b8b6621bf03a0a84f789b17094569` |
| Verified against | `4fee53d5` sources (patched) and the PR branch (unpatched); origin/main `441c51c9` has no talk-bench / talk-bench-talkdesk changes since the base |

Every change to this folder gets a changelog entry below and a version bump: patch for
fixes, minor for re-syncing with a newer upstream commit or adding data, major when a
launcher's command line changes.

## Where each piece comes from

Paths on the left are relative to this folder; on the right, to `ai-ds-research/`.

| here | upstream | how |
|---|---|---|
| `templates/retained-best/*.json` (7) | `talk-bench-talkdesk/libs/talkdesk_agent/templates/retained-best/` | byte-identical |
| `provision/*_text_best.yaml` (7) | `talk-bench-talkdesk/samples/provision/` | byte-identical |
| `seven_domain_talkdesk_project.yaml` | `talk-bench-talkdesk/samples/seven_domain_talkdesk_project.yaml` | byte-identical (PR version) |
| `seven_domain_server.py` | `talk-bench-talkdesk/apps/talkdesk_mcp_server/src/talkdesk_mcp_server/seven_domain_server.py` | verbatim + usage in the docstring |
| `patches.py` `install_judge_reasoning_effort` | `talk_bench_core/project_config.py`, `talk_bench/types/eval_config.py`, `talk_bench/runner/runner.py`, `talk_bench_evaluation/{config.py, judges/__init__.py, judges/cache.py, score_scenario.py, voice/conversation_quality.py, voice/customer_effort.py}` | patched in-process by `talk_bench_launch.py` |
| `patches.py` `install_provisioner` | `talkdesk_agent/provisioner.py`, `talkdesk_agent_cli/{main.py, update_cmd.py}` | patched in-process by `talkdesk_agent_launch.py` |

Left out: every test file (12), `talk-bench-talkdesk/scripts/seven_domain_talkdesk_best_results.json`
(provenance only — summarized below) and the `talkdesk-mcp-server-seven` script entry in
`apps/talkdesk_mcp_server/pyproject.toml` (run `seven_domain_server.py` instead).

Differences from the PR, by design:

- `BenchmarkConfig.judge_reasoning_effort` is the last field (the PR puts it after
  `judge_temperature`); only positional construction would notice.
- The judge effort is scoped around the four PR call sites (task completion, db semantic,
  conversation quality, customer effort) instead of passed as a `call_judge_async` argument;
  the requests, judge log entries and cache keys are identical (checked against the PR).
- Provisioning patches a copy in `.patched/` (gitignored) instead of rewriting the template
  in place, so account / agent-system IDs and the MCP URL never land in this repo.

## Checkpoints

From the PR's `seven_domain_talkdesk_best_results.json`: best retained valid whole-run STANDARD
essential-only score per domain (rubric `essential-only-communication-v1`), historical sandbox
runs, not re-run on this revision. Templates carry `PLACEHOLDER` for the agent-system ID, MCP
server ID and MCP URL; audit fields (`created_by`, `updated_by`, `created_utc`, `updated_utc`)
are removed.

| domain | version | run | passes | agents | max tools / agent | template sha256 |
|---|---|---|---|---|---|---|
| cedarway_specialty_pharmacy | Round 2 | run-1791081674-7f0b0faf | 13/70 | 11 | 27 | `67210ea3e741…aa6d` |
| everfield_utilities | Round 2 | run-1791136664-2f98b271 | 9/70 | 11 | 28 | `f822446cece2…ab9a` |
| griddlepoint | Round 2 | run-1791082158-004a6431 | 11/70 | 8 | 15 | `d3ee022e8132…8d62` |
| harborstone_credit_union | Baseline | run-1790974138-9928f268 | 10/70 | 6 | 30 | `ab881a91f732…f7f3` |
| meridian | Baseline | run-1790992000-55e6d7b0 | 14/70 | 11 | 20 | `69c68b54873b…93cf` |
| northline_insurance | Round 1 | run-1791037074-b65c9d56 | 7/70 | 7 | 30 | `12e31bd9b7f9…2690` |
| pinehaven_childrens_health | Baseline | run-1790975615-54d7df23 | 10/70 | 8 | 7 | `b234ef0b7ac3…f5fa` |

`shasum -a 256 templates/retained-best/*.json` must still match these after any change that
is not a deliberate template update (log those below).

## Re-syncing with upstream

```bash
git -C ~/GitHub/ai-ds-research fetch origin
git -C ~/GitHub/ai-ds-research diff 35ec94f7 origin/main -- talk-bench talk-bench-talkdesk --stat
```

Copy changed data files as-is, port changed code into `patches.py` (or drop a patch once
upstream has it — each install function already skips itself then), update the commit table
above and add a changelog entry.

## Changelog

### 1.0.0 — 2026-10-12

- Initial port of PR #473 at `35ec94f7`: seven retained text-agent templates and their
  provisioning YAMLs, the seven-domain project profile (Luna simulator, Sol judge with
  `reasoning_effort: xhigh`), the seven-domain MCP host, judge `reasoning_effort`,
  `external_tool_count_max`, variable-ID normalization and `talkdesk-agent update`.
