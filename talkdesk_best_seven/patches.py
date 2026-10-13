"""In-process patches carrying ai-ds-research PR #473 into an unmodified checkout.

Two independent sets, each installed by a launcher next to this file:

- ``install_judge_reasoning_effort()`` (talk_bench_launch.py) — talk-bench's
  ``evaluation.reasoning_effort``: the evaluation judge gets a reasoning effort
  instead of a temperature;
- ``install_provisioner(cli)`` (talkdesk_agent_launch.py) — talkdesk_agent's
  ``external_tool_count_max``, variable-ID normalization, ``update_agent_system``
  and the ``talkdesk-agent update`` command, plus template lookup in ./templates.

A set whose feature the checkout already has (the PR merged, or its branch checked
out) is skipped, so the launchers keep working after upstream catches up. VERSION.md
records which upstream commit each piece follows.
"""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import hashlib
import inspect
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = HERE / "templates"
# provisioning rewrites the template it patches; it patches a copy here, never ./templates
PATCHED_DIR = HERE / ".patched"

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


# ── helpers ──────────────────────────────────────────────────────────────────


def _add_model_field(model: Any, name: str, annotation: Any, default: Any = None) -> None:
    """Add a field to an already-built pydantic v2 model and rebuild its schema."""
    from pydantic.fields import FieldInfo

    model.__pydantic_fields__[name] = FieldInfo.from_annotated_attribute(annotation, default)
    model.model_rebuild(force=True)


def _rebuild_container(model: Any, patched: type) -> None:
    """Rebuild a model that embeds ``patched``; a default instance of ``patched`` made
    before the new field existed lacks it, so it is replaced by a fresh one."""
    for field in model.__pydantic_fields__.values():
        if isinstance(field.default, patched):
            field.default = patched(**field.default.__dict__)
    model.model_rebuild(force=True)


def _add_dataclass_field(cls: type, name: str, annotation: Any, field: Any) -> None:
    """Add a trailing field (with a default) to an existing, non-frozen dataclass
    that relies on the generated ``__init__`` / ``__repr__`` / ``__eq__``."""
    # dataclass() reads defaults back from class attributes; fields with a
    # default_factory have none, so hand every field its own Field object again
    for existing in dataclasses.fields(cls):
        setattr(cls, existing.name, existing)
    cls.__annotations__[name] = annotation
    setattr(cls, name, field)
    # dataclass() only adds the methods a class does not define itself
    for method in ("__init__", "__repr__", "__eq__"):
        if method in cls.__dict__:
            delattr(cls, method)
    dataclasses.dataclass(cls)


def _rebind(original: Any, replacement: Any) -> None:
    """Point every loaded talk-bench / talkdesk module that imported ``original`` by
    name at ``replacement``."""
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith(("talk_bench", "talkdesk_")):
            continue
        namespace = vars(module)
        for attr, value in list(namespace.items()):
            if value is original:
                namespace[attr] = replacement


def _say(message: str) -> None:
    print(f"talkdesk_best_seven: {message}", file=sys.stderr, flush=True)


# ── talk-bench: evaluation.reasoning_effort ──────────────────────────────────

# effort of this process's BenchmarkRunner: the default of BenchmarkConfig.judge_reasoning_effort
_run_effort: str | None = None
# effort for the judge calls made inside one of the scoped evaluation functions
_judge_effort: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "talkdesk_best_seven_judge_effort", default=None
)


def _effort_scope(func: Callable[..., Any]) -> Callable[..., Any]:
    """Run an async evaluation function with its ``config``'s judge effort in scope —
    stands in for passing ``reasoning_effort=config.judge_reasoning_effort`` to each
    ``call_judge_async`` inside it, as the PR does."""
    signature = inspect.signature(func)

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        config = signature.bind_partial(*args, **kwargs).arguments.get("config")
        token = _judge_effort.set(getattr(config, "judge_reasoning_effort", None))
        try:
            return await func(*args, **kwargs)
        finally:
            _judge_effort.reset(token)

    return wrapper


def install_judge_reasoning_effort() -> bool:
    """Patch talk-bench so ``evaluation.reasoning_effort`` reaches the judge.

    Returns False when the checkout already supports it."""
    from talk_bench.types.eval_config import EvaluationConfig

    if "reasoning_effort" in EvaluationConfig.model_fields:
        return False

    import talk_bench.types.runner as runner_types
    import talk_bench_core.project_config as project_config
    import talk_bench_evaluation.entrypoint  # noqa: F401 — binds the scoped functions by name
    from talk_bench.runner.runner import BenchmarkRunner
    from talk_bench_evaluation import judges, score_scenario
    from talk_bench_evaluation.config import BenchmarkConfig
    from talk_bench_evaluation.judges import cache as judge_cache
    from talk_bench_evaluation.voice import conversation_quality, customer_effort

    # config: talkbench.yaml `evaluation:` block → EvaluationConfig → BenchmarkConfig
    _add_model_field(project_config.EvaluationBlock, "reasoning_effort", ReasoningEffort | None)
    _rebuild_container(project_config.ProjectConfig, project_config.EvaluationBlock)
    _add_model_field(EvaluationConfig, "reasoning_effort", ReasoningEffort | None)
    _rebuild_container(runner_types.RunConfig, EvaluationConfig)
    _add_dataclass_field(
        BenchmarkConfig,
        "judge_reasoning_effort",
        "str | None",
        dataclasses.field(default_factory=lambda: _run_effort),
    )

    original_init = BenchmarkRunner.__init__

    @functools.wraps(original_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        global _run_effort
        original_init(self, *args, **kwargs)
        # both BenchmarkConfigs the runner builds (scoring, rejected-gate diagnostics)
        _run_effort = getattr(self.config.evaluation_config, "reasoning_effort", None)

    BenchmarkRunner.__init__ = __init__  # type: ignore[method-assign]

    # judge client: effort instead of temperature, effort in the cache key and the log
    original_build = judges._build_completion_args
    original_entry = judges._judge_request_entry

    def _build_completion_args(
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        response_format: dict[str, Any] | None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        reasoning_effort = reasoning_effort or _judge_effort.get()
        if reasoning_effort is None:
            return original_build(model, messages, temperature, response_format)
        completion_args: dict[str, Any] = {"model": model, "messages": messages}
        completion_args["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            completion_args["response_format"] = response_format
        return completion_args

    def cache_key(
        system_prompt: str,
        user_content: str,
        model: str,
        temperature: float,
        reasoning_effort: str | None = None,
    ) -> str:
        reasoning_effort = reasoning_effort or _judge_effort.get()
        payload = json.dumps([system_prompt, user_content, model, temperature, reasoning_effort])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _judge_request_entry(**kwargs: Any) -> tuple[list[dict[str, str]], dict[str, Any]]:
        messages, entry = original_entry(**kwargs)
        # same key order as the PR: right after "temperature"
        items = list(entry.items())
        at = [key for key, _ in items].index("temperature") + 1
        items.insert(at, ("reasoning_effort", _judge_effort.get()))
        return messages, dict(items)

    _rebind(judge_cache.cache_key, cache_key)
    judge_cache.cache_key = cache_key
    judges._build_completion_args = _build_completion_args
    judges._judge_request_entry = _judge_request_entry

    # the judges the PR hands the effort to: task completion + db semantic,
    # conversation quality, customer effort (UER and simulator-gate judges keep temperature)
    for module, name in (
        (score_scenario, "evaluate_task_completion"),
        (conversation_quality, "compute_conversation_quality"),
        (customer_effort, "_call_judge"),
    ):
        original = getattr(module, name)
        scoped = _effort_scope(original)
        _rebind(original, scoped)
        setattr(module, name, scoped)
    return True


# ── talkdesk_agent: provisioning ─────────────────────────────────────────────

_DEFAULT_MAX_EXTERNAL_TOOL_COUNT = 5
# external_tool_count_max of the provisioning call in progress
_tool_limit: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "talkdesk_best_seven_tool_limit", default=None
)


def _validate_template_limits(
    template: dict, *, external_tool_count_max: int | None = None
) -> None:
    """Enforce DesignTimeLimits.MAX_EXTERNAL_TOOL_COUNT / MAX_CHILD_AGENT_COUNT."""
    from talkdesk_agent.provisioner import _MAX_CHILD_AGENT_COUNT

    if external_tool_count_max is None:
        external_tool_count_max = _tool_limit.get()
    violations: list[str] = []
    external_tool_limit = (
        external_tool_count_max
        if external_tool_count_max is not None
        else _DEFAULT_MAX_EXTERNAL_TOOL_COUNT
    )
    if external_tool_limit < 1:
        raise ValueError("external_tool_count_max must be at least 1")

    for agent in template.get("agents", []):
        if not isinstance(agent, dict):
            continue
        name = agent.get("agent_name") or agent.get("agent_id") or "<unnamed agent>"
        tool_count = len(agent.get("tools", []) or [])
        if tool_count > external_tool_limit:
            violations.append(
                f"{name}: has {tool_count} tools (more than {external_tool_limit} tools allowed)"
            )

    children_per_source: dict[str, int] = {}
    for link in template.get("links", []) or []:
        if not isinstance(link, dict):
            continue
        source = link.get("source_agent_id")
        if source:
            children_per_source[source] = children_per_source.get(source, 0) + 1

    for source, count in children_per_source.items():
        if count > _MAX_CHILD_AGENT_COUNT:
            violations.append(
                f"agent '{source}': has {count} child agents "
                f"(more than {_MAX_CHILD_AGENT_COUNT} child agents allowed)"
            )

    if violations:
        raise ValueError(
            "Template validation failed: limits exceeded:\n  - " + "\n  - ".join(violations)
        )


def _normalize_variable_ids(template: dict, agent_system_id: str) -> bool:
    """Variable IDs are different from agent/tool IDs: the API deserializes them
    directly as java.util.UUID values. Preserve valid authored UUIDs, but
    deterministically normalize missing or human-readable template IDs so repeated
    updates of the same system remain stable. Returns whether anything changed."""
    changed = False
    for variable_name, variable in (template.get("variables") or {}).items():
        if not isinstance(variable, dict):
            continue
        raw_variable_id = variable.get("id")
        try:
            uuid.UUID(str(raw_variable_id))
        except (ValueError, TypeError, AttributeError):
            variable["id"] = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"talkdesk-agent-system:{agent_system_id}:variable:{variable_name}",
                )
            )
            changed = True
    return changed


def update_agent_system(
    config: Any,
    *,
    agent_system_id: str,
    mcp_server_id: str,
    template_path: Path | None = None,
    on_status: Callable[..., object] | None = None,
) -> Any:
    """Update an existing Talkdesk agent system without creating new records.

    The caller must supply both hosted identifiers explicitly. This function
    performs one authenticated agent-system PUT and never calls either create
    endpoint, making it suitable for revising a pre-provisioned DRAFT system
    while preserving its stable agent-system UUID.
    """
    import asyncio

    import requests
    from talkdesk_agent import provisioner

    if not agent_system_id or agent_system_id == "PLACEHOLDER":
        raise ValueError("agent_system_id must identify an existing agent system")
    if not mcp_server_id or mcp_server_id == "PLACEHOLDER":
        raise ValueError("mcp_server_id must identify an existing MCP server")
    if template_path is None:
        template_path = provisioner._TEMPLATES_DIR / config.template_name

    def _status(msg: str) -> None:
        if on_status:
            on_status(msg)
        else:
            provisioner.logger.info(msg)

    oauth_resolved = Path(config.oauth_path)
    if not oauth_resolved.is_absolute():
        oauth_resolved = Path.cwd() / config.oauth_path
    _status(f"Fetching OAuth token from {oauth_resolved}...")
    token = asyncio.run(
        provisioner.get_access_token(oauth_path=oauth_resolved, token_url=config.token_url)
    )
    auth_headers = {"Authorization": f"Bearer {token}"}

    _status(f"Preparing in-place update for agent system {agent_system_id}...")
    template = provisioner._patch_template(
        template_path,
        account_id=config.account_id,
        agent_system_id=agent_system_id,
        agent_system_name=config.agent_name,
        mcp_server_id=mcp_server_id,
        mcp_server_url=config.mcp_url,
        strict=True,
        supervisor_instruction_max_chars=config.supervisor_instruction_max_chars,
        agent_instruction_max_chars=config.agent_instruction_max_chars,
        function_max_chars=config.function_max_chars,
        memory_instruction_max_chars=config.memory_instruction_max_chars,
        external_tool_count_max=config.external_tool_count_max,
    )

    agent_system_endpoint = f"{config.api_base_url}/multi-agent/manage/agent-system"
    _status("Deploying agent system config in place...")
    resp = requests.put(agent_system_endpoint, json=template, headers=auth_headers)
    if resp.status_code >= 400:
        _status(f"In-place update failed ({resp.status_code}): {resp.text[:2000]}")
    provisioner._raise_for_status_with_body(resp)
    _status("Agent system config updated successfully; UUID preserved.")

    return provisioner.ProvisionResult(
        agent_name=config.agent_name,
        agent_system_id=agent_system_id,
        mcp_server_id=mcp_server_id,
    )


def _patch_provisioner_features() -> None:
    """external_tool_count_max + variable-ID normalization on a pre-PR provisioner."""
    from talkdesk_agent import provisioner

    ProvisionConfig = provisioner.ProvisionConfig
    _add_dataclass_field(ProvisionConfig, "external_tool_count_max", "int | None", None)

    original_from_yaml = ProvisionConfig.from_yaml_and_env.__func__

    def from_yaml_and_env(cls: type, yaml_path: Path, env: dict[str, str]) -> Any:
        import yaml

        config = original_from_yaml(cls, yaml_path=yaml_path, env=env)
        with open(yaml_path) as f:
            config.external_tool_count_max = (yaml.safe_load(f) or {}).get(
                "external_tool_count_max"
            )
        return config

    ProvisionConfig.from_yaml_and_env = classmethod(from_yaml_and_env)  # type: ignore[method-assign]

    original_patch = provisioner._patch_template
    signature = inspect.signature(original_patch)

    @functools.wraps(original_patch)
    def _patch_template(
        *args: Any, external_tool_count_max: int | None = None, **kwargs: Any
    ) -> dict:
        token = _tool_limit.set(
            external_tool_count_max if external_tool_count_max is not None else _tool_limit.get()
        )
        try:
            template = original_patch(*args, **kwargs)
        finally:
            _tool_limit.reset(token)
        bound = signature.bind(*args, **kwargs).arguments
        # nothing else in _patch_template reads a variable's "id", so normalizing
        # afterwards gives the PR's result; rewrite the file as _patch_template does
        if _normalize_variable_ids(template, bound["agent_system_id"]):
            with open(bound["template_path"], "w") as f:
                json.dump(template, f, indent=2)
                f.write("\n")
        return template

    provisioner._patch_template = _patch_template
    provisioner._validate_template_limits = _validate_template_limits


def _working_copy(template_name: str) -> Path:
    """Copy of ``template_name`` (./templates first, then the checkout's templates)
    that provisioning may rewrite."""
    from talkdesk_agent import provisioner

    source = TEMPLATES_DIR / template_name
    if not source.is_file():
        source = provisioner._TEMPLATES_DIR / template_name
    work = PATCHED_DIR / template_name
    work.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, work)
    return work


def _with_template(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(config: Any, *args: Any, template_path: Path | None = None, **kwargs: Any) -> Any:
        if template_path is None:
            template_path = _working_copy(config.template_name)
        # generate_template / provision of a pre-PR checkout don't pass the limit on
        token = _tool_limit.set(getattr(config, "external_tool_count_max", None))
        try:
            return func(config, *args, template_path=template_path, **kwargs)
        finally:
            _tool_limit.reset(token)

    return wrapper


def _update_command(cli: Any) -> None:
    """talkdesk-agent update — update a pre-provisioned Talkdesk agent system."""
    import click
    from talkdesk_agent import provisioner

    @cli.command("update")
    @click.option(
        "--config",
        "config_path",
        default="talkdesk_agent.yaml",
        type=click.Path(exists=True),
        help="Path to provisioning configuration YAML file.",
    )
    @click.option(
        "--agent-system-id",
        required=True,
        help="Existing Talkdesk agent-system UUID to update in place.",
    )
    @click.option(
        "--mcp-server-id",
        required=True,
        help="Existing Talkdesk MCP-server ID referenced by the template tools.",
    )
    def update_cmd(config_path: str, agent_system_id: str, mcp_server_id: str) -> None:
        """Update an existing agent system without creating another system."""
        try:
            config = provisioner.ProvisionConfig.from_yaml_and_env(
                yaml_path=Path(config_path),
                env=dict(os.environ),
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        try:
            result = provisioner.update_agent_system(
                config,
                agent_system_id=agent_system_id,
                mcp_server_id=mcp_server_id,
                on_status=click.echo,
            )
        except Exception as exc:
            raise click.ClickException(f"In-place update failed: {exc}") from exc

        click.echo()
        click.echo("Updated agent in place:")
        click.echo(f"  Agent Name:       {result.agent_name}")
        click.echo(f"  Agent System ID:  {result.agent_system_id}")
        click.echo(f"  MCP Server ID:    {result.mcp_server_id}")


def install_provisioner(cli: Any) -> list[str]:
    """Patch talkdesk_agent provisioning and add ``update`` to ``cli`` (the
    talkdesk-agent click group, already imported). Returns what was patched."""
    from talkdesk_agent import provisioner

    patched = []
    if "external_tool_count_max" not in provisioner.ProvisionConfig.__dataclass_fields__:
        _patch_provisioner_features()
        patched.append("external_tool_count_max + variable-ID normalization")
    if not hasattr(provisioner, "update_agent_system"):
        provisioner.update_agent_system = update_agent_system
        patched.append("update_agent_system")
    if "update" not in cli.commands:
        _update_command(cli)
        patched.append("`update` command")

    for name in ("generate_template", "provision", "update_agent_system"):
        original = getattr(provisioner, name)
        wrapped = _with_template(original)
        _rebind(original, wrapped)
        setattr(provisioner, name, wrapped)
    return patched
