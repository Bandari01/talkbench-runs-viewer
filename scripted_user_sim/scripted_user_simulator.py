"""A user simulator that never calls an LLM: fixed lines for fixed situations.

The script lives in the task itself, under ``metadata.user_script`` in tasks.json::

    "metadata": {
      "user_script": {
        "rules": [
          {"id": "intent", "when": {"agent_says": "help|assist"},
           "say": "Hi, I need to cancel my reservation."},
          {"id": "uid", "requires": ["intent"], "when": {"agent_says": "user id"},
           "say": "My user id is user_002."},
          {"id": "nudge", "once": false, "when": {"event": "silence_exceeded"},
           "say": "Hello, are you still there?"},
          {"id": "done", "when": {"agent_says": "has been cancel+ed"}, "hangup": "resolved"}
        ],
        "fallback": "Sorry, could you say that again?",
        "max_consecutive_waits": 1
      }
    }

On every wake the rules are tried top to bottom; the first match wins.

- ``when.event``: the wake event kind (default ``agent_utterance_ready``).
- ``when.agent_says``: a case-insensitive regex searched in everything the agent has said
  since the caller last spoke, as the simulator heard it (ASR text in voice runs). ASR
  often splits one agent turn into fragments ("I can help with that." + "May I have your
  user ID?"); matching the whole turn so far keeps a question in the second fragment
  from being missed. Only meaningful for ``agent_utterance_ready``.
- ``requires``: rule ids that must already have fired.
- ``once`` (default true): the rule fires at most once per conversation.
- exactly one action: ``say`` (with optional ``as``: ``speak`` | ``barge_in`` |
  ``re_engage``), ``hangup`` (``resolved`` | ``hangup`` | ``out_of_scope`` |
  ``transferred``) or ``wait`` (keep listening).

When nothing matches:

- an agent utterance first gets up to ``max_consecutive_waits`` waits (the agent may still
  be mid-turn, e.g. "Let me check that"), then the ``fallback`` line;
- a silence wake gets the ``fallback`` line right away (the agent is waiting for us);
- conversation timeout / loop detected / agent session ended hang up (``out_of_scope``);
- anything else (agent speaking for a while, empty utterance, …) keeps listening.

Never waiting indefinitely matters: talk-bench's watchdog judges progress by the
simulator's own turns, and a caller that stays silent while the agent talks is read as a
harness ASR failure (EndReason.ERROR, then the whole scenario is retried).

A line talk-bench drops before speaking it (its two-phase commit: the agent started its
next fragment while the line was being decided) un-fires the rule that produced it. The
line never entered the history, so the agent's turn text still contains what triggered
it, and the rule matches again once the agent finishes, unless a rule higher up now fits
the longer turn better. A dropped fallback line is just forgotten.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from talk_bench_core.exceptions import ConfigurationError
from user_simulator.types.simulator import ActionHandle, EndAction, MessageRole, WakeContext

__all__ = ["SCRIPT_KEY", "ScriptedUserSimulator", "parse_script"]

logger = logging.getLogger("scripted_user_simulator")

SCRIPT_KEY = "user_script"

_SPEAK_KINDS = ("speak", "barge_in", "re_engage")
_END_ACTIONS = tuple(a.value for a in EndAction)
_TERMINAL_EVENTS = frozenset({"conversation_timeout", "loop_detected", "agent_session_ended"})
_KNOWN_EVENTS = frozenset(
    {
        "agent_utterance_ready",
        "agent_utterance_empty",
        "silence_exceeded",
        "agent_speaking_for",
        "user_tool_result",
        "awaiting_reply_timeout",
    }
    | _TERMINAL_EVENTS
)
_RULE_KEYS = frozenset({"id", "when", "requires", "once", "say", "as", "hangup", "wait"})
_WHEN_KEYS = frozenset({"event", "agent_says"})
_SCRIPT_KEYS = frozenset({"rules", "fallback", "max_consecutive_waits"})
_DEFAULT_FALLBACK = "Sorry, could you say that again?"


@dataclass(frozen=True)
class Rule:
    id: str
    event: str
    pattern: re.Pattern[str] | None
    requires: tuple[str, ...]
    once: bool
    action: str  # "say" | "hangup" | "wait"
    kind: str = "speak"  # speak-family kind, for action == "say"
    text: str = ""  # for action == "say"
    end_action: str = ""  # for action == "hangup"


@dataclass(frozen=True)
class Script:
    rules: tuple[Rule, ...]
    fallback: str
    max_consecutive_waits: int


def _bad(where: str, problem: str) -> ConfigurationError:
    return ConfigurationError(f"user_script {where}: {problem}")


def _parse_rule(raw: Any, index: int) -> Rule:
    where = f"rules[{index}]"
    if not isinstance(raw, dict):
        raise _bad(where, "must be an object")
    unknown = set(raw) - _RULE_KEYS
    if unknown:
        raise _bad(where, f"unknown keys {sorted(unknown)}")
    rule_id = raw.get("id")
    if not isinstance(rule_id, str) or not rule_id:
        raise _bad(where, "needs a non-empty string 'id'")
    where = f"rule {rule_id!r}"

    when = raw.get("when", {})
    if not isinstance(when, dict):
        raise _bad(where, "'when' must be an object")
    unknown = set(when) - _WHEN_KEYS
    if unknown:
        raise _bad(where, f"unknown 'when' keys {sorted(unknown)}")
    event = when.get("event", "agent_utterance_ready")
    if event not in _KNOWN_EVENTS:
        raise _bad(where, f"unknown event {event!r}; known: {sorted(_KNOWN_EVENTS)}")
    pattern = None
    if "agent_says" in when:
        if event != "agent_utterance_ready":
            raise _bad(where, "'agent_says' only applies to event agent_utterance_ready")
        try:
            pattern = re.compile(str(when["agent_says"]), re.IGNORECASE)
        except re.error as exc:
            raise _bad(where, f"bad 'agent_says' regex: {exc}") from exc

    requires = raw.get("requires", [])
    if isinstance(requires, str):
        requires = [requires]
    if not isinstance(requires, list) or not all(isinstance(r, str) for r in requires):
        raise _bad(where, "'requires' must be a rule id or a list of rule ids")

    once = raw.get("once", True)
    if not isinstance(once, bool):
        raise _bad(where, "'once' must be true or false")

    actions = [k for k in ("say", "hangup", "wait") if k in raw]
    if len(actions) != 1:
        raise _bad(where, "needs exactly one of 'say', 'hangup', 'wait'")
    action = actions[0]
    if "as" in raw and action != "say":
        raise _bad(where, "'as' only applies to 'say'")

    common = {"id": rule_id, "event": event, "pattern": pattern}
    common |= {"requires": tuple(requires), "once": once, "action": action}
    if action == "say":
        text = raw["say"]
        if not isinstance(text, str) or not text.strip():
            raise _bad(where, "'say' must be a non-empty string")
        kind = raw.get("as", "speak")
        if kind not in _SPEAK_KINDS:
            raise _bad(where, f"'as' must be one of {list(_SPEAK_KINDS)}")
        return Rule(**common, kind=kind, text=text.strip())
    if action == "hangup":
        end_action = raw["hangup"]
        if end_action not in _END_ACTIONS:
            raise _bad(where, f"'hangup' must be one of {list(_END_ACTIONS)}")
        return Rule(**common, end_action=end_action)
    if raw["wait"] is not True:
        raise _bad(where, "'wait' must be true")
    return Rule(**common)


def parse_script(raw: Any) -> Script:
    """Validate a ``metadata.user_script`` object; raises ConfigurationError (no retry)."""
    if not isinstance(raw, dict):
        raise _bad("", "must be an object")
    unknown = set(raw) - _SCRIPT_KEYS
    if unknown:
        raise _bad("", f"unknown keys {sorted(unknown)}")
    raw_rules = raw.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise _bad("", "needs a non-empty 'rules' list")
    rules = tuple(_parse_rule(r, i) for i, r in enumerate(raw_rules))

    ids = [r.id for r in rules]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise _bad("", f"duplicate rule ids {duplicates}")
    for rule in rules:
        missing = [r for r in rule.requires if r not in ids]
        if missing:
            raise _bad(f"rule {rule.id!r}", f"'requires' names unknown rules {missing}")

    fallback = raw.get("fallback", _DEFAULT_FALLBACK)
    if not isinstance(fallback, str) or not fallback.strip():
        raise _bad("", "'fallback' must be a non-empty string")
    max_waits = raw.get("max_consecutive_waits", 1)
    if not isinstance(max_waits, int) or isinstance(max_waits, bool) or max_waits < 0:
        raise _bad("", "'max_consecutive_waits' must be an integer >= 0")
    return Script(rules=rules, fallback=fallback.strip(), max_consecutive_waits=max_waits)


def _agent_turn_text(ctx: WakeContext) -> str:
    """Everything the agent said since the caller's last line, oldest first.

    talk-bench's history stores the agent's utterances under the ``user`` role and the
    simulator's own lines under ``assistant`` (the simulator LLM is the assistant). A
    simulator line only enters the history once its audio has played, so a dropped line
    does not cut the agent's turn short.
    """
    parts = []
    for message in reversed(ctx.conversation_history):
        if message.role == MessageRole.ASSISTANT:
            break
        if message.role == MessageRole.USER and message.content:
            parts.append(message.content)
    text = " ".join(reversed(parts))
    latest = getattr(ctx.event, "text", "") or ""
    if latest and not text.endswith(latest):
        text = f"{text} {latest}".strip()
    return text


async def _one_chunk(text: str) -> AsyncIterator[str]:
    yield text


def _resolved(text: str) -> asyncio.Future[str]:
    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    future.set_result(text)
    return future


class ScriptedUserSimulator:
    """``SimulatorProtocol`` implementation driven only by ``metadata.user_script``."""

    def __init__(
        self,
        script: Any,
        *,
        scenario_id: str = "",
        trace_path: Path | None = None,
    ) -> None:
        self._script = script if isinstance(script, Script) else parse_script(script)
        self._scenario_id = scenario_id
        self._trace_path = trace_path
        self._fired: set[str] = set()
        self._waits = 0
        # (kind, rule id) of the last line handed to talk-bench
        self._last_spoken: tuple[str, str | None] | None = None
        self.decisions: list[dict[str, Any]] = []

    def describe(self) -> dict[str, Any]:
        return {
            "simulator": type(self).__name__,
            "rules": [r.id for r in self._script.rules],
            "fallback": self._script.fallback,
            "max_consecutive_waits": self._script.max_consecutive_waits,
        }

    async def on_wake(self, ctx: WakeContext) -> ActionHandle:
        event = ctx.event.kind
        agent_text = _agent_turn_text(ctx) if event == "agent_utterance_ready" else None
        dropped = ctx.last_action_dropped
        line_dropped = (
            dropped is not None
            and self._last_spoken is not None
            and dropped.action_kind == self._last_spoken[0]
        )
        if line_dropped and self._last_spoken is not None and self._last_spoken[1] is not None:
            # talk-bench's two-phase commit dropped our last line before any audio (the
            # agent started its next fragment). Un-fire its rule: the line never entered
            # the history, so the agent's turn text still holds what triggered it and the
            # rule can match again, unless a rule higher up now fits the longer turn
            # better. A dropped fallback line is simply forgotten.
            self._fired.discard(self._last_spoken[1])

        rule = self._match(event, agent_text or "")
        if rule is not None:
            self._fired.add(rule.id)
            return self._decide(
                event, agent_text, rule.id, *self._from_rule(rule), line_dropped=line_dropped
            )
        return self._decide(
            event, agent_text, None, *self._default(event), line_dropped=line_dropped
        )

    async def classify_barge_in_addressed(
        self, barge_in_text: str, agent_utterances: list[str]
    ) -> bool:
        # The real simulator asks an LLM here; this one has none. Same conservative
        # default it falls back to: the agent addressed the interruption (yielded).
        return True

    # ------------------------------------------------------------------

    def _match(self, event: str, agent_text: str) -> Rule | None:
        for rule in self._script.rules:
            if rule.event != event:
                continue
            if rule.once and rule.id in self._fired:
                continue
            if any(r not in self._fired for r in rule.requires):
                continue
            if rule.pattern is not None and not rule.pattern.search(agent_text):
                continue
            return rule
        return None

    def _from_rule(self, rule: Rule) -> tuple[str, str, str]:
        if rule.action == "say":
            return rule.kind, rule.text, ""
        if rule.action == "hangup":
            return "hangup", "", rule.end_action
        return "listen_more", "", ""

    def _default(self, event: str) -> tuple[str, str, str]:
        if event in _TERMINAL_EVENTS:
            return "hangup", "", EndAction.OUT_OF_SCOPE.value
        if event == "agent_utterance_ready" and self._waits < self._script.max_consecutive_waits:
            self._waits += 1
            return "listen_more", "", ""
        if event in ("agent_utterance_ready", "silence_exceeded"):
            return "speak", self._script.fallback, ""
        return "listen_more", "", ""

    def _decide(
        self,
        event: str,
        agent_text: str | None,
        rule_id: str | None,
        kind: str,
        text: str,
        end: str,
        *,
        line_dropped: bool = False,
    ) -> ActionHandle:
        if kind in _SPEAK_KINDS:
            self._waits = 0
            self._last_spoken = (kind, rule_id)
        decision = {
            "ts": time.time(),
            "event": event,
            "agent_text": agent_text,
            "previous_line_dropped": line_dropped,
            "rule": rule_id or "<default>",
            "action": kind,
            "text": text or None,
            "end_action": end or None,
        }
        self.decisions.append(decision)
        self._write_trace(decision)
        logger.info(
            "[%s] %s%s -> %s %s",
            self._scenario_id,
            event,
            " (previous line dropped)" if line_dropped else "",
            decision["rule"],
            f"{kind}: {text!r}" if text else (f"{kind}: {end}" if end else kind),
        )
        if kind in _SPEAK_KINDS:
            return ActionHandle(kind=kind, text_chunks=_one_chunk(text), completion=_resolved(text))
        if kind == "hangup":
            return ActionHandle(
                kind="hangup", text_chunks=None, completion=_resolved(""), hangup_reason=end
            )
        return ActionHandle(kind="listen_more", text_chunks=None, completion=_resolved(""))

    def _write_trace(self, decision: dict[str, Any]) -> None:
        if self._trace_path is None:
            return
        try:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(decision, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("could not write %s", self._trace_path, exc_info=True)
