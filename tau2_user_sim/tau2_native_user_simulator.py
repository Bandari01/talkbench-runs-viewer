"""tau2-bench's native voice user simulator, driving a talk-bench voice run.

Imported by ``launch.py`` inside the talk-bench (or talk-bench-talkdesk) venv: it
needs talk-bench and the vendored tau2-bench, never the viewer's own dependencies.
tau2 is imported lazily so it reads ``TAU2_DATA_DIR`` after the CLI loaded ``.env``.

What the caller says and when it ends the call is tau2's, taken from the vendored
package at runtime rather than copied:

- system prompt: tau2's voice guidelines (``simulation_guidelines_voice.md``), the
  ``PersonaConfig`` guidelines and the task's ``<scenario>`` block, assembled by
  tau2's own ``UserSimulator.system_prompt`` (identical to
  ``VoiceStreamingUserSimulator``'s);
- every reply: ``tau2.utils.llm_utils.generate`` over the role-flipped history,
  silence annotations and tau2's trailing CUSTOMER reminder, as in
  ``VoiceStreamingUserSimulator._generate_full_duplex_voice_message``;
- the end tokens ###STOP### / ###TRANSFER### / ###OUT-OF-SCOPE###;
- interruptions: tau2's ``INTERRUPTION_DECISION_PROMPT`` YES/NO policy, for
  personas with ``interrupt_tendency=interrupts`` (tau2 ``--speech-complexity regular``).

TTS/ASR, audio effects, the agent connection, scoring and *when* the simulator
is woken stay talk-bench's — tau2's 200 ms tick loop is not reproduced. Its
turn-taking maps onto talk-bench's wake events:

    tau2 full-duplex                          talk-bench wake → action
    reply once the agent stops (1 s)          agent_utterance_ready → speak
    check in after mutual silence (5 s)       silence_exceeded / awaiting_reply_timeout → re_engage
    interrupt when the policy says YES        agent_speaking_for → barge_in, else listen_more
    backchannel ("uh-huh", "mm-hmm")          not reproduced: talk-bench has no backchannel action

Like talk-bench's text-channel ``Tau2FaithfulUserSimulator``, the user gets no
user-side tools.
"""

from __future__ import annotations

import ast
import asyncio
import functools
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from user_simulator.exceptions import ProviderError, SimulatorOutputError
from user_simulator.types.simulator import ActionHandle, EndAction, MessageRole, WakeContext

if TYPE_CHECKING:
    from talk_bench_core.scenario import Scenario
    from tau2.data_model.persona import PersonaConfig

__all__ = ["Tau2NativeVoiceUserSimulator", "scenario_instructions", "tau2_persona_config"]

logger = logging.getLogger(__name__)

# Inline in tau2's VoiceStreamingUserSimulator._generate_full_duplex_voice_message,
# appended after the history on every reply — copied verbatim.
ROLE_REMINDER = (
    "REMINDER: You are the CUSTOMER calling for help. Respond as the customer would - "
    "with questions, requests, or information about your issue. Do NOT respond as the "
    "customer service agent."
)
# tau2.agent.base.streaming's annotation for a stretch where both parties are silent.
SILENCE_ANNOTATION = "[Both parties silent for {seconds:.1f} seconds]"
# What tau2's linearization appends to the agent's still-in-progress speech.
INCOMPLETE_SUFFIX = " [CURRENTLY SPEAKING, INCOMPLETE]"
# tau2 turns the guidelines' [pause] markers into "..." before any non-v3 TTS
# (tau2.voice.utils.elevenlabs_utils.PAUSE_TAG_PATTERN); talk-bench's TTS is never v3.
_PAUSE_TAG = re.compile(r"\[pause\]", re.IGNORECASE)

_END_TOKEN_ACTIONS = (
    ("###STOP###", EndAction.RESOLVED),
    ("###TRANSFER###", EndAction.TRANSFERRED),
    ("###OUT-OF-SCOPE###", EndAction.OUT_OF_SCOPE),
)
_ABORT_EVENT_KINDS = frozenset({"conversation_timeout", "loop_detected", "agent_session_ended"})
# litellm status codes worth a scenario retry; everything else (auth, bad request) is permanent
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


@functools.lru_cache(maxsize=1)
def interruption_decision_prompt() -> str:
    """tau2's INTERRUPTION_DECISION_PROMPT, read from the vendored source: the module
    defining it needs tau2's [voice] extra (pyaudio), which talk-bench does not install."""
    import tau2

    path = Path(tau2.__file__).parent / "user" / "user_simulator_streaming.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "INTERRUPTION_DECISION_PROMPT" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SimulatorOutputError(
        f"INTERRUPTION_DECISION_PROMPT not found in {path}", retryable=False
    )


def scenario_instructions(scenario: Scenario) -> str:
    """The ``<scenario>`` block tau2 gives its user: ``str(task.user_scenario)`` for a
    tau2 task (what tau2.runner.build passes), and the same StructuredUserInstructions
    layout rendered from talk-bench's UserScenario for a native task."""
    raw = (scenario.metadata or {}).get("tau2_task")
    if raw:
        from tau2.data_model.tasks import Task

        return str(Task.model_validate(raw).user_scenario)
    us = scenario.user_scenario
    if us is None:
        raise SimulatorOutputError(
            f"scenario {scenario.id!r} has no user_scenario for the tau2 user simulator",
            retryable=False,
        )
    from tau2.data_model.tasks import StructuredUserInstructions, UserScenario

    instructions = StructuredUserInstructions(
        domain=scenario.domain,
        reason_for_call=us.reason_for_call or scenario.goal.description,
        known_info="\n".join(v for v in us.known_info.values() if v) or None,
        unknown_info="\n".join(us.unknown_info) or None,
        task_instructions=us.task_instructions,
    )
    return str(UserScenario(instructions=instructions))


def tau2_persona_config(scenario: Scenario, speech_complexity: str | None) -> PersonaConfig:
    """tau2's runtime persona (verbosity, interrupt tendency) for a tau2 task at
    ``--speech-complexity``, from the same voice config talk-bench samples for that
    run's voice and audio effects. Without it, or for a native task, tau2's default
    persona — the one its ``control`` complexity uses as well (never interrupts)."""
    from tau2.data_model.persona import PersonaConfig

    raw = (scenario.metadata or {}).get("tau2_task")
    if not speech_complexity or not raw:
        return PersonaConfig()
    from talk_bench_domains.tau2.adapter import load_tau2_voice_config

    voice_config = load_tau2_voice_config(scenario.domain, str(raw["id"]), speech_complexity)
    return PersonaConfig.model_validate((voice_config or {}).get("persona_config") or {})


def build_system_prompt(instructions: str, persona: PersonaConfig) -> str:
    """tau2's own UserSimulator.system_prompt with the voice guidelines swapped in —
    exactly what VoiceStreamingUserSimulator.system_prompt returns."""
    from tau2.user.user_simulator import UserSimulator, get_global_user_sim_guidelines_voice

    class _VoicePrompt(UserSimulator):
        @property
        def global_simulation_guidelines(self) -> str:
            return get_global_user_sim_guidelines_voice(use_tools=self.tools is not None)

    return _VoicePrompt(llm="unused", instructions=instructions, persona_config=persona).system_prompt


async def _single_chunk(text: str) -> AsyncIterator[str]:
    yield text


def _resolved(text: str) -> asyncio.Future[str]:
    fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    fut.set_result(text)
    return fut


def _say(kind: str, text: str) -> ActionHandle:
    return ActionHandle(kind=kind, text_chunks=_single_chunk(text), completion=_resolved(text))


def _listen_more() -> ActionHandle:
    return ActionHandle(kind="listen_more", text_chunks=None, completion=_resolved(""))


def _hangup(end_action: EndAction) -> ActionHandle:
    return ActionHandle(
        kind="hangup", text_chunks=None, completion=_resolved(""), hangup_reason=end_action.value
    )


def _strip_end_token(content: str) -> tuple[str, EndAction | None]:
    # tau2's is_stop checks all three with no precedence; one generation never carries two
    for token, end_action in _END_TOKEN_ACTIONS:
        if token in content:
            return content.replace(token, "").strip(), end_action
    return content.strip(), None


class Tau2NativeVoiceUserSimulator:
    """SimulatorProtocol implementation whose decisions come from tau2-bench (see module docstring)."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        llm: str,
        llm_args: dict[str, Any] | None = None,
        seed: int | None = None,
        decision_llm: str | None = None,
        persona_config: PersonaConfig | None = None,
    ) -> None:
        from tau2.data_model.persona import InterruptTendency, PersonaConfig

        self.scenario_id = scenario.id
        self.persona = persona_config or PersonaConfig()
        self.system_prompt = build_system_prompt(scenario_instructions(scenario), self.persona)
        self.llm = llm
        self.llm_args = dict(llm_args or {})
        if seed is not None:
            self.llm_args.setdefault("seed", seed)
        # tau2 pins this to VOICE_USER_SIMULATOR_DECISION_MODEL ("gpt-4.1", OpenAI); the
        # simulator's own model keeps the run on the provider talkbench.yaml configures
        self.decision_llm = decision_llm or llm
        self.interrupts = self.persona.interrupt_tendency == InterruptTendency.INTERRUPTS
        if self.interrupts:
            interruption_decision_prompt()  # fail at construction, not mid-call
        # (len(conversation_history) when noticed, annotation): a silence sits in the
        # history where it happened, as tau2's linearization places it
        self._silence_notes: list[tuple[int, str]] = []
        # Set when a token-bearing reply also had words: they are spoken first and the
        # hangup fires on the next wake, so they are never dropped.
        self._pending_end_action: EndAction | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "simulator": type(self).__name__,
            "llm": self.llm,
            "llm_args": {k: v for k, v in self.llm_args.items() if k != "seed"},
            "decision_llm": self.decision_llm,
        }

    async def on_wake(self, ctx: WakeContext) -> ActionHandle:
        if self._pending_end_action is not None:
            end_action, self._pending_end_action = self._pending_end_action, None
            return _hangup(end_action)

        kind = getattr(ctx.event, "kind", None)
        if kind == "agent_utterance_ready":
            return await self._reply(ctx, "speak")
        if kind in ("silence_exceeded", "awaiting_reply_timeout"):
            if kind == "silence_exceeded":
                seconds = ctx.silence_s or ctx.event.threshold_s  # type: ignore[union-attr]
            else:
                seconds = ctx.event.waited_s  # type: ignore[union-attr]
            self._silence_notes.append((len(ctx.conversation_history), SILENCE_ANNOTATION.format(seconds=seconds)))
            return await self._reply(ctx, "re_engage")
        if kind == "agent_speaking_for":
            partial = (ctx.agent_live_partial_text or "").strip()
            if self.interrupts and partial and await asyncio.to_thread(self._decide_interrupt, ctx):
                return await self._reply(ctx, "barge_in", partial=partial)
            return _listen_more()
        if kind == "agent_utterance_empty":
            # nothing was heard: wait, as tau2 does, until the silence wake checks in
            return _listen_more()
        if kind in _ABORT_EVENT_KINDS or kind == "user_tool_result":
            # user_tool_result is unreachable: this simulator never emits execute_user_tool
            return _hangup(EndAction.OUT_OF_SCOPE)
        logger.warning("[%s] tau2 user simulator: unknown wake event %r — listening", self.scenario_id, kind)
        return _listen_more()

    # ------------------------------------------------------------------

    def timeline(self, ctx: WakeContext, *, partial: str | None = None) -> list[tuple[str, str]]:
        """The conversation as tau2 linearizes it, in tau2's own roles: "assistant" is
        the agent, "user" the simulated caller, "system" a silence annotation.
        Consecutive turns of one speaker merge, as one speech segment would."""
        items: list[tuple[str, str]] = []

        def add(role: str, text: str) -> None:
            if role != "system" and items and items[-1][0] == role:
                items[-1] = (role, f"{items[-1][1]} {text}")
            else:
                items.append((role, text))

        history = ctx.conversation_history
        notes = sorted(self._silence_notes, key=lambda n: n[0])
        for i, msg in enumerate(history):
            for _, note in (n for n in notes if n[0] == i):
                add("system", note)
            text = (msg.content or "").strip()
            if not text:
                continue
            # talk-bench files the agent's turns as USER and the simulator's as ASSISTANT
            if msg.role == MessageRole.USER:
                add("assistant", text)
            elif msg.role == MessageRole.ASSISTANT:
                add("user", text)
        for _, note in (n for n in notes if n[0] >= len(history)):
            add("system", note)
        if partial:
            add("assistant", partial + INCOMPLETE_SUFFIX)
        return items

    def reply_messages(self, ctx: WakeContext, *, partial: str | None = None) -> list[Any]:
        """System prompt + role-flipped timeline + reminder: the simulator plays the
        assistant, so the agent's words arrive as user messages."""
        from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage

        messages: list[Any] = [SystemMessage(role="system", content=self.system_prompt)]
        for role, text in self.timeline(ctx, partial=partial):
            if role == "assistant":
                messages.append(UserMessage(role="user", content=text))
            elif role == "user":
                messages.append(AssistantMessage(role="assistant", content=text))
            else:
                messages.append(SystemMessage(role="system", content=text))
        messages.append(SystemMessage(role="system", content=ROLE_REMINDER))
        return messages

    def _generate(self, ctx: WakeContext, partial: str | None) -> str:
        from tau2.utils.llm_utils import generate

        try:
            reply = generate(
                model=self.llm,
                messages=self.reply_messages(ctx, partial=partial),
                tools=None,
                call_name="user_streaming_response",
                **self.llm_args,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            raise ProviderError(
                f"tau2 user simulator LLM call failed ({self.llm}): {exc}",
                retryable=status is None or status in _RETRYABLE_STATUS,
            ) from exc
        return reply.content or ""

    async def _reply(self, ctx: WakeContext, kind: str, *, partial: str | None = None) -> ActionHandle:
        content = await asyncio.to_thread(self._generate, ctx, partial)
        text, end_action = _strip_end_token(content)
        text = _PAUSE_TAG.sub("...", text).strip()
        if end_action is None:
            return _say(kind, text) if text else _listen_more()
        if text:
            self._pending_end_action = end_action
            return _say(kind, text)
        return _hangup(end_action)

    def _decide_interrupt(self, ctx: WakeContext) -> bool:
        """tau2's user_interruption_policy: the agent's speech so far, its current
        utterance marked incomplete, and a YES/NO from the decision model."""
        from tau2.data_model.message import UserMessage
        from tau2.utils.llm_utils import generate

        items = self.timeline(ctx, partial=ctx.agent_live_partial_text)
        history = "\n".join(f"{role.upper()}: {text}" for role, text in items[-100:])
        prompt = interruption_decision_prompt().format(conversation_history=history)
        try:
            reply = generate(
                model=self.decision_llm,
                messages=[UserMessage(role="user", content=prompt)],
                call_name="interruption_decision",
            )
        except Exception:
            # tau2 defaults to not interrupting when the decision call fails
            logger.exception("[%s] tau2 user simulator: interruption decision failed", self.scenario_id)
            return False
        decision = (reply.content or "").strip().upper() == "YES"
        logger.info("[%s] tau2 user simulator: interrupt? %s", self.scenario_id, "YES" if decision else "NO")
        return decision
