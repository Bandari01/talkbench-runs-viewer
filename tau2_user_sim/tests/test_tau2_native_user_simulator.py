import asyncio
import json

import pytest
import tau2.utils.llm_utils as llm_utils
from talk_bench_core.scenario import GoalConfig, Scenario, UserScenario
from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau2.data_model.persona import InterruptTendency, PersonaConfig, Verbosity
from tau2_native_user_simulator import (
    ROLE_REMINDER,
    Tau2NativeVoiceUserSimulator,
    interruption_decision_prompt,
    scenario_instructions,
)
from user_simulator.exceptions import ProviderError
from user_simulator.types.simulator import Message, MessageRole, WakeContext

TAU2_TASK = {
    "id": "7",
    "user_scenario": {
        "persona": None,
        "instructions": {
            "domain": "airline",
            "reason_for_call": "You want to cancel reservation ABC123.",
            "known_info": "You are Ann Lee.\nYour user id is ann_lee_1.",
            "unknown_info": None,
            "task_instructions": "Only cancel if you get a refund.",
        },
    },
}


def tau2_scenario() -> Scenario:
    return Scenario(
        id="airline-tau2-7", domain="airline",
        goal=GoalConfig(description="cancel"), metadata={"tau2_task": TAU2_TASK},
    )


def native_scenario() -> Scenario:
    return Scenario(
        id="internal-test-001", domain="internal-test", goal=GoalConfig(description="book"),
        user_scenario=UserScenario(
            reason_for_call="You want to book JFK to Miami.",
            known_info={"name": "You are James Wilson.", "user_id": "Your user id is user_003."},
            unknown_info=["the flight options"],
            task_instructions="Give your user id when asked.",
        ),
    )


def ctx(kind: str, history: list[tuple[str, str]], **extra) -> WakeContext:
    events = {
        "agent_utterance_ready": {"kind": kind, "text": history[-1][1] if history else "", "duration_ms": 900},
        "agent_utterance_empty": {"kind": kind},
        "silence_exceeded": {"kind": kind, "threshold_s": 8.0},
        "agent_speaking_for": {"kind": kind, "duration_s": 4.0},
        "awaiting_reply_timeout": {"kind": kind, "waited_s": 60.0},
        "conversation_timeout": {"kind": kind, "elapsed_s": 900.0},
    }
    roles = {"agent": MessageRole.USER, "sim": MessageRole.ASSISTANT}
    return WakeContext(
        event=events[kind], agent_speaking=kind == "agent_speaking_for", silence_s=extra.pop("silence_s", 0.0),
        last_agent_utterance=None, last_agent_utterance_age_s=None,
        agent_live_partial_text=extra.pop("partial", None), last_action_dropped=None,
        last_barge_in_outcome=None, last_barge_in_outcome_reason=None, turn_count=0, elapsed_s=5.0,
        conversation_history=[Message(role=roles[who], content=text) for who, text in history],
        recent_tool_results=[],
    )


class FakeLLM:
    """Stands in for tau2.utils.llm_utils.generate; records every call."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def __call__(self, model, messages, tools=None, call_name=None, **kwargs):
        self.calls.append({"model": model, "messages": messages, "tools": tools, "call_name": call_name, **kwargs})
        return AssistantMessage(role="assistant", content=self.replies.pop(0))


@pytest.fixture
def llm(monkeypatch):
    def install(*replies: str) -> FakeLLM:
        fake = FakeLLM(*replies)
        monkeypatch.setattr(llm_utils, "generate", fake)
        return fake

    return install


def run(sim, wake):
    async def go():
        handle = await sim.on_wake(wake)
        text = await handle.completion
        chunks = [c async for c in handle.text_chunks] if handle.text_chunks else []
        return handle, text, chunks

    return asyncio.run(go())


def make_sim(scenario=None, **kwargs) -> Tau2NativeVoiceUserSimulator:
    kwargs.setdefault("llm", "azure/gpt-4.1")
    return Tau2NativeVoiceUserSimulator(scenario or tau2_scenario(), **kwargs)


# --- prompt ---------------------------------------------------------------------


def test_system_prompt_is_tau2_voice_prompt():
    prompt = make_sim().system_prompt
    assert prompt.startswith("# Voice Call Simulation Guidelines")
    assert "<PERSONA_GUIDELINES>" not in prompt
    assert "MINIMAL VERBOSITY" not in prompt  # default persona adds nothing
    assert "<scenario>\nInstructions:\n\tDomain: airline" in prompt
    assert "Your user id is ann_lee_1." in prompt


def test_minimal_persona_adds_tau2_guidelines():
    sim = make_sim(persona_config=PersonaConfig(verbosity=Verbosity.MINIMAL))
    assert "## MINIMAL VERBOSITY" in sim.system_prompt
    assert not sim.interrupts


def test_native_scenario_rendered_in_tau2_layout():
    text = scenario_instructions(native_scenario())
    assert text.startswith("Instructions:\n\tDomain: internal-test\n\tReason for call:")
    assert "Known info:\n\t\tYou are James Wilson.\n\t\tYour user id is user_003." in text
    assert "Unknown info:\n\t\tthe flight options" in text
    assert "Task instructions:\n\t\tGive your user id when asked." in text


def test_interruption_prompt_read_from_vendored_tau2():
    prompt = interruption_decision_prompt()
    assert "{conversation_history}" in prompt
    assert 'Respond with ONLY "YES"' in prompt


# --- replies --------------------------------------------------------------------


def test_reply_flips_roles_and_appends_reminder(llm):
    fake = llm("Hi, um, I need to cancel a reservation.")
    sim = make_sim(llm_args={"temperature": 0.0}, seed=11)
    handle, text, chunks = run(sim, ctx("agent_utterance_ready", [("agent", "Hello, how can I help?")]))
    assert handle.kind == "speak"
    assert text == chunks[0] == "Hi, um, I need to cancel a reservation."
    call = fake.calls[0]
    assert call["call_name"] == "user_streaming_response"
    assert call["model"] == "azure/gpt-4.1" and call["temperature"] == 0.0 and call["seed"] == 11
    assert call["tools"] is None
    msgs = call["messages"]
    assert isinstance(msgs[0], SystemMessage) and msgs[0].content == sim.system_prompt
    assert isinstance(msgs[1], UserMessage) and msgs[1].content == "Hello, how can I help?"
    assert isinstance(msgs[-1], SystemMessage) and msgs[-1].content == ROLE_REMINDER


def test_history_merges_consecutive_agent_utterances(llm):
    fake = llm("It's ann underscore lee underscore one.")
    history = [("agent", "Hi."), ("sim", "I want to cancel."), ("agent", "Sure."), ("agent", "Your user id?")]
    run(make_sim(), ctx("agent_utterance_ready", history))
    msgs = fake.calls[0]["messages"][1:-1]
    assert [(type(m).__name__, m.content) for m in msgs] == [
        ("UserMessage", "Hi."), ("AssistantMessage", "I want to cancel."), ("UserMessage", "Sure. Your user id?"),
    ]


def test_pause_markers_become_ellipses(llm):
    llm("My id is [pause] ann underscore lee")
    _, text, _ = run(make_sim(), ctx("agent_utterance_ready", [("agent", "Your id?")]))
    assert text == "My id is ... ann underscore lee"


def test_stop_token_with_words_speaks_then_hangs_up(llm):
    llm("Great, thanks, bye! ###STOP###")
    sim = make_sim()
    handle, text, _ = run(sim, ctx("agent_utterance_ready", [("agent", "It's cancelled.")]))
    assert (handle.kind, text) == ("speak", "Great, thanks, bye!")
    handle, _, _ = run(sim, ctx("agent_utterance_ready", [("agent", "Goodbye.")]))
    assert (handle.kind, handle.hangup_reason) == ("hangup", "resolved")


@pytest.mark.parametrize("token,reason", [("###STOP###", "resolved"), ("###TRANSFER###", "transferred"),
                                          ("###OUT-OF-SCOPE###", "out_of_scope")])
def test_bare_end_token_hangs_up(llm, token, reason):
    llm(token)
    handle, _, _ = run(make_sim(), ctx("agent_utterance_ready", [("agent", "Anything else?")]))
    assert (handle.kind, handle.hangup_reason) == ("hangup", reason)


def test_empty_reply_listens(llm):
    llm("")
    handle, _, _ = run(make_sim(), ctx("agent_utterance_ready", [("agent", "Okay.")]))
    assert handle.kind == "listen_more"


# --- silence --------------------------------------------------------------------


def test_silence_checks_in_with_tau2_annotation_that_stays_in_history(llm):
    fake = llm("Hello? Are you still there?", "Yes, ann underscore lee underscore one.")
    sim = make_sim()
    history = [("agent", "Hi."), ("sim", "Cancel please."), ("agent", "Let me check.")]
    handle, _, _ = run(sim, ctx("silence_exceeded", history, silence_s=8.4))
    assert handle.kind == "re_engage"
    assert fake.calls[0]["messages"][-2].content == "[Both parties silent for 8.4 seconds]"
    history += [("sim", "Hello? Are you still there?"), ("agent", "Yes, your user id?")]
    run(sim, ctx("agent_utterance_ready", history))
    contents = [m.content for m in fake.calls[1]["messages"][1:-1]]
    assert contents.index("[Both parties silent for 8.4 seconds]") == 3  # after "Let me check.", before the check-in


def test_agent_utterance_empty_waits_without_llm(llm):
    fake = llm()
    handle, _, _ = run(make_sim(), ctx("agent_utterance_empty", [("agent", "Hi.")]))
    assert handle.kind == "listen_more" and not fake.calls


# --- interruptions --------------------------------------------------------------


def interrupting_sim():
    return make_sim(persona_config=PersonaConfig(interrupt_tendency=InterruptTendency.INTERRUPTS))


def test_non_interrupting_persona_never_barges_in(llm):
    fake = llm()
    handle, _, _ = run(make_sim(), ctx("agent_speaking_for", [("agent", "Hi.")], partial="So the fare rules say"))
    assert handle.kind == "listen_more" and not fake.calls


def test_interrupt_yes_barges_in_with_partial_marked_incomplete(llm):
    fake = llm("YES", "Sorry, I just want a refund.")
    history = [("agent", "Hi."), ("sim", "Cancel ABC123.")]
    handle, text, _ = run(interrupting_sim(), ctx("agent_speaking_for", history, partial="The fare rules say that"))
    assert (handle.kind, text) == ("barge_in", "Sorry, I just want a refund.")
    decision = fake.calls[0]
    assert decision["call_name"] == "interruption_decision" and decision["model"] == "azure/gpt-4.1"
    prompt = decision["messages"][0].content
    assert "ASSISTANT: Hi.\nUSER: Cancel ABC123.\nASSISTANT: The fare rules say that [CURRENTLY SPEAKING, INCOMPLETE]" in prompt
    assert fake.calls[1]["messages"][-2].content == "The fare rules say that [CURRENTLY SPEAKING, INCOMPLETE]"


def test_interrupt_no_keeps_listening(llm):
    fake = llm("NO")
    handle, _, _ = run(interrupting_sim(), ctx("agent_speaking_for", [("agent", "Hi.")], partial="Let me"))
    assert handle.kind == "listen_more" and len(fake.calls) == 1


def test_decision_model_override(llm):
    fake = llm("NO")
    sim = make_sim(decision_llm="azure/gpt-4.1-mini",
                   persona_config=PersonaConfig(interrupt_tendency=InterruptTendency.INTERRUPTS))
    run(sim, ctx("agent_speaking_for", [("agent", "Hi.")], partial="Let me"))
    assert fake.calls[0]["model"] == "azure/gpt-4.1-mini"


# --- errors / terminal events ---------------------------------------------------


@pytest.mark.parametrize("status,retryable", [(429, True), (None, True), (401, False)])
def test_llm_failure_is_a_provider_error(monkeypatch, status, retryable):
    def boom(**_):
        exc = RuntimeError("upstream said no")
        if status is not None:
            exc.status_code = status
        raise exc

    monkeypatch.setattr(llm_utils, "generate", boom)
    with pytest.raises(ProviderError) as err:
        run(make_sim(), ctx("agent_utterance_ready", [("agent", "Hi.")]))
    assert err.value.retryable is retryable


def test_conversation_timeout_hangs_up(llm):
    handle, _, _ = run(make_sim(), ctx("conversation_timeout", [("agent", "Hi.")]))
    assert (handle.kind, handle.hangup_reason) == ("hangup", "out_of_scope")


# --- launcher -------------------------------------------------------------------


class StubRunner:
    """The slice of BenchmarkRunner the launcher patches."""

    def __init__(self, config, out_dir):
        self.config = config
        self._run_output_dir = out_dir

    async def _create_components(self, scenario, seed=None, *, is_text=False):
        return "env", "talk-bench-simulator"


def test_launcher_swaps_voice_simulator_only(monkeypatch, tmp_path):
    import launch
    import talk_bench.runner.runner as runner_mod
    from talk_bench.types.runner import RunConfig
    from voice_agent.abc import AgentConfig

    monkeypatch.setattr(runner_mod, "BenchmarkRunner", StubRunner)
    monkeypatch.setattr(StubRunner, "__init__", StubRunner.__init__)
    monkeypatch.setattr(StubRunner, "_create_components", StubRunner._create_components)
    monkeypatch.setattr(launch, "_markers_written", set())
    launch.install()

    config = RunConfig(
        agent_config_path="agent.yaml",
        agent_config=AgentConfig(name="a", module="voice_agent.cascaded.CascadedAgent"),
        terminal_hangup_gate_enabled=True,
    )
    runner = StubRunner(config, tmp_path)
    assert runner.config.terminal_hangup_gate_enabled is False

    env, sim = asyncio.run(runner._create_components(tau2_scenario(), 5))
    assert env == "env" and isinstance(sim, Tau2NativeVoiceUserSimulator)
    assert sim.llm == config.llm_config.model and sim.llm_args["seed"] == 5
    marker = json.loads((tmp_path / "user_simulator.json").read_text())
    assert marker["engine"] == "tau2" and marker["simulator"] == "Tau2NativeVoiceUserSimulator"

    _, text_sim = asyncio.run(runner._create_components(tau2_scenario(), 5, is_text=True))
    assert text_sim == "talk-bench-simulator"


def test_launcher_refuses_an_incompatible_runner(monkeypatch):
    import launch
    import talk_bench.runner.runner as runner_mod

    class Changed(StubRunner):
        async def _create_components(self, scenario, rng):  # signature talk-bench no longer has
            return None

    monkeypatch.setattr(runner_mod, "BenchmarkRunner", Changed)
    with pytest.raises(SystemExit) as err:
        launch.install()
    assert err.value.code == 2


def test_stub_matches_real_runner_signature():
    import inspect

    from talk_bench.runner.runner import BenchmarkRunner

    real = inspect.signature(BenchmarkRunner._create_components).parameters
    assert list(real) == list(inspect.signature(StubRunner._create_components).parameters)
