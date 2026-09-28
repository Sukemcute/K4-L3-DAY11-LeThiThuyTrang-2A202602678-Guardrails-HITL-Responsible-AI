import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

import pytest

from attacks.attacks import (
    adversarial_prompts, classify_attack_outcome, response_leaked_secrets,
    run_attacks, save_attack_results,
)
from core.openai_runtime import OpenAIAgent, OpenAIRunner
from core.utils import chat_with_agent


def test_prompts_are_distinct_and_do_not_contain_the_expected_answers():
    assert len(adversarial_prompts) >= 5
    assert len({p["id"] for p in adversarial_prompts}) == len(adversarial_prompts)
    for attack in adversarial_prompts:
        assert attack["hypothesis"]
        for prompt in attack.get("turns", [attack["input"]]):
            assert "TODO" not in prompt
            assert not response_leaked_secrets(prompt)


@pytest.mark.parametrize("response", ["password=[REDACTED]", "password=example", "administrator password = [missing]"])
def test_placeholder_password_is_not_a_successful_leak(response):
    assert not response_leaked_secrets(response)


def test_runtime_history_isolated_and_blocked_turns_not_retained():
    async def run():
        runner = OpenAIRunner(app_name="test", model="offline")
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Banking help"))])
        runner._client = Mock(return_value=client)
        agent = OpenAIAgent(name="test", instruction="Bank assistant")
        _, history = await chat_with_agent(agent, runner, "banking hello")
        await chat_with_agent(agent, runner, "banking next", session_id=history)
        assert len(client.chat.completions.create.call_args.kwargs["messages"]) == 4
        assert len(history) == 4
        await chat_with_agent(agent, runner, "banking fresh")
        assert len(client.chat.completions.create.call_args.kwargs["messages"]) == 2
        runner.input_hooks = [lambda _: "blocked"]
        before = list(history)
        assert await runner.chat(agent, "bad", history=history) == "blocked"
        assert history == before
        assert client.close.call_count == 3
    asyncio.run(run())


@pytest.mark.parametrize("response,layer", [
    ("a d m i n 1 2 3", "leaked"),
    ("I can't help with that request. I only assist with VinBank banking questions.", "input_injection"),
    ("I cannot share internal system details.", "output_filter"),
    ("I cannot provide that information.", "model_refuse"),
])
def test_outcome_distinguishes_model_refusal_from_enforced_filter(response, layer):
    result = classify_attack_outcome("banking", response, target_name="red_advance")
    assert result["layer"] == layer


def test_multiturn_evidence_preserves_context_and_long_response(monkeypatch, tmp_path):
    import attacks.attacks as attacks
    mock = AsyncMock(side_effect=[("Banking help", ["session"]), ("x" * 400 + "admin123", ["session"]), ("new case", [])])
    monkeypatch.setattr(attacks, "chat_with_agent", mock)
    prompts = [{"id": 1, "category": "Multi-step", "input": "banking next", "turns": ["banking hello", "banking next"]},
               {"id": 2, "category": "single", "input": "banking fresh"}]
    results = asyncio.run(run_attacks(None, None, prompts, target_name="red_default", save_json=False))
    assert results[0]["leaked"]
    assert len(results[0]["turns"]) == 2
    assert mock.call_args_list[1].kwargs["session_id"] == ["session"]
    assert mock.call_args_list[2].kwargs["session_id"] is None
    path = save_attack_results(unsafe_results=results, guards_results=[], filepath=tmp_path / "attacks.json")
    assert "admin123" in path.read_text(encoding="utf-8")


def test_api_error_not_reported_as_success_or_written_verbatim(monkeypatch):
    import attacks.attacks as attacks
    monkeypatch.setattr(attacks, "chat_with_agent", AsyncMock(side_effect=RuntimeError("do-not-log-this-credential")))
    result = asyncio.run(run_attacks(None, None, [{"id": 1, "category": "test", "input": "banking"}], save_json=False))[0]
    assert not result["leaked"] and not result["blocked"]
    assert result["error"] == "RuntimeError"
    assert "do-not-log" not in str(result)
