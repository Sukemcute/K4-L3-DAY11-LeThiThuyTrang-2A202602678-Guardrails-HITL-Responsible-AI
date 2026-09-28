import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import NotFoundError
import pytest

from core.config import get_blue_model
from core.openai_runtime import OpenAIAgent, OpenAIRunner


def not_found():
    return NotFoundError("No route", response=httpx.Response(404, request=httpx.Request("POST", "https://example.invalid")), body={})


def test_only_same_blue_model_free_route_is_used_on_404():
    runner = OpenAIRunner(app_name="test", model=get_blue_model(), provider="openrouter")
    client = Mock()
    client.chat.completions.create.side_effect = [not_found(), SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Banking help"))])]
    runner._client = Mock(return_value=client)
    assert asyncio.run(runner.chat(OpenAIAgent("test", "banking"), "account")) == "Banking help"
    assert runner.model == get_blue_model() + ":free"
    assert [c.kwargs["model"] for c in client.chat.completions.create.call_args_list] == [get_blue_model(), get_blue_model() + ":free"]
    client.close.assert_called_once()


def test_red_model_errors_never_trigger_blue_fallback():
    runner = OpenAIRunner(app_name="test", model="gpt-4o-mini", provider="openai")
    client = Mock()
    client.chat.completions.create.side_effect = not_found()
    runner._client = Mock(return_value=client)
    with pytest.raises(NotFoundError):
        asyncio.run(runner.chat(OpenAIAgent("test", "banking"), "account"))
    assert client.chat.completions.create.call_count == 1


def test_real_adk_rate_limit_stops_agent_execution():
    from google.adk.agents import BaseAgent
    from google.adk.events import Event
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    from assignment.rate_limiter import RateLimitPlugin

    class Sentinel(BaseAgent):
        calls: int = 0

        async def _run_async_impl(self, ctx):
            self.calls += 1
            yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part.from_text(text="Banking help")]))

    async def run():
        agent = Sentinel(name="sentinel")
        limiter = RateLimitPlugin(max_requests=1)
        runner = InMemoryRunner(agent=agent, app_name="test", plugins=[limiter])
        try:
            session = await runner.session_service.create_session(app_name="test", user_id="student")
            for _ in range(2):
                events = [e async for e in runner.run_async(user_id="student", session_id=session.id,
                    new_message=types.Content(role="user", parts=[types.Part.from_text(text="account balance")]))]
            assert agent.calls == 1
            assert limiter.blocked_count == 1
            assert any(e.content and any((p.text or "").startswith("Rate limit exceeded.") for p in e.content.parts or []) for e in events)
        finally:
            await runner.close()
    asyncio.run(run())
