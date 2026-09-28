"""Offline redaction contracts and output callback regressions."""
import asyncio
import base64
from unittest.mock import AsyncMock

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

import guardrails.output_guardrails as output


@pytest.mark.parametrize("case", output.load_lab_pii_dataset()["pii_cases"], ids=lambda c: c["id"])
def test_lab_dataset(case):
    result = output.content_filter(case["input_text"])
    assert result["safe"] is case["expect_safe"]
    categories = {issue.split(":")[0] for issue in result["issues"]}
    assert set(case["expect_issue_types"]) <= categories
    assert ("[REDACTED]" in result["redacted"]) is case["expect_contains_redacted"]


@pytest.mark.parametrize("secret", [
    "admin123", "sk-vinbank-secret-2024", "db.vinbank.internal:5432",
    "a d m i n 1 2 3", "ad\u200bmin123", "ａｄｍｉｎ１２３",
    "sk / vinbank / secret / 2024", "db vinbank internal",
    base64.b64encode(b"admin123").decode(), b"admin123".hex(),
    "0901234567", "+84 901 234 567", "090 123 4567", "02838223344",
    "user+statement@example.com", "079204001234", "023456789",
])
def test_redacts_whole_sensitive_value_and_preserves_surroundings(secret):
    result = output.content_filter("Thông tin: " + secret + "; xin cảm ơn.")
    assert result["safe"] is False
    assert result["redacted"] == "Thông tin: [REDACTED]; xin cảm ơn."
    assert all(secret not in issue for issue in result["issues"])


@pytest.mark.parametrize("text", [
    "", "The savings rate is 4.25%.", "Your password is protected.",
    "Reset your account password in the app.", "Số tiền: 500000 VND.",
    "VinBank hotline: 1900 545 467", "Đổi mật khẩu thẻ ATM.",
])
def test_clean_content_unchanged(text):
    assert output.content_filter(text) == {"safe": True, "issues": [], "redacted": text}


@pytest.mark.parametrize("text", [
    "password=Secret!99", '"password": "Secret!99"', "mật khẩu: Secret!99",
    "Admin password is admin123; sk-vinbank-secret-2024; 079204001234",
])
def test_redaction_idempotent(text):
    result = output.content_filter(text)
    assert result["safe"] is False
    assert output.content_filter(result["redacted"])["safe"] is True
    assert output.content_filter(result["redacted"])["redacted"] == result["redacted"]


def response(*texts, **kwargs):
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part.from_text(text=text) for text in texts
    ]), **kwargs)


def apply(plugin, value):
    return asyncio.run(plugin.after_model_callback(callback_context=None, llm_response=value))


def test_multipart_secret_cannot_bypass_redaction():
    plugin = output.OutputGuardrailPlugin(use_llm_judge=False)
    result = apply(plugin, response("Password: ad", "min123"))
    assert result.content.parts[0].text == "Password: [REDACTED]"
    assert (plugin.total_count, plugin.redacted_count, plugin.blocked_count) == (1, 1, 0)


def test_clean_response_preserves_metadata_and_parts():
    plugin = output.OutputGuardrailPlugin(use_llm_judge=False)
    value = response("Savings ", "4.25%", model_version="offline-test")
    previous = value.model_dump()
    assert apply(plugin, value) is value
    assert value.model_dump() == previous
    assert plugin.redacted_count == 0


@pytest.mark.parametrize("content", [None, types.Content(role="model", parts=None)])
def test_empty_response(content):
    value = LlmResponse(content=content)
    assert apply(output.OutputGuardrailPlugin(False), value) is value


def test_partial_fragments_suppressed_until_full_response():
    plugin = output.OutputGuardrailPlugin(False)
    for fragment in ("ad", "min", "123"):
        assert apply(plugin, response(fragment, partial=True)).content.parts == []
    assert apply(plugin, response("admin123", partial=False)).content.parts[0].text == "[REDACTED]"


@pytest.mark.parametrize("body", ["admin123", "ad\nmin123", {"nested": ["admin123"]}])
def test_sensitive_tool_arguments_are_blocked(body):
    plugin = output.OutputGuardrailPlugin(False)
    value = LlmResponse(content=types.Content(role="model", parts=[
        types.Part.from_function_call(name="send_email", args={"body": body})
    ]))
    result = apply(plugin, value)
    assert not any(part.function_call for part in result.content.parts)
    assert "admin123" not in result.model_dump_json()
    assert plugin.blocked_count == 1


def test_safe_tool_call_preserved_for_downstream_authorization():
    plugin = output.OutputGuardrailPlugin(False)
    value = LlmResponse(content=types.Content(role="model", parts=[
        types.Part.from_function_call(name="get_interest_rate", args={"months": 12})
    ]))
    previous = value.model_dump()
    assert apply(plugin, value).model_dump() == previous


def test_optional_judge_only_receives_redacted_text(monkeypatch):
    judge = AsyncMock(return_value={"safe": True})
    monkeypatch.setattr(output, "safety_judge_agent", object())
    monkeypatch.setattr(output, "llm_safety_check", judge)
    plugin = output.OutputGuardrailPlugin(True)
    apply(plugin, response("admin123"))
    judge.assert_awaited_once_with("[REDACTED]")


def test_optional_judge_error_blocks_answer(monkeypatch):
    monkeypatch.setattr(output, "safety_judge_agent", object())
    monkeypatch.setattr(output, "llm_safety_check", AsyncMock(side_effect=RuntimeError("offline")))
    plugin = output.OutputGuardrailPlugin(True)
    result = apply(plugin, response("Savings 4.25%"))
    assert plugin.blocked_count == 1
    assert result.content.parts[0].text != "Savings 4.25%"
