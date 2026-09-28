"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import base64
import json
import unicodedata

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.utils import chat_with_agent
from core.config import load_protected_payload


_PII_PATTERNS = {
    "phone": re.compile(r"(?<!\w)(?:0|\+84)[ .-]?(?:\d[ .-]?){8,9}\d(?!\w)"),
    "email": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.I),
    "national_id": re.compile(r"(?<!\w)(?:\d{12}|\d{9})(?!\w)"),
    "api_key": re.compile(r"(?<!\w)sk-[a-z0-9_-]{8,}(?!\w)", re.I),
    "password": re.compile(
        r"\b(?:password|mật\s*khẩu)\s*[\"']?\s*[:=]\s*[\"']?(?P<value>[^\s\"',;<>}]+)", re.I
    ),
    "db_host": re.compile(r"\b(?:[a-z0-9-]+\.)+internal\b(?::\d+)?", re.I),
}

# Follow the protected fixture so moving the implementation to a new class repo
# does not silently retain only this class's literal demo credentials.
_PROTECTED = load_protected_payload()
_SECRET_CATEGORY = {"admin_password": "password", "api_key": "api_key", "db_host": "db_host"}
_KNOWN_SECRETS = tuple(
    (_SECRET_CATEGORY.get(target["id"], "secret"), str(value))
    for target in _PROTECTED["leak_targets"]
    for value in {target["value"], *target.get("match_substrings", [])}
    if value
)
_ENCODED_SECRETS = tuple(
    (category, encoded)
    for category, value in _KNOWN_SECRETS
    for encoded in (base64.b64encode(value.encode()).decode(), value.encode().hex())
)


def _matching_view(text: str) -> tuple[str, list[int]]:
    """Map normalized characters back to source offsets for exact redaction.

    Keep clean responses byte-for-byte; only the matching view is normalized.
    A span covers any invisible characters between its visible endpoints.
    """
    chars, offsets = [], []
    for index, char in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", char):
            if unicodedata.category(normalized) != "Cf":
                chars.append(normalized)
                offsets.append(index)
    return "".join(chars), offsets


def _argument_texts(value):
    """Inspect decoded argument strings too (JSON escapes can hide separators)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _argument_texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _argument_texts(item)


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

def content_filter(response: str) -> dict:
    """Redact recognized PII and secrets (not a semantic safety classifier).

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    view, offsets = _matching_view(response)
    findings: dict[str, set[tuple[int, int]]] = {}

    def add(category, start, end, index_map):
        findings.setdefault(category, set()).add((index_map[start], index_map[end - 1] + 1))

    for category, pattern in _PII_PATTERNS.items():
        for match in pattern.finditer(view):
            start, end = match.span("value") if category == "password" else match.span()
            # Re-filtering a previously sanitized message must be idempotent.
            if view[start:end] != "[REDACTED]":
                add(category, start, end, offsets)

    # Recognize demo credentials even when separated by punctuation/whitespace.
    compact_chars, compact_offsets = [], []
    for index, char in enumerate(view):
        if char.isascii() and char.isalnum():
            compact_chars.append(char.lower())
            compact_offsets.append(offsets[index])
    compact = "".join(compact_chars)
    for category, secret in _KNOWN_SECRETS:
        needle = re.sub(r"[^a-z0-9]", "", secret.lower())
        if needle:
            for match in re.finditer(re.escape(needle), compact):
                add(category, *match.span(), compact_offsets)

    # Exact encodings of known fixtures only; arbitrary encoded data and
    # multi-turn reconstruction still require additional security boundaries.
    for category, encoded in _ENCODED_SECRETS:
        for match in re.finditer(re.escape(encoded), view):
            add(category, *match.span(), offsets)

    # Merge overlapping phone/ID/secret spans so no tail survives redaction.
    spans = sorted({span for matches in findings.values() for span in matches})
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    chunks, previous = [], 0
    for start, end in merged:
        chunks.extend((response[previous:start], "[REDACTED]"))
        previous = end
    chunks.append(response[previous:])
    redacted = "".join(chunks)
    # Only categories/counts enter diagnostics, never the matching secret value.
    issues = [f"{category}: {len(matches)} found" for category, matches in sorted(findings.items())]

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts or []:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        # A fragment can split a secret across callbacks ("admin" + "123").
        # This text lab uses complete responses: suppress all partial content
        # and inspect the full response that the ADK streaming contract supplies.
        if getattr(llm_response, "partial", False):
            llm_response.content = types.Content(role="model", parts=[])
            return llm_response

        parts = getattr(getattr(llm_response, "content", None), "parts", None) or []
        for part in parts:
            call = getattr(part, "function_call", None)
            call_texts = (
                [json.dumps(call.model_dump(), ensure_ascii=False), *_argument_texts(call.args)]
                if call else []
            )
            if any(not content_filter(text)["safe"] for text in call_texts):
                self.blocked_count += 1
                llm_response.content = types.Content(role="model", parts=[types.Part.from_text(
                    text="Không thể gửi dữ liệu nhạy cảm qua công cụ."
                )])
                return llm_response

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        result = content_filter(response_text)
        if not result["safe"]:
            self.redacted_count += 1
            # Rebuild a text-only response: don't forward calls attached to an
            # unsafe answer. Clean responses retain their original parts.
            llm_response.content = types.Content(role="model", parts=[
                types.Part.from_text(text=result["redacted"])
            ])
            if any(getattr(part, "function_call", None) for part in parts):
                self.blocked_count += 1

        if self.use_llm_judge:
            # Do not send recognized credentials to a separate judge service.
            try:
                verdict = await llm_safety_check(result["redacted"])
                safe = verdict.get("safe") is True
            except Exception:
                safe = False
            if not safe:
                self.blocked_count += 1
                llm_response.content = types.Content(role="model", parts=[types.Part.from_text(
                    text="Không thể cung cấp câu trả lời này sau bước kiểm tra an toàn."
                )])
        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
