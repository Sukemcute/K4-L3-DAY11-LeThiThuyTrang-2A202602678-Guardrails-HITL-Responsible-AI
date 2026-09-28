"""
Assignment 11 — Rate Limiter starter (TODO).

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60, *, clock=None):
        super().__init__(name="rate_limiter")
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("max_requests must be a positive integer")
        if type(window_seconds) is not int or window_seconds < 1:
            raise ValueError("window_seconds must be a positive integer")
        self._clock = clock or time.monotonic
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = self._clock()
        window = self.user_windows[user_id]

        while window and window[0] <= now - self.window_seconds:
            window.popleft()
        if len(window) >= self.max_requests:
            self.blocked_count += 1
            wait = self.window_seconds - (now - window[0])
            return self._block_response(f"Rate limit exceeded. Try again in {wait:.0f}s.")
        window.append(now)
        return None

    async def before_run_callback(self, *, invocation_context):
        """ADK replaces input in on_user_message; this hook actually stops it."""
        content = invocation_context.user_content
        if content and content.role == "model" and any(
            (part.text or "").startswith("Rate limit exceeded.") for part in content.parts or []
        ):
            return content
        return None
