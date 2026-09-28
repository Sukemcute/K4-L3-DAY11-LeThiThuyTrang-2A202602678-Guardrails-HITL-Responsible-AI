"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time
from uuid import uuid4

from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str], dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Track requests independently; persist only sanitized text in logs."""
        rid = request_id or str(uuid4())
        key = (user_id, rid)
        if key in self._open:
            raise ValueError("Duplicate pending request; supply a unique request_id")
        self._open[key] = {
            "request_id": rid, "user_id": user_id, "timestamp": utc_now_iso(),
            "input": content_filter(text)["redacted"], "started": time.monotonic(),
        }
        return rid

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Finish a matching request; never manufacture missing input records."""
        if request_id is None:
            matches = [key for key in self._open if key[0] == user_id]
            if len(matches) != 1:
                raise ValueError("Supply request_id when there is not exactly one pending request")
            key = matches[0]
        else:
            key = (user_id, request_id)
        record = self._open.pop(key)
        started = record.pop("started")
        record.update(
            response=content_filter(text)["redacted"], blocked=blocked, layer=layer,
            latency_sec=round(max(0.0, time.monotonic() - started), 6),
        )
        self.logs.append(record)
        return record

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
