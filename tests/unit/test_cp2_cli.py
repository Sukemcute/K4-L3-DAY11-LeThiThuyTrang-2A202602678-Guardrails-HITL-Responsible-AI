"""CP2 must run offline, including on Windows with a legacy console encoding."""
import os
from pathlib import Path
import subprocess
import sys


def test_cp2_without_credentials_or_interactive_input():
    env = dict(os.environ)
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY"):
        env.pop(name, None)
    env["PYTHONIOENCODING"] = "cp1252"
    result = subprocess.run(
        [sys.executable, "src/main.py", "--part", "2"],
        cwd=Path(__file__).resolve().parents[2], env=env,
        input="", capture_output=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "[FAIL]" not in result.stdout
    assert "[REDACTED]" in result.stdout
    assert "API Key" not in result.stdout
