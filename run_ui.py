#!/usr/bin/env python3
"""
Launcher for VinBank Guardrails Playground & Trace Inspector Web UI.
Run from repo root:
    python run_ui.py
"""
import sys
import webbrowser
from pathlib import Path

# Add src to sys.path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if __name__ == "__main__":
    import uvicorn

    port = 8000
    host = "0.0.0.0"
    url = f"http://localhost:{port}"

    print("\n" + "=" * 65)
    print("[*] VINBANK GUARDRAILS & TRACE INSPECTOR PLAYGROUND")
    print("=" * 65)
    print(f"--> Local Web UI:    {url}")
    print(f"--> Network Web UI:  http://127.0.0.1:{port}")
    print("Features:")
    print("   - LangSmith-style Waterfall Trace Inspector")
    print("   - 5 Defensive Layers Visualization")
    print("   - Quick Attack Presets (DAN, Injection, Unicode, PII, Spam)")
    print("   - Forensics Audit Log & Real-time Security Metrics")
    print("=" * 65 + "\n")

    uvicorn.run("ui.server:app", host=host, port=port, reload=False)
