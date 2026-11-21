"""Stub OpenAI-compatible endpoint for zero-cost smoke tests of the parity harness.

Records every /v1/chat/completions request body under --log-dir and answers with:
  * EXP-Bench judge prompts -> fixed, well-formed judge JSON (monitor: nothing flagged;
    design 50, conclusion "incorrect"; setup 25), so the judge pipeline runs end to end;
  * anything else (the agent) -> a `finish` tool call (native or prompt-based, so
    OpenHands stops after one LLM call), otherwise a short text message.

No real model is called. Usage:
  python harbor_parity/stub_llm_server.py --port 8765 --log-dir outputs/harbor_parity/stub_requests
then point MODEL / *_API_BASE (agent) or OPENAI_BASE_URL (judge) at http://<host>:8765/v1.
"""

from __future__ import annotations

import argparse
import itertools
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

COUNTER = itertools.count(1)
LOCK = threading.Lock()

MONITOR = {
    "paper_access": False,
    "git_operations": False,
    "faked_or_nonexperimental_data": False,
    "setup_monitor_comprehensive_reason": "stub: nothing flagged",
}
DESIGN_CONCLUSION = {
    "design_evaluation_explanation": "stub",
    "design_score": 50,
    "design_error_analysis": "stub",
    "conclusion_evaluation_explanation": "stub",
    "conclusion_score": "incorrect",
    "conclusion_error_analysis": "stub",
}
SETUP = {"setup_evaluation_explanation": "stub", "setup_score": 25, "setup_error_analysis": "stub"}


def classify(body: dict) -> str:
    text = "\n".join(
        m["content"] if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
        for m in body.get("messages", [])
    )
    if '"paper_access"' in text:
        return "judge_monitor"
    if '"design_score"' in text:
        return "judge_design_conclusion"
    if '"setup_score"' in text:
        return "judge_setup"
    return "agent"


class Handler(BaseHTTPRequestHandler):
    log_dir: Path

    def log_message(self, fmt, *args):  # quieter default logging
        print(f"[stub-llm] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._send(200, {"object": "list", "data": [{"id": "stub", "object": "model"}]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": f"unsupported path {self.path}"})
        kind = classify(body)
        with LOCK:
            n = next(COUNTER)
        (self.log_dir / f"{n:04d}_{kind}.json").write_text(json.dumps(body, indent=2))

        message: dict = {"role": "assistant", "content": None}
        finish_reason = "stop"
        if kind == "judge_monitor":
            message["content"] = json.dumps(MONITOR)
        elif kind == "judge_design_conclusion":
            message["content"] = json.dumps(DESIGN_CONCLUSION)
        elif kind == "judge_setup":
            message["content"] = json.dumps(SETUP)
        elif kind == "agent" and not body.get("tools") and "<function=" in json.dumps(body.get("messages", [])):
            # Prompt-based tool calling (OpenHands without native tool calling).
            message["content"] = "<function=finish>\n</function>"
        elif any(t.get("function", {}).get("name") == "finish" for t in body.get("tools") or []):
            message["tool_calls"] = [{
                "id": f"call_stub_{n}",
                "type": "function",
                "function": {"name": "finish", "arguments": "{}"},
            }]
            finish_reason = "tool_calls"
        else:
            message["content"] = "Stub LLM: no real model is attached."
        self._send(200, {
            "id": f"chatcmpl-stub-{n}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "stub"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/harbor_parity/stub_requests"))
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    Handler.log_dir = args.log_dir
    print(f"[stub-llm] listening on {args.host}:{args.port}, logging to {args.log_dir}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
