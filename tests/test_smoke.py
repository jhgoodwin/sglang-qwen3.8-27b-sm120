import json, sys, unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from bench import smoke

class SmokeTests(unittest.TestCase):
    def test_fake_server_equivalent(self):
        with TemporaryDirectory() as out:
            def fake(url, payload=None):
                if url.endswith("/health"): return 200, "OK"
                if url.endswith("/v1/models"): return 200, {"data": [{"id": "Qwen/Qwen3.8-27B"}]}
                if url.endswith("/get_server_info"): return 404, {}
                if payload and payload.get("stream"): return 200, {"events": [{"choices": [{"delta": {"content": "ok"}}]}]}
                msg = {"role": "assistant", "content": "ok"}
                if payload and payload.get("tool_choice"): msg = {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "smoke", "arguments": "{}"}}]}
                return 200, {"choices": [{"message": msg}]}
            with patch.object(smoke, "call", side_effect=fake), patch.object(sys, "argv", ["smoke.py", "--output", out]): smoke.main()
            self.assertTrue(json.loads((Path(out) / "metadata.json").read_text())["checks"]["tool_calls"])

    def test_malformed_response_fails(self):
        with self.assertRaises(AssertionError): smoke.assert_chat({}, "malformed")
