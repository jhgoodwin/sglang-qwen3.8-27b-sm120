import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bench import prompt_runner


class PromptRunnerTests(unittest.TestCase):
    def test_stream_contract_metadata_and_prompt_hash(self):
        requests = []
        def fake_request(url, body=None, timeout=60):
            if body is None:
                return 200, "application/json", {"model": "fake", "max_total_num_tokens": 1000}
            requests.append(body)
            return 200, "text/event-stream", {"events": [
                {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
                           "speculative_acceptance_rate": 0.5}}], "content": "ok",
                "reasoning_content": "", "finish_reason": "stop", "usage":
                {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
                 "speculative_acceptance_rate": 0.5}, "first_token_time": 1.0}
        with TemporaryDirectory() as directory:
                prompt_dir = Path(directory) / "prompts"
                prompt_dir.mkdir()
                (prompt_dir / "b.txt").write_bytes(b"second\n")
                (prompt_dir / "a.txt").write_bytes("first — exact\n".encode())
                output = Path(directory) / "run.json"
                with patch.object(prompt_runner, "json_request", side_effect=fake_request):
                    prompt_runner.main(["--prompt-dir", str(prompt_dir), "--base-url",
                        "http://fake", "--output", str(output)])
                document = json.loads(output.read_text())
                self.assertEqual([r["prompt_file"] for r in document["results"]], ["a.txt", "b.txt"])
                self.assertEqual(document["metadata"]["max_tokens"], 32768)
                self.assertEqual(document["results"][0]["content"], "ok")
                self.assertEqual(document["results"][0]["finish_reason"], "stop")
                self.assertTrue(document["results"][0]["completion_valid"])
                self.assertNotIn("incomplete", document["results"][0])
                self.assertEqual(document["results"][0]["usage"]["completion_tokens"], 2)
                self.assertIn("speculative_acceptance_rate", document["results"][0]["speculative_stats"])
                self.assertIsNotNone(document["results"][0]["ttft_s"])
                for request in requests:
                    self.assertEqual(request["max_tokens"], 32768)
                    self.assertEqual(set(request) - {"model", "messages", "max_tokens", "stream", "stream_options"}, set())
                    self.assertEqual(request["stream_options"], {"include_usage": True})
                    self.assertNotIn("temperature", request)
                    self.assertNotIn("top_p", request)
                    self.assertNotIn("top_k", request)
                    self.assertNotIn("reasoning_effort", request)
                self.assertEqual(requests[0]["messages"][0]["content"], "first — exact\n")

    def test_dry_run_does_not_contact_server_and_preserves_default(self):
        with TemporaryDirectory() as directory:
            prompt_dir = Path(directory) / "prompts"
            prompt_dir.mkdir()
            (prompt_dir / "one.txt").write_text("unchanged")
            output = Path(directory) / "dry.json"
            prompt_runner.main(["--prompt-dir", str(prompt_dir), "--base-url", "http://127.0.0.1:1",
                                "--output", str(output), "--dry-run"])
            document = json.loads(output.read_text())
            self.assertEqual(document["results"][0]["request"]["max_tokens"], 32768)
            self.assertNotIn("temperature", document["results"][0]["request"])

            explicit = Path(directory) / "explicit.json"
            prompt_runner.main(["--prompt-dir", str(prompt_dir), "--output", str(explicit),
                                "--dry-run", "--reasoning-effort", "low"])
            request = json.loads(explicit.read_text())["results"][0]["request"]
            self.assertEqual(request["reasoning_effort"], "low")

    def test_dry_run_request_matches_live_request(self):
        with TemporaryDirectory() as directory:
            prompt_dir = Path(directory) / "prompts"
            prompt_dir.mkdir()
            (prompt_dir / "one.txt").write_text("unchanged — exact\n")
            dry_output = Path(directory) / "dry.json"
            prompt_runner.main(["--prompt-dir", str(prompt_dir), "--output", str(dry_output),
                                "--dry-run", "--max-tokens", "32768",
                                "--reasoning-effort", "medium"])
            dry_request = json.loads(dry_output.read_text())["results"][0]["request"]

            requests = []
            def fake_request(url, body=None, timeout=1800):
                requests.append(body)
                return 200, "text/event-stream", {"events": [], "content": "",
                    "reasoning_content": "", "finish_reason": "stop", "usage": None,
                    "first_token_time": None}
            with patch.object(prompt_runner, "json_request", side_effect=fake_request):
                live = prompt_runner.run_prompt("http://fake", "Qwen/Qwen3.8-27B", prompt_dir / "one.txt",
                                                32768, 1800, True, "medium")
            self.assertEqual(dry_request, requests[0])
            self.assertEqual(live["request"], dry_request)
            self.assertEqual(dry_request["stream_options"], {"include_usage": True})
            for name in ("temperature", "top_p", "top_k"):
                self.assertNotIn(name, dry_request)

    def test_monotonic_timing_and_post_first_token_formula(self):
        with TemporaryDirectory() as directory:
            prompt = Path(directory) / "one.txt"
            prompt.write_text("prompt")
            response = {"content": "ok", "reasoning_content": "", "finish_reason": "stop",
                        "usage": {"completion_tokens": 2}, "first_token_time": 11.5}
            with patch.object(prompt_runner, "json_request", return_value=(200, "application/json", response)), \
                 patch.object(prompt_runner.time, "time", side_effect=[1000.0, 1002.0]), \
                 patch.object(prompt_runner.time, "monotonic", side_effect=[10.0, 12.0]):
                result = prompt_runner.run_prompt("http://fake", "model", prompt, 32768,
                                                  1800, True, "medium")
            self.assertEqual(result["ttft_s"], 1.5)
            self.assertEqual(result["wall_duration_s"], 2.0)
            self.assertEqual(result["completion_tok_s_after_first"], 2.0)
            self.assertEqual(result["completion_tok_s_end_to_end"], 1.0)
            self.assertEqual(result["request"]["reasoning_effort"], "medium")
            for name in ("temperature", "top_p", "top_k"):
                self.assertNotIn(name, result["request"])

    def test_missing_usage_is_explicit(self):
        with TemporaryDirectory() as directory:
            prompt = Path(directory) / "one.txt"
            prompt.write_text("prompt")
            response = {"content": "ok", "reasoning_content": "", "finish_reason": "stop",
                        "usage": None, "first_token_time": None}
            with patch.object(prompt_runner, "json_request", return_value=(200, "application/json", response)):
                result = prompt_runner.run_prompt("http://fake", "model", prompt, 32768,
                                                  1800, True)
            self.assertIsNone(result["completion_tok_s_end_to_end"])
            self.assertIn("completion_token_usage_missing", result["metrics_unavailable"])

    def test_length_finish_is_recorded_as_incomplete(self):
        with TemporaryDirectory() as directory:
            prompt = Path(directory) / "one.txt"
            prompt.write_text("prompt")
            response = {"content": "partial", "reasoning_content": "",
                        "finish_reason": "length", "usage": {"completion_tokens": 32},
                        "first_token_time": None}
            with patch.object(prompt_runner, "json_request",
                              return_value=(200, "application/json", response)):
                result = prompt_runner.run_prompt("http://fake", "model", prompt, 32,
                                                  1800, True, "medium")
            self.assertFalse(result["completion_valid"])
            self.assertEqual(result["incomplete"], "finish_reason='length'")
            self.assertNotIn("error", result)


if __name__ == "__main__":
    unittest.main()
