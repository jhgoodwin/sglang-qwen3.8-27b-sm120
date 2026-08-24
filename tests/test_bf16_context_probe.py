import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from bench import bf16_context_probe as probe


class BF16ContextProbeTests(unittest.TestCase):
    def good_info(self):
        return {"model_path": "/models/319f741cce68d7914884900c138a1fbb70a42f30",
                "version": "0.0.0.dev1+g5f55db35e",
                "context_length": 262144, "kv_cache_dtype": "bf16",
                "max_running_requests": 2, "max_total_num_tokens": 512000}

    def test_file_path_entrypoint_loads_repository_package(self):
        script = pathlib.Path(__file__).resolve().parents[1] / "bench" / "bf16_context_probe.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--prompt-tokens", result.stdout)

    def test_capacity_arithmetic_and_dtype_reject(self):
        evidence = probe.capacity_preflight(self.good_info())
        self.assertEqual(evidence["required_token_capacity"], 512000)
        bad = self.good_info(); bad["max_total_num_tokens"] = 511999
        with self.assertRaisesRegex(ValueError, "token_capacity"):
            probe.capacity_preflight(bad)
        bad = self.good_info(); bad["kv_cache_dtype"] = "fp8"
        with self.assertRaisesRegex(ValueError, "BF16"):
            probe.capacity_preflight(bad)

    def test_request_shape_and_exact_validation(self):
        body = probe.build_request("m", "prompt", "rid")
        self.assertEqual(body["max_tokens"], 6000)
        self.assertTrue(body["ignore_eos"] and body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertNotIn("request_id", body)
        result = {"usage": {"prompt_tokens": 250000, "completion_tokens": 6000,
                             "total_tokens": 256000}, "finish_reason": "length"}
        self.assertEqual(probe.validate_result(result), [])
        result["usage"]["total_tokens"] = 1
        self.assertTrue(probe.validate_result(result))

    def test_partial_stream_and_http_error_are_retained(self):
        result = {"request_id": "r", "prompt": "p", "raw_sse": [], "events": [],
                  "timestamps": {}, "usage": None, "error": "disconnect"}
        self.assertIn("disconnect", probe.validate_result(result))
        result["raw_sse"].append("data: partial\n")
        self.assertEqual(result["raw_sse"], ["data: partial\n"])

    def test_unavailable_capacity_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unavailable"):
            probe.capacity_preflight({"kv_cache_dtype": "bf16"})

    def test_main_preflight_failure_writes_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "probe.json"
            args = ["--base-url", "http://fake", "--output", str(output)]
            info = {"model_path": "/models/319f741cce68d7914884900c138a1fbb70a42f30",
                    "version": "0.0.0.dev1+g5f55db35e",
                    "kv_cache_dtype": "bf16"}
            with patch.object(probe, "_json_request", return_value=(200, "application/json", info)):
                self.assertEqual(probe.main(args), 1)
            document = json.loads(output.read_text())
            self.assertEqual(document["status"], "fail")
            self.assertIn("unavailable", document["failure"])

    def test_identity_conflict_rejected(self):
        bad = self.good_info(); bad["version"] = "0.0.0+g0000000"
        with self.assertRaisesRegex(ValueError, "runtime version"):
            probe.capacity_preflight(bad)

    def test_real_server_info_shape_and_internal_conflict(self):
        info = self.good_info()
        info["internal_states"] = [{"context_length": 262144,
                                     "max_running_requests": 2,
                                     "max_total_num_tokens": 512000}]
        self.assertEqual(probe.capacity_preflight(info)["context_length"], 262144)
        info["internal_states"][0]["max_running_requests"] = 3
        with self.assertRaisesRegex(ValueError, "internal_states"):
            probe.capacity_preflight(info)


if __name__ == "__main__":
    unittest.main()
