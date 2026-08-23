import copy
import json
import tempfile
import unittest
from pathlib import Path

from bench.benchmark_contract import (
    CONTEXT_LIMIT, build_manifest, evaluate_gates, matched_comparison,
    analyze_cell, validate_manifest, validate_result, main,
)


def result(run_id="r1"):
    metrics = {name: 1.0 for name in build_manifest()["required_metrics"]}
    return {"schema": "qwen38.phase7", "run_id": run_id, "cell_id": "decode-8192-1024-c1",
            "model_snapshot": "model-snapshot-abc", "image_digest": "sha256:" + "a" * 64,
            "source_revision": "abc", "dependency_lock": "sha256:deps",
            "hardware_identity": "sha256:hardware", "timestamps": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T00:00:01Z"},
            "process": {"pid": 1, "unchanged": True}, "server_args": ["--context-length", "262144"],
            "resolved_capacity": {"max_running_requests": 1}, "raw_run": "raw/r1.json",
            "cache_state": "cold",
            "occupancy": {"expected": 1, "observed": 1}, "interval": {"aligned": True, "queueing": False},
            "request_errors": 0, "restarts": 0, "ooms": 0, "malformed_responses": 0,
            "clamped": False, "metrics": metrics}


class BenchmarkContractTests(unittest.TestCase):
    def test_panel_is_complete_and_context_safe(self):
        manifest = build_manifest()
        self.assertEqual(validate_manifest(manifest), [])
        self.assertEqual(len(manifest["cells"]), 75)
        self.assertEqual({c["concurrency"] for c in manifest["cells"] if c["kind"] == "decode"}, {1, 2, 4})
        engine = [c for c in manifest["cells"] if c["kind"] != "production"]
        self.assertTrue(all(c["input_tokens"] + c["output_tokens"] <= CONTEXT_LIMIT for c in engine))
        bad = copy.deepcopy(manifest); bad["cells"][0]["output_tokens"] = CONTEXT_LIMIT
        self.assertTrue(any("exceeds context" in e for e in validate_manifest(bad)))
        missing = copy.deepcopy(manifest); missing["cells"].pop()
        self.assertTrue(any("incomplete" in e for e in validate_manifest(missing)))
        duplicate = copy.deepcopy(manifest); duplicate["cells"][-1]["id"] = duplicate["cells"][0]["id"]
        self.assertTrue(any("duplicate" in e for e in validate_manifest(duplicate)))

    def test_manifest_is_semantically_exact_and_static_is_canonical(self):
        manifest = build_manifest()
        mutated = copy.deepcopy(manifest)
        mutated["cells"][0]["input_tokens"] = 1
        self.assertTrue(any("exactly match" in e for e in validate_manifest(mutated)))
        mutated = copy.deepcopy(manifest)
        mutated["cells"][-1]["requests"][0]["input_tokens"] = 1
        self.assertTrue(any("exactly match" in e for e in validate_manifest(mutated)))
        static = json.loads((Path(__file__).parents[1] / "bench/phase7-minimum.json").read_text())
        self.assertEqual(static, manifest)

    def test_result_rejection_and_missing_measurements(self):
        r = result()
        self.assertEqual(validate_result(r), [])
        r["metrics"]["power_w"] = None; r["interval"]["aligned"] = False; r["request_errors"] = 1
        errors = validate_result(r)
        self.assertIn("missing measurement: power_w", errors)
        self.assertIn("client/server intervals are not aligned", errors)
        self.assertIn("request errors present", errors)
        self.assertEqual(analyze_cell([r])["status"], "unresolved")

    def test_quantiles_require_enough_samples(self):
        summary = analyze_cell([result(str(i)) for i in range(5)])
        metric = summary["metrics"]["ttft_ms"]
        self.assertEqual(metric["median"], 1.0)
        self.assertIsNone(metric["p95"]); self.assertIsNone(metric["p99"])

    def test_result_shape_and_mixed_cells_are_rejected(self):
        r = result()
        r["resolved_capacity"]["max_running_requests"] = 0
        self.assertTrue(any("positive" in e for e in validate_result(r)))
        other = result("two")
        other["cell_id"] = "decode-8192-1024-c2"
        other["occupancy"] = {"expected": 2, "observed": 2}
        self.assertEqual(analyze_cell([result(), other])["status"], "unresolved")

    def test_zero_error_evidence_is_required(self):
        for key in ("request_errors", "restarts", "ooms", "malformed_responses", "clamped"):
            candidate = result()
            del candidate[key]
            self.assertTrue(any(f"missing {key}" in error for error in validate_result(candidate)), key)
        candidate = result()
        del candidate["interval"]["queueing"]
        self.assertTrue(any("aligned interval required" in error for error in validate_result(candidate)))

    def test_gates_pass_fail_and_unresolved(self):
        passing = {"free_vram_fraction": .06, "max_itl_ms": 900, "itl_p99_ms": 900, "mixed_itl_p99_ms": 100,
                   "isolated_itl_p99_ms": 80, "request_errors": 0, "restarts": 0, "ooms": 0,
                   "malformed_responses": 0, "clamped": False}
        self.assertEqual(evaluate_gates(passing)["overall"], "pass")
        failing = dict(passing); failing["itl_p99_ms"] = 900; failing["max_itl_ms"] = 1200
        self.assertEqual(evaluate_gates(failing)["overall"], "fail")
        unresolved = dict(passing); unresolved.pop("free_vram_fraction")
        self.assertEqual(evaluate_gates(unresolved)["overall"], "unresolved")

    def test_matched_comparison_math_and_cli(self):
        comparison = matched_comparison([100, 200], [90, 220])
        self.assertEqual(comparison["absolute_deltas"], [-10, 20])
        self.assertAlmostEqual(comparison["median_percentage_delta"], 0)
        self.assertEqual(matched_comparison([1], [1, 2])["status"], "unresolved")
        self.assertEqual(matched_comparison([0, 0], [0, 0])["median_percentage_delta"], 0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "manifest.json"; main(["manifest", str(path)])
            self.assertEqual(json.loads(path.read_text())["schema"], "qwen38.phase7")


if __name__ == "__main__": unittest.main()
