import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bench import c2_c3_runner


ROOT = Path(__file__).parents[1]


class C2C3CampaignContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "bench/c2-c3-native-context-campaign.json").read_text())

    def test_queue_manifest_is_queued_and_identity_locked(self):
        m = self.manifest
        self.assertEqual(m["schema"], "qwen38.c2-c3-native-context-campaign")
        self.assertEqual(m["version"], 1)
        self.assertEqual(m["status"], "queued")
        self.assertEqual(m["runtime"]["gpu"], "0")
        self.assertEqual(m["runtime"]["tp"], 1)
        for key in ("base_profile", "model", "draft_model", "runtime_identity", "recipe_identity"):
            self.assertIn("#", m["runtime"][key])
        self.assertEqual(m["runtime"]["context_length"], 262144)
        self.assertEqual(m["runtime"]["max_output_tokens"], 131072)

    def test_initial_profiles_pin_state_without_ratio(self):
        m = self.manifest
        self.assertTrue(m["sizing_rules"]["initial_profiles_omit_mamba_full_memory_ratio"])
        encoded = json.dumps(m["profiles"])
        self.assertNotIn("mamba_full_memory_ratio", encoded)
        self.assertEqual([(p["id"], p["max_running_requests"], p["max_mamba_cache_size"])
                          for p in m["profiles"]], [("c2", 2, 8), ("c3", 3, 12)])
        self.assertTrue(m["sizing_rules"]["ratio_comparison"]["mutually_exclusive_with_explicit_state_pins"])

    def test_stage_order_shapes_repetitions_and_queue(self):
        m = self.manifest
        stages = m["stages"]
        self.assertEqual([s["id"] for s in stages], [
            "A-boot-admission", "B-near-native-prefill", "C-max-output-decode",
            "D-combined-boundary-safe", "E-four-arrival-queue"])
        self.assertTrue(all(s["repetitions"] >= 3 for s in stages))
        by_id = {s["id"]: s for s in stages}
        self.assertEqual((by_id["B-near-native-prefill"]["input_tokens"], by_id["B-near-native-prefill"]["output_tokens"], by_id["B-near-native-prefill"]["combined_tokens"]), (261120, 1024, 262144))
        self.assertEqual((by_id["C-max-output-decode"]["input_tokens"], by_id["C-max-output-decode"]["output_tokens"]), (1024, 131072))
        self.assertEqual((by_id["D-combined-boundary-safe"]["input_tokens"], by_id["D-combined-boundary-safe"]["output_tokens"], by_id["D-combined-boundary-safe"]["combined_tokens"]), (130048, 131072, 261120))
        self.assertEqual(by_id["D-combined-boundary-safe"]["transport_template_margin_tokens"], 1024)
        self.assertEqual(by_id["E-four-arrival-queue"]["arrivals"], 4)
        self.assertEqual(by_id["E-four-arrival-queue"]["request_shape"]["output_tokens"], 131072)
        self.assertEqual(by_id["E-four-arrival-queue"]["requested_output_tokens_per_rep"], 524288)
        self.assertEqual(by_id["E-four-arrival-queue"]["processes"], "one server process; do not mix processes")

    def test_dependencies_and_runner_gate(self):
        m = self.manifest
        self.assertEqual(m["execution_prerequisite"]["status"], "blocking")
        self.assertIn("sequential", m["execution_prerequisite"]["reason"])
        stages = m["stages"]
        for previous, current in zip(stages, stages[1:]):
            self.assertEqual(current["depends_on"], [previous["id"]])
            self.assertTrue(current["fail_fast"])
        self.assertTrue(m["measurement_contract"]["streaming"] if "streaming" in m["measurement_contract"] else m["runtime"]["invariants"]["streaming"])
        self.assertEqual(m["runtime"]["invariants"]["sampling"], "server_default")
        self.assertEqual(m["measurement_contract"]["natural_sampling"], "omit temperature/top_p/top_k overrides")
        self.assertNotIn("reasoning", m["measurement_contract"]["natural_sampling"])
        self.assertEqual(m["measurement_contract"]["reasoning_effort"], "medium for natural and operational cells")

    def test_no_phase7_mutation(self):
        phase7 = json.loads((ROOT / "bench/phase7-minimum.json").read_text())
        self.assertEqual(phase7["schema"], "qwen38.phase7")
        self.assertEqual(phase7["version"], 1)

    def test_concurrent_schema_is_distinct_and_example_dry_run_is_runnable(self):
        schema = json.loads((ROOT / "bench/c2-c3-run-schema.json").read_text())
        self.assertEqual(schema["properties"]["schema"]["const"], "qwen38.c2-c3-concurrent-run")
        self.assertNotEqual(schema["properties"]["schema"]["const"], "qwen38.phase7")
        example = ROOT / "bench/c2-c3-run-spec.example.json"
        with TemporaryDirectory() as directory:
            output = Path(directory) / "dry.json"
            self.assertEqual(c2_c3_runner.main(["dry-run", "--spec", str(example),
                                               "--output", str(output)]), 0)
            document = json.loads(output.read_text())
        self.assertEqual(document["mode"], "dry-run")
        self.assertEqual(document["barrier_parties"], 3)


if __name__ == "__main__":
    unittest.main()
