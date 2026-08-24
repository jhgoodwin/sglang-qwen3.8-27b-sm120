import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from bench import c2_c3_campaign as campaign


class CampaignDriverTests(unittest.TestCase):
    def fake_tokenize(self, url, body, timeout):
        text = body["messages"][0]["content"]
        # Simulate a merge boundary: nearby character coordinates are not
        # perfectly monotonic, while the exact server response remains clear.
        n = text.count("-filler")
        return 200, {"input_ids": list(range(max(0, n + (1 if n % 17 == 0 else 0))))}

    def test_exact_builder_retains_calls_and_handles_local_merge(self):
        builder = campaign.ExactPromptBuilder("http://fake", "model", max_calls=240,
                                              tokenize=self.fake_tokenize)
        prompt, proof = builder.build(97)
        self.assertEqual(self.fake_tokenize("", {"messages": [{"content": prompt}]}, 1)[1]["input_ids"].__len__(), 97)
        self.assertEqual(proof["observed"], 97)
        self.assertGreater(len(builder.calls), 1)
        self.assertLessEqual(len(builder.calls), 240)

    def test_plan_is_network_free_and_boundary_disabled(self):
        with tempfile.TemporaryDirectory() as directory, patch("urllib.request.urlopen", side_effect=AssertionError):
            self.assertEqual(campaign.main(["plan", "--profile", "c3", "--artifact-root", directory]), 0)
            value = json.loads((pathlib.Path(directory) / "plan.json").read_text())
        self.assertFalse(value["network_contacted"])
        self.assertFalse(value["optional_boundary"]["enabled"])
        self.assertEqual(value["base_url"], "http://127.0.0.1:11448")
        self.assertEqual([item["id"] for item in value["stages"]], list(campaign.STAGES))

    def test_request_shapes_are_exactly_concurrent_and_sampling_defaults_omitted(self):
        manifest = json.loads(campaign.MANIFEST.read_text())
        spec = campaign.request_shape("E-four-arrival-queue", "c2", 2,
                                      [f"prompt-{index}" for index in range(4)], manifest)
        self.assertEqual(len(spec["requests"]), 4)
        self.assertTrue(all(item["forced_output"] and item["ignore_eos"] for item in spec["requests"]))
        self.assertTrue(all("temperature" not in item and "top_p" not in item and "top_k" not in item
                            for item in spec["requests"]))
        self.assertEqual(spec["requests"][0]["expected_prompt_tokens"], 130048)
        self.assertEqual(campaign.request_shape("B-near-native-prefill", "c2", 1, ["p0", "p1"], manifest,
                                                 model="/models/nvfp4")["model"], "/models/nvfp4")

    def test_request_shape_rejects_shared_prompt_shortcut(self):
        manifest = json.loads(campaign.MANIFEST.read_text())
        with self.assertRaisesRegex(ValueError, "distinct prompt"):
            campaign.request_shape("B-near-native-prefill", "c3", 1, ["shared"], manifest)

    def test_namespaced_exact_prompts_have_distinct_prefixes_and_hashes(self):
        builder = campaign.ExactPromptBuilder("http://fake", "model", max_calls=240,
                                              tokenize=self.fake_tokenize)
        prompts = [builder.build(19, namespace=f"c2/B/r{rep}/q{index}")[0]
                   for rep in (1, 2) for index in (0, 1)]
        self.assertEqual(len({prompt[:64] for prompt in prompts}), len(prompts))
        self.assertEqual(len({campaign._messages_hash(builder._messages(prompt)) for prompt in prompts}),
                         len(prompts))

    def test_multi_token_coarse_unit_requires_suffix_residue_correction(self):
        def tokenize(url, body, timeout):
            text = body["messages"][0]["content"]
            n = text.count("-filler")
            suffix = text.rsplit("-filler", 1)[-1]
            return 200, {"count": n * 3 + (1 if suffix.endswith("a") else 0)}
        builder = campaign.ExactPromptBuilder("http://fake", "model", max_calls=240, tokenize=tokenize)
        prompt, proof = builder.build(97)
        self.assertEqual(proof["observed"], 97)
        self.assertTrue(prompt.endswith("a"))

    def test_repeated_suffix_spans_live_sized_coarse_token_gap(self):
        def tokenize(url, body, timeout):
            text = body["messages"][0]["content"]
            coarse = text.count("-filler") * 20
            residue = text.count(" a")
            return 200, {"count": coarse + residue}
        builder = campaign.ExactPromptBuilder("http://fake", "model", max_calls=240, tokenize=tokenize)
        prompt, proof = builder.build(261117, namespace="c2/B/r1/q0")
        self.assertEqual(tokenize("", {"messages": [{"content": prompt}]}, 1)[1]["count"], 261117)
        self.assertEqual(proof["coarse_token_gap"], 20)
        self.assertGreater(proof["suffix_repetitions"], 2)
        self.assertLessEqual(len(builder.calls), 240)

    def test_short_warmup_is_bounded_and_retains_request_id(self):
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def readline(self):
                if hasattr(self, "done"): return b""
                self.done = True
                return b"data: [DONE]\n\n"
        with tempfile.TemporaryDirectory() as directory, patch("urllib.request.urlopen", return_value=Response()) as opened:
            path = pathlib.Path(directory) / "warmup.json"
            campaign.short_warmup("http://fake", "/models/nvfp4", path)
            value = json.loads(path.read_text())
        self.assertLessEqual(value["request"]["max_tokens"], 8)
        self.assertTrue(value["request_id"].startswith("campaign-warmup-"))
        self.assertEqual(opened.call_args.kwargs["timeout"], 180)

    def test_short_warmup_preserves_partial_stream_on_disconnect(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def readline(self):
                if not hasattr(self, "read"):
                    self.read = True
                    return b"data: partial\n"
                raise ConnectionError("lost")
        with tempfile.TemporaryDirectory() as directory, \
             patch("urllib.request.urlopen", return_value=Response()):
            path = pathlib.Path(directory) / "warmup.json"
            with self.assertRaises(ConnectionError):
                campaign.short_warmup("http://fake", "/model", path)
            value = json.loads(path.read_text())
        self.assertEqual(value["raw_sse"], ["data: partial\n"])
        self.assertIn("ConnectionError", value["error"])

    def test_resume_and_vram_gate(self):
        with self.assertRaisesRegex(RuntimeError, "free VRAM"):
            campaign.free_vram_gate({"gpu_telemetry": [{"free_vram_bytes": 4, "total_vram_bytes": 100}]})
        with self.assertRaisesRegex(RuntimeError, "no measured telemetry"):
            campaign.free_vram_gate({"gpu_telemetry": []})

    def test_cold_cell_exits_for_restart_and_rejects_reused_lifecycle(self):
        class FakeBuilder:
            def __init__(self, *args, **kwargs): self.calls = []
            def _messages(self, text): return [{"role": "user", "content": text}]
            def build(self, target, namespace="default"):
                text = f"{namespace}-exact-{target}"
                return text, {"target": target, "observed": target,
                              "messages_sha256": campaign._messages_hash(self._messages(text))}
            def prove(self, text, target):
                return {"target": target, "observed": target,
                        "messages_sha256": campaign._messages_hash(self._messages(text))}

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            evidence = root / "boot-1-server.json"
            scheduler = root / "boot-1-scheduler.jsonl"
            scheduler.write_text("{}\n")
            evidence.write_text(json.dumps({"observed_server_args": {"model_path": "/model"},
                "launch_provenance": {"container_id": "container-1",
                                      "artifact_reference": str(root / "boot-1-launch.json")}}))
            argv = ["run", "--profile", "c2", "--artifact-root", str(root / "artifacts"),
                    "--server-evidence", str(evidence), "--scheduler-events", str(scheduler),
                    "--gpu", "1"]
            raw = {"gpu_telemetry": [{"free_vram_bytes": 10, "total_vram_bytes": 100}]}
            with patch.object(campaign, "ExactPromptBuilder", FakeBuilder), \
                 patch.object(campaign, "short_warmup", return_value="warmup-1") as warmup, \
                 patch.object(campaign, "_scheduler_has_request", return_value=True), \
                 patch.object(campaign, "preflight_free_vram_gate", return_value=.10) as preflight, \
                 patch.object(campaign.c2_c3_runner, "bootstrap_server_pid",
                              return_value=(7, "pid:7:start_ticks:11")), \
                 patch.object(campaign.c2_c3_runner, "run_concurrent", return_value=raw) as run, \
                 patch.object(campaign.c2_c3_importer, "validate_and_import", return_value={"accepted": True}):
                self.assertEqual(campaign.main(argv), campaign.RESTART_EXIT)
                state = json.loads((root / "artifacts" / "state.json").read_text())
                self.assertEqual(state["status"], "ready_for_restart")
                self.assertIn("A-boot-admission/r1", state["accepted"])
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.kwargs["gpu"], "1")
                preflight.assert_called_once_with("1")
                with self.assertRaisesRegex(SystemExit, "fresh container_id"):
                    campaign.main(argv)
                self.assertEqual(warmup.call_count, 1, "old lifecycle must be rejected before another warmup")

    def test_lifecycle_gate_requires_all_append_only_identity_fields(self):
        previous = {"container_id": "old", "process_identity": "pid:1:start_ticks:2",
                    "server_evidence_path": "/old/server.json", "scheduler_events_path": "/old/events.jsonl",
                    "launch_artifact": "/old/launch.json"}
        state = {"lifecycles": [previous]}
        fresh = {key: "new-" + value for key, value in previous.items()}
        campaign._require_fresh_lifecycle(state, fresh)
        for field in previous:
            reused = dict(fresh); reused[field] = previous[field]
            with self.assertRaisesRegex(RuntimeError, field):
                campaign._require_fresh_lifecycle(state, reused)

    def test_rejected_cell_is_reimported_without_overwriting_or_rerunning(self):
        class FakeBuilder:
            def __init__(self, *args, **kwargs): self.calls = []
            def _messages(self, text): return [{"role": "user", "content": text}]
            def build(self, target, namespace="default"):
                text = f"{namespace}-exact-{target}"
                return text, {"target": target, "observed": target}
            def prove(self, text, target): return {"target": target, "observed": target}

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            evidence = root / "server.json"; scheduler = root / "scheduler.jsonl"
            scheduler.write_text("{}\n")
            evidence.write_text(json.dumps({"observed_server_args": {"model_path": "/model"},
                "launch_provenance": {"container_id": "container-1",
                                      "artifact_reference": str(root / "launch.json")}}))
            argv = ["run", "--profile", "c2", "--artifact-root", str(root / "artifacts"),
                    "--server-evidence", str(evidence), "--scheduler-events", str(scheduler)]
            raw = {"gpu_telemetry": [{"free_vram_bytes": 10, "total_vram_bytes": 100}]}
            imports = [{"accepted": False, "errors": ["first rejection"]}, {"accepted": True}]
            with patch.object(campaign, "ExactPromptBuilder", FakeBuilder), \
                 patch.object(campaign, "short_warmup", return_value="warmup"), \
                 patch.object(campaign, "_scheduler_has_request", return_value=True), \
                 patch.object(campaign, "preflight_free_vram_gate", return_value=.10), \
                 patch.object(campaign.c2_c3_runner, "bootstrap_server_pid",
                              return_value=(7, "pid:7:start_ticks:11")), \
                 patch.object(campaign.c2_c3_runner, "run_concurrent", return_value=raw) as run, \
                 patch.object(campaign.c2_c3_importer, "validate_and_import", side_effect=imports):
                with self.assertRaisesRegex(SystemExit, "importer rejected"):
                    campaign.main(argv)
                first = root / "artifacts/raw/c2-A-boot-admission-r1.json"
                first_contents = first.read_text()
                self.assertEqual(campaign.main(argv), campaign.RESTART_EXIT)
            retry = root / "artifacts/reimports/c2-A-boot-admission-r1-attempt2-reimport.json"
            self.assertTrue(retry.is_file())
            self.assertEqual(first.read_text(), first_contents)
            self.assertEqual(run.call_count, 1, "a corrected importer must not repeat completed GPU work")
            state = json.loads((root / "artifacts/state.json").read_text())
            self.assertEqual(state["accepted"]["A-boot-admission/r1"], str(retry))
            self.assertFalse((root / "artifacts/raw/c2-A-boot-admission-r1-attempt2.json").exists())

    def test_b_reserve_waiver_is_explicit_recorded_and_dependency_only(self):
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "b.json"
            source.write_text(json.dumps({"raw": {"stage": "B-near-native-prefill", "requests": []}}))
            state = {"accepted": {f"A-boot-admission/r{rep}": "ok" for rep in range(1, 4)},
                     "failures": [{"key": "B-near-native-prefill/r1", "artifact": str(source)}]}
            self.assertEqual(campaign._next_cell(state), ("B-near-native-prefill", 1),
                             "B must not be skipped before the explicit waiver")
            lifecycle = {}
            facts = {"prompt_tokens": 261120, "observed_completion_tokens": 1022,
                     "reserved_boundary_tokens": 2}
            with patch.object(campaign, "_validate_b_two_token_reserve", return_value=facts):
                campaign._apply_b_two_token_reserve_waiver(state, lifecycle)
            waiver = state["dependency_waivers"]["B-near-native-prefill"]
            self.assertFalse(waiver["exact_B_success"])
            self.assertEqual(waiver["scope"], "dependency-only")
            self.assertEqual(waiver["observed"], facts)
            self.assertEqual(campaign._next_cell(state), ("C-max-output-decode", 1))

    def test_preflight_vram_gate_parses_and_rejects(self):
        completed = type("Completed", (), {"stdout": "6000, 100000\n"})()
        with patch.object(campaign.subprocess, "run", return_value=completed):
            self.assertEqual(campaign.preflight_free_vram_gate(), .06)
        completed.stdout = "4999, 100000\n"
        with patch.object(campaign.subprocess, "run", return_value=completed), \
             self.assertRaisesRegex(RuntimeError, "before submission"):
            campaign.preflight_free_vram_gate()

    def test_profiles_have_independent_state_and_default_ports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            c2_root, c3_root = root / "c2", root / "c3"
            self.assertEqual(campaign.main(["plan", "--profile", "c2", "--artifact-root", str(c2_root)]), 0)
            self.assertEqual(campaign.main(["plan", "--profile", "c3", "--artifact-root", str(c3_root)]), 0)
            self.assertEqual(json.loads((c2_root / "plan.json").read_text())["base_url"],
                             "http://127.0.0.1:11447")
            self.assertEqual(json.loads((c3_root / "plan.json").read_text())["base_url"],
                             "http://127.0.0.1:11448")
            c2_state = campaign._load_state(c2_root / "state.json")
            c3_state = campaign._load_state(c3_root / "state.json")
            c2_state["accepted"]["A-boot-admission/r1"] = "c2-only"
            self.assertNotIn("A-boot-admission/r1", c3_state["accepted"])


if __name__ == "__main__":
    unittest.main()
