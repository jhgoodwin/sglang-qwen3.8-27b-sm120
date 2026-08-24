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
        n = text.count(" qwen38-campaign-filler")
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
        self.assertEqual([item["id"] for item in value["stages"]], list(campaign.STAGES))

    def test_request_shapes_are_exactly_concurrent_and_sampling_defaults_omitted(self):
        manifest = json.loads(campaign.MANIFEST.read_text())
        spec = campaign.request_shape("E-four-arrival-queue", "c2", 2, ["prompt"], manifest)
        self.assertEqual(len(spec["requests"]), 4)
        self.assertTrue(all(item["forced_output"] and item["ignore_eos"] for item in spec["requests"]))
        self.assertTrue(all("temperature" not in item and "top_p" not in item and "top_k" not in item
                            for item in spec["requests"]))
        self.assertEqual(spec["requests"][0]["expected_prompt_tokens"], 130048)
        self.assertEqual(campaign.request_shape("B-near-native-prefill", "c2", 1, ["p"], manifest,
                                                 model="/models/nvfp4")["model"], "/models/nvfp4")

    def test_multi_token_coarse_unit_requires_suffix_residue_correction(self):
        def tokenize(url, body, timeout):
            text = body["messages"][0]["content"]
            n = text.count(" qwen38-campaign-filler")
            suffix = text.rsplit(" qwen38-campaign-filler", 1)[-1]
            return 200, {"count": n * 3 + (1 if suffix.endswith("a") else 0)}
        builder = campaign.ExactPromptBuilder("http://fake", "model", max_calls=240, tokenize=tokenize)
        prompt, proof = builder.build(97)
        self.assertEqual(proof["observed"], 97)
        self.assertTrue(prompt.endswith("a"))

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

    def test_resume_and_vram_gate(self):
        with self.assertRaisesRegex(RuntimeError, "free VRAM"):
            campaign.free_vram_gate({"gpu_telemetry": [{"free_vram_bytes": 4, "total_vram_bytes": 100}]})


if __name__ == "__main__":
    unittest.main()
