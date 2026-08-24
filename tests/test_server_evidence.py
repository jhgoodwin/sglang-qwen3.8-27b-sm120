import json
import os
import pathlib
import tempfile
import unittest

from bench import c2_c3_importer as importer
from bench import c2_c3_runner as runner
from bench import server_evidence

MODEL_REV = "319f741cce68d7914884900c138a1fbb70a42f30"
DRAFT_REV = "dedf8df68adfb1afeaf7b7480c0a0243108177b4"
MODEL_CONTAINER = f"/models/Qwen3.8-27B-cache/snapshots/{MODEL_REV}"
DRAFT_CONTAINER = f"/models/Qwen3.8-27B-DFlash2-cache/snapshots/{DRAFT_REV}"


def _command(profile="c2"):
    slots, concurrency = ((8, 2) if profile == "c2" else (12, 3))
    return ["python3", "-m", "sglang.launch_server", "--model-path", MODEL_CONTAINER,
            "--speculative-draft-model-path", DRAFT_CONTAINER, "--context-length", "262144",
            "--tp-size", "1", "--kv-cache-dtype", "fp8_e4m3", "--attention-backend", "flashinfer",
            "--chunked-prefill-size", "2048", "--mamba-ssm-dtype", "float32",
            "--mem-fraction-static", "0.85", "--mamba-radix-cache-strategy", "extra_buffer_lazy",
            "--speculative-algorithm", "DFLASH", "--speculative-num-draft-tokens", "8",
            "--max-running-requests", str(concurrency), "--max-mamba-cache-size", str(slots)]


def _inspect(profile="c2"):
    image = importer.EXPECTED_IDENTITIES["image_digest"]
    return {"Id": "container-id", "Name": "/qwen3.8-27b-sglang", "Image": image,
            "Config": {"Image": f"qwen38-c2c3-evidence@{image}", "Cmd": _command(profile),
                       "Labels": {"org.opencontainers.image.revision": importer.EXPECTED_IDENTITIES["source_revision"]}},
            "Mounts": [
                {"Source": "/data/models/models--RadixArk--Qwen3.8-27B-NVFP4",
                 "Destination": "/models/Qwen3.8-27B-cache"},
                {"Source": "/data/models/models--incoai--Qwen3.8-27B-DFlash2",
                 "Destination": "/models/Qwen3.8-27B-DFlash2-cache"},
            ]}


def _provenance(profile="c2"):
    value = server_evidence.build_launch_provenance(
        _inspect(profile), profile=profile, hardware_identity="GPU-uuid", raw_reference="raw-inspect.json")
    value["artifact_reference"] = "launch-provenance.json"
    return value


def _response(profile="c2"):
    slots, concurrency = ((8, 2) if profile == "c2" else (12, 3))
    # This is the exact endpoint organization: flattened dataclass fields plus
    # a list of scheduler internal-state dictionaries.
    return {"model_path": MODEL_CONTAINER, "speculative_draft_model_path": DRAFT_CONTAINER,
            "context_length": 262144, "tp_size": 1, "kv_cache_dtype": "fp8_e4m3",
            "attention_backend": "flashinfer", "chunked_prefill_size": 2048,
            "mamba_ssm_dtype": "float32", "mem_fraction_static": 0.85,
            "mamba_radix_cache_strategy": "extra_buffer_lazy", "speculative_algorithm": "DFLASH",
            "speculative_num_draft_tokens": 8, "max_running_requests": concurrency,
            "max_mamba_cache_size": slots, "version": "0.5.9",
            "internal_states": [{"effective_max_running_requests_per_dp": concurrency,
                "c2c3_evidence": {"resolved_capacity": {"max_running_requests": concurrency,
                    "context_length": 262144, "max_total_num_tokens": 262144, "tp_size": 1,
                    "max_mamba_cache_size": slots}, "memory_pools": {
                    "source": "runtime_introspection", "kv_cache": {"bytes": 10, "dtype": "fp8_e4m3"},
                    "mamba_state_cache": {"bytes": 20, "dtype": "float32", "slots": slots},
                    "dflash_intermediate": {"bytes": 30, "states": 8}},
                    "cuda_graphs": {"source": "runtime_introspection", "enabled": True,
                                    "captured_batch_sizes": [1, concurrency], "memory_bytes": 40}}}]}


class ServerEvidenceTests(unittest.TestCase):
    def test_collector_consumes_flattened_response_and_separates_metadata(self):
        result = server_evidence.build_evidence(
            _response(), provenance=_provenance(), profile="c2", endpoint="http://127.0.0.1:11447",
            raw_reference="raw-server-info.json")
        self.assertEqual(result["observed_server_args"]["context_length"], 262144)
        self.assertNotIn("server_args", result)
        self.assertNotIn("max_output_tokens", result["resolved_capacity"])
        self.assertEqual(result["campaign_request_limits"], {"max_output_tokens": 131072})
        self.assertEqual(result["launch_metadata"]["planned_port"], 11447)
        self.assertEqual(result["identities"]["image_digest"], importer.EXPECTED_IDENTITIES["image_digest"])

    def test_collector_rejects_absent_or_mismatched_flattened_runtime_fields(self):
        for mutate in (
            lambda value: value.pop("context_length"),
            lambda value: value.update(attention_backend="triton"),
            lambda value: value["internal_states"].append({}),
            lambda value: value["internal_states"][0]["c2c3_evidence"]["memory_pools"].pop("kv_cache"),
        ):
            with self.subTest(mutate=mutate):
                value = _response(); mutate(value)
                with self.assertRaises(ValueError):
                    server_evidence.build_evidence(value, provenance=_provenance(), profile="c2",
                                                   endpoint="http://127.0.0.1:11447", raw_reference="raw.json")

    def test_provenance_rejects_image_revision_mount_command_and_gpu_drift(self):
        mutations = [
            lambda value: value.update(Image="sha256:" + "0" * 64),
            lambda value: value["Config"]["Labels"].update({"org.opencontainers.image.revision": "0" * 40}),
            lambda value: value["Mounts"][0].update(Source="/wrong"),
            lambda value: value["Config"]["Cmd"].extend(["--mamba-full-memory-ratio", "0.22"]),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value = _inspect(); mutate(value)
                with self.assertRaises(ValueError):
                    server_evidence.build_launch_provenance(value, profile="c2", hardware_identity="GPU-uuid",
                                                            raw_reference="raw.json")
        with self.assertRaises(ValueError):
            server_evidence.build_launch_provenance(_inspect(), profile="c2", hardware_identity="unknown",
                                                    raw_reference="raw.json")

    def test_join_rejects_provenance_identity_drift(self):
        for key, value in (("image_digest", "sha256:" + "0" * 64),
                           ("source_revision", "0" * 40), ("hardware_identity", "unknown"),
                           ("container_name", "wrong-service")):
            provenance = _provenance(); provenance[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                server_evidence.build_evidence(_response(), provenance=provenance, profile="c2",
                                               endpoint="http://127.0.0.1:11447", raw_reference="raw.json")

    def test_pid_bootstrap_accepts_current_identity_and_rejects_stale(self):
        stat = pathlib.Path(f"/proc/{os.getpid()}/stat").read_text(); tail = stat[stat.rfind(")") + 2:].split()
        identity = f"pid:{os.getpid()}:start_ticks:{tail[19]}"
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "events.jsonl"
            path.write_text(json.dumps({"schema": "qwen38.server-scheduler-event", "source": "server_scheduler",
                "server_process_id": identity}) + "\n")
            self.assertEqual(runner.bootstrap_server_pid(path)[0], os.getpid())
            path.write_text(json.dumps({"schema": "qwen38.server-scheduler-event", "source": "server_scheduler",
                "server_process_id": f"pid:{os.getpid()}:start_ticks:1"}) + "\n")
            with self.assertRaises(ValueError): runner.bootstrap_server_pid(path)

    def test_pid_bootstrap_rejects_malformed_and_multiple_identities(self):
        stat = pathlib.Path(f"/proc/{os.getpid()}/stat").read_text(); tail = stat[stat.rfind(")") + 2:].split()
        identity = f"pid:{os.getpid()}:start_ticks:{tail[19]}"
        row = {"schema": "qwen38.server-scheduler-event", "source": "server_scheduler",
               "server_process_id": identity}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "events.jsonl"
            path.write_text("not-json\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "malformed scheduler evidence"):
                runner.bootstrap_server_pid(path)
            other = {**row, "server_process_id": f"pid:{os.getpid() + 1}:start_ticks:1"}
            path.write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n")
            with self.assertRaisesRegex(ValueError, "multiple process identities"):
                runner.bootstrap_server_pid(path)


if __name__ == "__main__":
    unittest.main()
