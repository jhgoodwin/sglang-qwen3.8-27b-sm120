import copy
import datetime as dt
import io
import json
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from bench import c2_c3_importer as importer
from bench import c2_c3_runner as runner


BASE = dt.datetime(2026, 8, 24, tzinfo=dt.timezone.utc)


def iso(seconds):
    return (BASE + dt.timedelta(seconds=seconds)).isoformat()


def stamp(seconds):
    return {"utc": iso(seconds), "monotonic_s": 100 + seconds}


def spec(count=2):
    return {"schema": runner.SPEC_SCHEMA, "version": 1, "run_id": "run-1", "profile": "c2",
            "stage": "B-near-native-prefill", "base_url": "http://127.0.0.1:11447", "model": "model",
            "requests": [{"client_request_id": f"r{i}", "prompt": f"prompt {i}",
                          "max_tokens": 2, "expected_prompt_tokens": 3,
                          "forced_output": True, "ignore_eos": True}
                         for i in range(count)]}


def token_event(token, at):
    return {"utc": iso(at), "monotonic_s": 100 + at, "raw_data": "data",
            "parsed": {"token_ids": [token], "token_timestamps_s": [100 + at],
                       "choices": [{"delta": {"content": chr(96 + token)}, "finish_reason": None}]}}


def raw_run(count=2, capacity=2):
    profile = "c3" if capacity == 3 else "c2"
    mamba_slots = 12 if profile == "c3" else 8
    planned_port = 11448 if profile == "c3" else 11447
    requests = []
    scheduler = []
    for i in range(count):
        rid = f"r{i}"
        wave = i // capacity
        admitted_at = .04 + i * .02 if wave == 0 else 2.04 + (i - capacity) * .02
        first_at = .2 + i * .001 if wave == 0 else 2.2 + (i - capacity) * .001
        completed_at = 2 + i * .001 if wave == 0 else 4 + (i - capacity) * .001
        request = {"client_request_id": rid, "forced_output": True, "requested_output_tokens": 2,
            "expected_prompt_tokens": 3,
            "timestamps": {"arrival": stamp(i * .001), "submission": stamp(.01 + i * .001),
                "headers": stamp(.1 + i * .001), "first_event": stamp(first_at),
                "first_token": stamp(first_at), "completion": stamp(completed_at)},
            "http_status": 200, "response_headers": {"x-any": "not-admission"},
            "raw_sse": [{**stamp(.2), "line": "data: one\n"}, {**stamp(.3), "line": "data: two\n"}],
            "events": [token_event(1, first_at), token_event(2, first_at + .1)],
            "content": "ab", "reasoning_content": "", "finish_reason": "length",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2,
                      "completion_tokens_details": {"reasoning_tokens": 0}},
            "failure_bucket": None, "error": None}
        requests.append(request)
        queued = i >= capacity
        position = i % capacity
        wave_size = min(capacity, count - wave * capacity)
        event_rows = ([('queued', .09 + (i - capacity) * .001, capacity, i - capacity + 1)] if queued else []) + [
            ('admitted', admitted_at, position + 1, max(0, count - i - 1) if queued else 0),
            ('started', admitted_at + .01, position + 1, max(0, count - i - 1) if queued else 0),
            ('completed', completed_at - .1, max(0, wave_size - position - 1),
             max(0, count - capacity) if i < capacity else 0)]
        for name, at, running, queue_count in event_rows:
            scheduler.append({"collector_timestamp": stamp(at), "event": {
                "schema": importer.SCHEDULER_SCHEMA, "source": "server_scheduler",
                "timestamp": iso(at), "event": name, "client_request_id": rid,
                "server_process_id": "pid:7:start_ticks:9", "running": running, "queued": queue_count}})
    model_path = "/models/Qwen3.8-27B-cache/snapshots/319f741cce68d7914884900c138a1fbb70a42f30"
    draft_path = "/models/Qwen3.8-27B-DFlash2-cache/snapshots/dedf8df68adfb1afeaf7b7480c0a0243108177b4"
    container_command = ["python3", "-m", "sglang.launch_server", "--model-path", model_path,
        "--speculative-draft-model-path", draft_path, "--context-length", "262144", "--tp-size", "1",
        "--kv-cache-dtype", "fp8_e4m3", "--attention-backend", "flashinfer",
        "--chunked-prefill-size", "2048", "--mamba-ssm-dtype", "float32",
        "--mem-fraction-static", "0.85", "--mamba-radix-cache-strategy", "extra_buffer_lazy",
        "--speculative-algorithm", "DFLASH", "--speculative-num-draft-tokens", "8",
        "--max-running-requests", str(capacity), "--max-mamba-cache-size", str(mamba_slots)]
    return {"schema": runner.SCHEMA, "version": 1, "run_id": "run-1", "profile": profile,
        "stage": "B-near-native-prefill", "repetition": 1,
        "timestamps": {"collector_start": stamp(-1), "barrier_release": stamp(0), "run_end": stamp(3)},
        "request_count": count, "requests": requests,
        "server": {"observed_server_args": {
                "model_path": model_path, "speculative_draft_model_path": draft_path,
                "context_length": 262144, "tp_size": 1, "kv_cache_dtype": "fp8_e4m3",
                "attention_backend": "flashinfer", "chunked_prefill_size": 2048,
                "mamba_ssm_dtype": "float32", "mem_fraction_static": 0.85,
                "mamba_radix_cache_strategy": "extra_buffer_lazy", "speculative_algorithm": "DFLASH",
                "speculative_num_draft_tokens": 8, "max_running_requests": capacity,
                "max_mamba_cache_size": mamba_slots},
            "identities": {"image_digest": importer.EXPECTED_IDENTITIES["image_digest"],
                "source_revision": importer.EXPECTED_IDENTITIES["source_revision"],
                "model_snapshot": importer.EXPECTED_IDENTITIES["model_snapshot"],
                "draft_model_snapshot": importer.EXPECTED_IDENTITIES["draft_model_snapshot"],
                "recipe_identity": importer.EXPECTED_IDENTITIES["recipe_identity"],
                "hardware_identity": "GPU-uuid"},
            "resolved_capacity": {"max_running_requests": capacity, "context_length": 262144,
                "max_total_num_tokens": 262144, "tp_size": 1, "max_mamba_cache_size": mamba_slots},
            "campaign_request_limits": {"max_output_tokens": 131072},
            "launch_metadata": {"planned_port": planned_port, "observed_endpoint": f"http://127.0.0.1:{planned_port}"},
            "launch_provenance": {"source": "docker_inspect", "artifact_reference": "launch.json",
                "image_digest": importer.EXPECTED_IDENTITIES["image_digest"],
                "source_revision": importer.EXPECTED_IDENTITIES["source_revision"],
                "container_name": "qwen3.8-27b-sglang", "container_command": container_command},
            "memory_pools": {"source": "server_info",
                "kv_cache": {"bytes": 100, "dtype": "fp8_e4m3"},
                "mamba_state_cache": {"bytes": 200, "dtype": "float32", "slots": mamba_slots},
                "dflash_intermediate": {"bytes": 300, "states": 8}},
            "cuda_graphs": {"source": "server_log", "enabled": True,
                            "captured_batch_sizes": [1, capacity], "memory_bytes": 400},
            "process": {"restart_count": 0, "sampling_interval_s": 1,
                "samples": [{"timestamp": iso(-1), "server_process_id": "pid:7:start_ticks:9"},
                            {"timestamp": iso(1), "server_process_id": "pid:7:start_ticks:9"},
                            {"timestamp": iso(3), "server_process_id": "pid:7:start_ticks:9"},
                            {"timestamp": iso(5), "server_process_id": "pid:7:start_ticks:9"}]}},
        "scheduler_evidence": scheduler,
        "gpu_telemetry": [{**stamp(-1), "source": "nvidia-smi", "gpu": "0",
            "total_vram_bytes": 1000, "free_vram_bytes": 100, "gpu_utilization_pct": 0,
            "power_w": 50, "temperature_c": 30},
            {**stamp(1), "source": "nvidia-smi", "gpu": "0", "total_vram_bytes": 1000,
             "free_vram_bytes": 80, "gpu_utilization_pct": 90, "power_w": 400, "temperature_c": 60},
            {**stamp(3), "source": "nvidia-smi", "gpu": "0", "total_vram_bytes": 1000,
             "free_vram_bytes": 60, "gpu_utilization_pct": 80, "power_w": 390, "temperature_c": 61},
            {**stamp(5), "source": "nvidia-smi", "gpu": "0", "total_vram_bytes": 1000,
             "free_vram_bytes": 70, "gpu_utilization_pct": 10, "power_w": 80, "temperature_c": 40}],
        "collector": {"gpu_interval_s": 1, "scheduler_source": "server.jsonl",
                      "scheduler_source_contract": "server-generated request-correlated JSONL"},
        "boundary_proof": None}


class FakeMonitor:
    def __init__(self, samples): self.samples = samples
    def start(self): pass
    def stop(self): return self.samples


class ConcurrentRunnerTests(unittest.TestCase):
    def test_dry_run_request_contract_and_sampling_defaults(self):
        document = runner.dry_run(spec(4))
        self.assertEqual(document["barrier_parties"], 4)
        for request in document["requests"]:
            self.assertEqual(request["headers"]["x-request-id"], request["client_request_id"])
            self.assertTrue(request["body"]["stream"])
            self.assertEqual(request["body"]["stream_options"], {"include_usage": True})
            self.assertTrue(request["body"]["ignore_eos"])
            for key in ("temperature", "top_p", "top_k"):
                self.assertNotIn(key, request["body"])

    def test_scheduler_tail_starts_after_startup_warmup_history(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scheduler.jsonl"
            warmup = {"client_request_id": "__sglang_c2c3_startup_warmup__"}
            measured = {"client_request_id": "run-r0"}
            path.write_text(json.dumps(warmup) + "\n")
            monitor = runner.JsonlTail(path, interval=.005)
            monitor.start()
            with path.open("a") as handle:
                handle.write(json.dumps(measured) + "\n")
            time.sleep(.03)
            rows = monitor.stop()
        self.assertEqual([row["event"] for row in rows], [measured])

    def test_threads_are_barrier_aligned_and_concurrent(self):
        active = 0
        peak = 0
        lock = threading.Lock()
        def transport(_spec, _request, result, _timeout):
            nonlocal active, peak
            result["timestamps"]["submission"] = runner._stamp()
            with lock:
                active += 1; peak = max(peak, active)
            time.sleep(.03)
            result["http_status"] = 200
            with lock: active -= 1
        process = FakeMonitor([{"timestamp": iso(-1), "server_process_id": "p"},
                               {"timestamp": iso(3), "server_process_id": "p"}])
        gpu = FakeMonitor(raw_run()["gpu_telemetry"])
        scheduler = FakeMonitor([])
        document = runner.run_concurrent(spec(4), {}, Path("unused"), transport=transport,
            gpu_sampler=gpu, scheduler_tail=scheduler, process_sampler=process)
        arrivals = [r["timestamps"]["arrival"]["monotonic_s"] for r in document["requests"]]
        self.assertEqual(peak, 4)
        self.assertLess(max(arrivals) - min(arrivals), .1)

    def test_partial_events_survive_each_transport_failure_bucket(self):
        cases = [(TimeoutError("timed out"), "timeout"),
                 (RuntimeError("CUDA out of memory"), "oom"),
                 (ConnectionResetError("connection reset by peer"), "restart"),
                 (urllib.error.HTTPError("http://x", 500, "bad", {}, None), "http_error"),
                 (OSError("disconnected"), "disconnect")]
        for exception, bucket in cases:
            with self.subTest(bucket=bucket):
                result = runner._empty_result(spec(1)["requests"][0])
                def transport(_spec, _request, current, _timeout):
                    current["timestamps"]["submission"] = runner._stamp()
                    current["raw_sse"].append({**runner._stamp(), "line": "data: partial\n"})
                    runner._append_event([json.dumps({"choices": [{"delta": {"content": "partial"}}]})], current)
                    raise exception
                runner._worker(spec(1), spec(1)["requests"][0], result, _ImmediateBarrier(), 1, transport)
                self.assertEqual(result["failure_bucket"], bucket)
                self.assertEqual(result["content"], "partial")
                self.assertEqual(len(result["events"]), 1)
                self.assertEqual(len(result["raw_sse"]), 1)

    def test_http_oom_body_retains_http_status_and_classifies_root_failure(self):
        error = urllib.error.HTTPError("http://x", 500, "bad",
                                      {"Content-Type": "application/json"},
                                      io.BytesIO(b'{"error":"CUDA out of memory"}'))
        result = runner._empty_result(spec(1)["requests"][0])
        with patch.object(runner.urllib.request, "urlopen", side_effect=error):
            runner.stream_request(spec(1), spec(1)["requests"][0], result, 1)
        self.assertEqual(result["http_status"], 500)
        self.assertEqual(result["failure_bucket"], "oom")
        self.assertIn("CUDA out of memory", result["raw_error_body"])

    def test_process_sampler_handles_parenthesized_comm_with_spaces(self):
        # Fields after comm start at field 3; index 19 is Linux stat starttime.
        tail = ["S"] + [str(i) for i in range(4, 22)] + ["98765"]
        sampler = runner.ProcessSampler(123, 1)
        with patch.object(runner.pathlib.Path, "read_text",
                          return_value="123 (server worker name) " + " ".join(tail)):
            sampler._sample()
        self.assertEqual(sampler.samples[0]["server_process_id"], "pid:123:start_ticks:98765")

    def test_incomplete_stream_retains_every_event(self):
        result = runner._empty_result(spec(1)["requests"][0])
        def transport(_spec, _request, current, _timeout):
            current["timestamps"]["submission"] = runner._stamp(); current["http_status"] = 200
            runner._append_event([json.dumps({"choices": [{"delta": {"reasoning_content": "think"}}]})], current)
            runner._append_event([json.dumps({"choices": [{"delta": {"content": "answer"}}]})], current)
        runner._worker(spec(1), spec(1)["requests"][0], result, _ImmediateBarrier(), 1, transport)
        self.assertEqual((result["reasoning_content"], result["content"]), ("think", "answer"))
        self.assertEqual(len(result["events"]), 2)
        self.assertIsNone(result["finish_reason"])

    def test_undelimited_partial_event_is_promoted_when_readline_fails(self):
        class BrokenResponse:
            def __init__(self): self.calls = 0
            def readline(self):
                self.calls += 1
                if self.calls == 1:
                    return b'data: {"choices":[{"delta":{"content":"kept"}}]}\n'
                raise TimeoutError("stream timed out")
        result = runner._empty_result(spec(1)["requests"][0])
        with self.assertRaises(TimeoutError):
            runner._record_sse(BrokenResponse(), result)
        self.assertEqual(result["content"], "kept")
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(len(result["raw_sse"]), 1)


class _ImmediateBarrier:
    def wait(self): return 0


class CampaignImporterTests(unittest.TestCase):
    def test_complete_run_imports_counts_timing_throughput_and_evidence(self):
        imported = importer.validate_and_import(raw_run())
        self.assertTrue(imported["accepted"], imported["errors"])
        metrics = imported["requests"][0]["metrics"]
        self.assertEqual((metrics["prompt_tokens"], metrics["completion_tokens"],
                          metrics["reasoning_tokens"], metrics["visible_tokens"]), (3, 2, 0, 2))
        self.assertTrue(metrics["forced_output_valid"])
        self.assertAlmostEqual(metrics["ttft_s"], .19, places=6)
        self.assertEqual(metrics["itl"]["status"], "available")
        self.assertAlmostEqual(metrics["itl"]["max_itl_ms"], 100)
        self.assertGreater(metrics["completion_tok_s_end_to_end"], 0)
        self.assertGreater(imported["aggregate_metrics"]["completion_tok_s"], 0)
        self.assertEqual(imported["aggregate_metrics"]["observed_max_running"], 2)
        self.assertEqual(imported["interval"]["minimum_free_vram_fraction"], .06)

    def test_response_headers_and_client_timing_cannot_replace_admission(self):
        raw = raw_run()
        raw["scheduler_evidence"] = []
        imported = importer.validate_and_import(raw)
        self.assertFalse(imported["accepted"])
        self.assertTrue(any("admission/start" in error for error in imported["errors"]))

    def test_unsupported_chunk_gap_itl_is_explicitly_unavailable(self):
        raw = raw_run()
        for request in raw["requests"]:
            for event in request["events"]:
                event["parsed"].pop("token_ids"); event["parsed"].pop("token_timestamps_s")
        imported = importer.validate_and_import(raw)
        self.assertFalse(imported["accepted"])
        self.assertEqual(imported["requests"][0]["metrics"]["itl"]["status"], "unavailable")
        self.assertIn("one timestamp per emitted token", imported["requests"][0]["metrics"]["itl"]["basis"])

    def test_failed_partial_outcome_is_embedded_not_discarded(self):
        raw = raw_run()
        request = raw["requests"][0]
        request["failure_bucket"] = "timeout"; request["error"] = {"type": "TimeoutError", "message": "x"}
        request["finish_reason"] = None; request["usage"] = None
        imported = importer.validate_and_import(raw)
        self.assertFalse(imported["accepted"])
        retained = imported["requests"][0]["raw_request"]
        self.assertEqual(retained["failure_bucket"], "timeout")
        self.assertEqual(len(retained["raw_sse"]), 2)
        self.assertEqual(len(retained["events"]), 2)

    def test_forced_output_requires_exact_count_and_length_finish(self):
        for mutation in (lambda r: r["usage"].update(completion_tokens=1),
                         lambda r: r.update(finish_reason="stop")):
            with self.subTest(mutation=mutation):
                raw = raw_run(); mutation(raw["requests"][0])
                imported = importer.validate_and_import(raw)
                self.assertFalse(imported["accepted"])
                self.assertFalse(imported["requests"][0]["metrics"]["forced_output_valid"])

    def test_queue_probe_requires_observed_queue_and_exact_occupancy(self):
        raw = raw_run(count=3, capacity=2)
        accepted = importer.validate_and_import(raw)
        self.assertTrue(accepted["accepted"], accepted["errors"])
        self.assertEqual(accepted["aggregate_metrics"]["observed_max_queued"], 1)
        no_queue = copy.deepcopy(raw)
        for row in no_queue["scheduler_evidence"]:
            row["event"]["queued"] = 0
            if row["event"]["event"] == "queued": row["event"]["event"] = "admitted"
        self.assertFalse(importer.validate_and_import(no_queue)["accepted"])
        wrong = copy.deepcopy(raw)
        for row in wrong["scheduler_evidence"]: row["event"]["running"] = min(1, row["event"]["running"])
        self.assertFalse(importer.validate_and_import(wrong)["accepted"])

    def test_contradictory_scheduler_transition_is_rejected_even_when_maxima_match(self):
        raw = raw_run()
        row = next(row for row in raw["scheduler_evidence"]
                   if row["event"]["client_request_id"] == "r1" and row["event"]["event"] == "started")
        row["event"]["running"] = 1
        imported = importer.validate_and_import(raw)
        self.assertFalse(imported["accepted"])
        self.assertTrue(any("contradictory server running count" in error for error in imported["errors"]))

    def test_fail_closed_server_gpu_process_and_restart_evidence(self):
        mutations = [
            lambda r: r["server"].update(observed_server_args={"bad": "UNRESOLVED"}),
            lambda r: r["server"]["identities"].update(image_digest="UNRESOLVED"),
            lambda r: r["server"].update(resolved_capacity={}),
            lambda r: r["server"].update(memory_pools={}),
            lambda r: r["server"].update(cuda_graphs={}),
            lambda r: r["gpu_telemetry"].pop(0),
            lambda r: r["gpu_telemetry"][1].update(free_vram_bytes=40),
            lambda r: r["server"]["process"].update(restart_count=1),
            lambda r: r["server"]["process"]["samples"][1].update(server_process_id="changed"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                raw = raw_run(); mutation(raw)
                self.assertFalse(importer.validate_and_import(raw)["accepted"])

    def test_observed_server_fields_and_launch_provenance_require_exact_recipe(self):
        self.assertTrue(importer.validate_and_import(raw_run(count=3, capacity=3))["accepted"])
        mutations = [
            lambda server: server["observed_server_args"].pop("attention_backend"),
            lambda server: server["observed_server_args"].update(max_running_requests=3),
            lambda server: server["observed_server_args"].update(mamba_ssm_dtype="bfloat16"),
            lambda server: server["observed_server_args"].update(speculative_algorithm="EAGLE"),
            lambda server: server["observed_server_args"].update(mamba_radix_cache_strategy="extra_buffer"),
            lambda server: server["observed_server_args"].update(model_path="/models/wrong/snapshots/" + "0" * 40),
            lambda server: server["launch_provenance"]["container_command"].extend(["--mamba-full-memory-ratio", "0.22"]),
            lambda server: server["launch_provenance"].update(image_digest="sha256:" + "0" * 64),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                raw = raw_run(); mutation(raw["server"])
                self.assertFalse(importer.validate_and_import(raw)["accepted"])

    def test_pool_and_graph_evidence_requires_semantic_measured_shapes(self):
        mutations = [
            lambda s: s.update(memory_pools={"anything": "passes"}),
            lambda s: s["memory_pools"].update(source="UNRESOLVED"),
            lambda s: s["memory_pools"].update(source="anything"),
            lambda s: s["memory_pools"]["kv_cache"].update(dtype="bf16"),
            lambda s: s["memory_pools"]["kv_cache"].update(bytes=0),
            lambda s: s["memory_pools"]["mamba_state_cache"].update(slots=4),
            lambda s: s["memory_pools"]["dflash_intermediate"].update(states=7),
            lambda s: s.update(cuda_graphs={"anything": "passes"}),
            lambda s: s["cuda_graphs"].update(source="TODO"),
            lambda s: s["cuda_graphs"].update(source="anything"),
            lambda s: s["cuda_graphs"].update(enabled=False),
            lambda s: s["cuda_graphs"].update(captured_batch_sizes=[1]),
            lambda s: s["cuda_graphs"].update(captured_batch_sizes=[1, 1, 2]),
            lambda s: s["cuda_graphs"].update(memory_bytes=0),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                raw = raw_run(); mutation(raw["server"])
                self.assertFalse(importer.validate_and_import(raw)["accepted"])

    def test_capacity_and_request_metadata_are_separate(self):
        mutations = [
            lambda c: c.update(max_mamba_cache_size=4),
            lambda c: c.update(tp_size=2),
            lambda c: c.update(max_running_requests=3),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                raw = raw_run(); mutation(raw["server"]["resolved_capacity"])
                self.assertFalse(importer.validate_and_import(raw)["accepted"])
        raw = raw_run(); raw["server"]["campaign_request_limits"]["max_output_tokens"] = 131071
        self.assertFalse(importer.validate_and_import(raw)["accepted"])
        raw = raw_run(); raw["server"]["launch_metadata"]["planned_port"] = 11448
        self.assertFalse(importer.validate_and_import(raw)["accepted"])

    def test_malformed_or_interval_misaligned_evidence_is_rejected(self):
        raw = raw_run(); raw["scheduler_evidence"].append({"event": {"malformed_line": "{"}})
        self.assertFalse(importer.validate_and_import(raw)["accepted"])
        raw = raw_run(); raw["gpu_telemetry"][1]["utc"] = iso(10)
        self.assertFalse(importer.validate_and_import(raw)["accepted"])

    def test_optional_exact_boundary_uses_only_server_prompt_usage(self):
        raw = raw_run(); raw["boundary_proof"] = {"expected_server_prompt_tokens": 3}
        self.assertTrue(importer.validate_and_import(raw)["accepted"])
        raw["boundary_proof"] = {"expected_server_prompt_tokens": 131072}
        self.assertFalse(importer.validate_and_import(raw)["accepted"])

    def test_reasoning_and_visible_counts_are_separate(self):
        raw = raw_run()
        for request in raw["requests"]:
            request["usage"]["completion_tokens"] = 2
            request["usage"]["completion_tokens_details"]["reasoning_tokens"] = 1
            request["events"][0]["parsed"]["choices"][0]["delta"] = {"reasoning_content": "a"}
        imported = importer.validate_and_import(raw)
        self.assertTrue(imported["accepted"], imported["errors"])
        self.assertEqual(imported["requests"][0]["metrics"]["visible_tokens"], 1)
        self.assertEqual(imported["requests"][0]["metrics"]["emitted_reasoning_tokens"], 1)
        self.assertEqual(imported["requests"][0]["metrics"]["reasoning_tokens"], 1)

    def test_rendered_reasoning_usage_may_exceed_emitted_completion_tokens(self):
        raw = raw_run()
        for request in raw["requests"]:
            request["usage"].pop("completion_tokens_details")
            request["usage"]["reasoning_tokens"] = 5
            for event in request["events"]:
                event["parsed"]["choices"][0]["delta"] = {"reasoning_content": "expanded text"}
            request["content"] = ""
            request["reasoning_content"] = "expanded text expanded text"
        imported = importer.validate_and_import(raw)
        self.assertTrue(imported["accepted"], imported["errors"])
        metrics = imported["requests"][0]["metrics"]
        self.assertEqual(metrics["completion_tokens"], 2)
        self.assertEqual(metrics["reasoning_tokens"], 5)
        self.assertEqual(metrics["emitted_reasoning_tokens"], 2)
        self.assertEqual(metrics["visible_tokens"], 0)

    def test_stream_channel_counts_fail_closed_on_mixed_or_missing_ids(self):
        raw = raw_run()
        raw["requests"][0]["events"][0]["parsed"]["choices"][0]["delta"] = {
            "content": "a", "reasoning_content": "thought"}
        imported = importer.validate_and_import(raw)
        self.assertFalse(imported["accepted"])
        self.assertTrue(any("cannot classify emitted tokens" in error for error in imported["errors"]))


if __name__ == "__main__":
    unittest.main()
