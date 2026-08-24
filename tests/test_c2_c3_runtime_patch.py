import ast
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


REPO = pathlib.Path(__file__).resolve().parents[1]
PATCH = REPO / "patches/sglang/0002-c2c3-server-evidence.patch"
PINNED_HASHES = {
    "entrypoints/openai/sse_utils.py": "a04b5b4548067c8932466aa99f9758d7a87d547f6a1843f9d104bbbfe3ea2ac4",
    "entrypoints/openai/serving_chat.py": "4f14086fe02f4f1efea0a953784c8ac01d3d5d5209df85ea7cd2baa5a5c16027",
    "managers/scheduler.py": "7beb1b7108f4e6eebdeda57ea2126f7a1404fc0af7a2b24179cf8259f8e0b2a2",
    "managers/scheduler_components/output_streamer.py": "4f069fbba82a77360fec29330edf5f6cf3740960984c5e1d01ef4e385c90ecaa",
}


def _source_root():
    configured = os.environ.get("SGLANG_PINNED_SRT_SOURCE")
    if configured:
        return pathlib.Path(configured)
    matches = sorted(pathlib.Path("/tmp").glob("qwen38-sglang-source.*/srt"))
    return matches[-1] if matches else None


def _function(tree, name):
    matches = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if not matches:
        raise AssertionError(f"production function disappeared: {name}")
    return matches[0]


def _calls(function):
    names = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    return names


class RuntimePatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _source_root()
        if cls.source is None:
            raise unittest.SkipTest("set SGLANG_PINNED_SRT_SOURCE to run pinned overlay tests")
        for relative, expected in PINNED_HASHES.items():
            actual = hashlib.sha256((cls.source / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise AssertionError(f"pinned source mismatch for {relative}: {actual}")
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.temp.name)
        target = cls.root / "python/sglang/srt"
        target.parent.mkdir(parents=True)
        shutil.copytree(cls.source, target, symlinks=True)
        completed = subprocess.run(
            ["patch", "--batch", "--forward", "-d", str(cls.root), "-p1"],
            input=PATCH.read_bytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode:
            raise AssertionError(completed.stdout.decode(errors="replace"))
        patch_output = completed.stdout.decode(errors="replace")
        if "fuzz" in patch_output.lower() or "offset" in patch_output.lower():
            raise AssertionError(f"overlay must apply exactly to pinned source:\n{patch_output}")
        cls.srt = target
        for relative in (
            "campaign_evidence.py",
            "entrypoints/openai/sse_utils.py",
            "entrypoints/openai/serving_chat.py",
            "managers/scheduler.py",
            "managers/scheduler_components/output_streamer.py",
        ):
            compile((target / relative).read_text(), str(target / relative), "exec")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _load_evidence(self):
        name = f"campaign_evidence_test_{id(self)}"
        spec = importlib.util.spec_from_file_location(name, self.srt / "campaign_evidence.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.addCleanup(sys.modules.pop, name, None)
        return module

    def test_disabled_mode_is_a_noop(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            module = self._load_evidence()
            self.assertIsNone(module.request_id_from_headers({"x-request-id": "client-a"}))
            self.assertEqual(module.stream_token_ids([1], incremental=True, offsets={}, index=0), [])
            module.get_scheduler_recorder().queued("client-a")

    def test_header_tokens_and_real_jsonl_transitions(self):
        evidence_path = self.root / "scheduler.jsonl"
        with mock.patch.dict(os.environ, {"SGLANG_C2C3_EVIDENCE_PATH": str(evidence_path)}, clear=True):
            module = self._load_evidence()
            self.assertEqual(module.request_id_from_headers({"x-request-id": " client-a "}), "client-a")
            with self.assertRaises(ValueError):
                module.request_id_from_headers({})
            offsets = {}
            ids1 = module.stream_token_ids([10, 11], incremental=False, offsets=offsets, index=0)
            ids2 = module.stream_token_ids([10, 11, 12], incremental=False, offsets=offsets, index=0)
            self.assertEqual(ids1 + ids2, [10, 11, 12])
            timestamp_state = {}
            with mock.patch.object(module.time, "time_ns", side_effect=[1_000_000_000, 1_000_000_000, 1_000_005_000]):
                times1 = module.token_emission_timestamps(ids1, timestamp_state)
                times2 = module.token_emission_timestamps(ids2, timestamp_state)
            self.assertEqual(times1 + times2, [1.0, 1.000001, 1.000005])

            recorder = module.get_scheduler_recorder()
            recorder.queued("client-a")
            recorder.admitted("client-a")
            recorder.started("client-a")
            recorder.terminal("client-a", failed=False)
            rows = [json.loads(line) for line in evidence_path.read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows], ["queued", "admitted", "started", "completed"])
            self.assertEqual([(row["running"], row["queued"]) for row in rows], [(0, 1), (1, 0), (1, 0), (0, 0)])
            self.assertEqual({row["client_request_id"] for row in rows}, {"client-a"})
            self.assertEqual(len({row["server_process_id"] for row in rows}), 1)
            self.assertRegex(rows[0]["server_process_id"], r"^pid:\d+:start_ticks:\d+$")
            stat = pathlib.Path(f"/proc/{os.getpid()}/stat").read_text()
            tail = stat[stat.rfind(")") + 2:].split()
            # tail[0] is Linux stat field 3 (state), so field 22 is tail[19].
            self.assertIn(tail[0], {"R", "S", "D", "I"})
            expected_start = tail[19]
            self.assertEqual(rows[0]["server_process_id"], f"pid:{os.getpid()}:start_ticks:{expected_start}")
            clock_ticks = os.sysconf("SC_CLK_TCK")
            uptime_ticks = float(pathlib.Path("/proc/uptime").read_text().split()[0]) * clock_ticks
            self.assertGreater(int(expected_start), uptime_ticks - 120 * clock_ticks)

    def test_production_call_sites_are_ast_connected(self):
        chat = ast.parse((self.srt / "entrypoints/openai/serving_chat.py").read_text())
        scheduler = ast.parse((self.srt / "managers/scheduler.py").read_text())
        streamer = ast.parse((self.srt / "managers/scheduler_components/output_streamer.py").read_text())
        self.assertIn("request_id_from_headers", _calls(_function(chat, "_convert_to_internal_request")))
        self.assertIn("stream_token_ids", _calls(_function(chat, "_generate_chat_stream")))
        stream_content_calls = _calls(_function(chat, "_generate_stream_content"))
        self.assertIn("build_sse_content", stream_content_calls)
        self.assertIn("token_emission_timestamps", stream_content_calls)
        self.assertIn("queued", _calls(_function(scheduler, "_add_request_to_queue")))
        self.assertIn("admitted", _calls(_function(scheduler, "_get_new_batch_prefill_raw")))
        accept_calls = _calls(_function(streamer, "accept"))
        self.assertIn("started", accept_calls)
        self.assertIn("terminal", accept_calls)
        accept = _function(streamer, "accept")
        started = next(
            node for node in ast.walk(accept)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "started"
        )
        output_gate = next(
            node for node in ast.walk(accept)
            if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not) and isinstance(node.test.operand, ast.Name)
            and node.test.operand.id == "should_output"
        )
        self.assertLess(started.lineno, output_gate.lineno)

    def test_real_sse_builder_serializes_token_pairs(self):
        class Struct:
            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__()

            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def plain(value):
            if isinstance(value, list):
                return [plain(item) for item in value]
            if hasattr(value, "__dict__"):
                return {key: plain(item) for key, item in value.__dict__.items() if item is not None}
            return value

        class Encoder:
            def encode(self, value):
                return json.dumps(plain(value), separators=(",", ":")).encode()

        msgspec = types.ModuleType("msgspec")
        msgspec.Struct = Struct
        msgspec.json = types.SimpleNamespace(Encoder=Encoder)
        name = f"sse_utils_test_{id(self)}"
        spec = importlib.util.spec_from_file_location(name, self.srt / "entrypoints/openai/sse_utils.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"msgspec": msgspec, name: module}):
            spec.loader.exec_module(module)
            wire = module.build_sse_content(
                chunk_id="client-a", created=1, model="m", index=0, content="x",
                token_ids=[42], token_timestamps_s=[123.25],
            )
        payload = json.loads(wire.removeprefix("data: ").strip())
        self.assertEqual(payload["token_ids"], [42])
        self.assertEqual(payload["token_timestamps_s"], [123.25])
        self.assertEqual(payload["choices"][0]["delta"]["content"], "x")


if __name__ == "__main__":
    unittest.main()
