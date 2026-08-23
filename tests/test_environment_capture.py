import json
import tempfile
import unittest
from pathlib import Path

from bench.environment_capture import capture, command_result, main


class EnvironmentCaptureTests(unittest.TestCase):
    def fake_runner(self, argv):
        if argv[0] == "nvidia-smi":
            return 0, "GPU 0, 96 GB, 1.0, 300, 1600, Enabled, Disabled, 300, 42", ""
        if argv[0] == "uname":
            return 0, "Linux fixture 6.1 x86_64", ""
        if argv[0] == "ss":
            return 0, "LISTEN 0 128 127.0.0.1:11436", ""
        if argv[0] in {"lspci", "numactl", "lscpu", "free", "docker", "nvidia-container-cli"}:
            return 127, "", "not installed in fixture"
        return 0, "fixture", ""

    def test_success_and_unavailable_capabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "environment.json"
            data = capture(out, runner=self.fake_runner)
            self.assertEqual(data["schema"], "environment.v1")
            self.assertEqual(data["commands"]["gpu_inventory"]["status"], "available")
            self.assertEqual(data["commands"]["docker"]["status"], "unavailable")
            self.assertEqual(data["commands"]["docker"]["command"][0], "docker")
            self.assertTrue((Path(tmp) / "source-compatibility.md").exists())
            self.assertEqual(json.loads(out.read_text())["inputs"]["source_lock"], "UNRESOLVED until explicitly verified")

    def test_snapshot_identity_without_hashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "snapshots" / "abc123"
            root.mkdir(parents=True)
            (root / "config.json").write_text("{}")
            blob = root / ".." / "blobs" / "etag-123"
            blob.parent.mkdir()
            blob.write_text("weights")
            (root / "tokenizer.json").symlink_to(blob)
            (root / "model-00001-of-00001.safetensors").write_text("weights")
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}}))
            data = capture(Path(tmp) / "environment.json", snapshot=root, repo="Qwen/Qwen3.8-27B", revision="abc123", runner=self.fake_runner, lock_paths=[Path(tmp) / "missing.lock.json"])
            snap = data["inputs"]["huggingface_snapshot"]
            self.assertEqual(snap["status"], "available")
            self.assertEqual(snap["revision"], "abc123")
            self.assertEqual(snap["weight_shard_count"], 1)
            self.assertEqual(snap["safetensors_index"]["status"], "valid")
            self.assertFalse(snap["full_hash"])
            self.assertEqual(snap["identity_coverage"]["status"], "partial")
            self.assertEqual(data["locks"][str(Path(tmp) / "missing.lock.json")]["status"], "unavailable")
            self.assertTrue(any(item.get("etag") == "etag-123" for item in snap["files"]))

    def test_snapshot_revision_mismatch_and_bad_index_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "snapshots" / "actual"
            root.mkdir(parents=True)
            with self.assertRaises(ValueError):
                capture(Path(tmp) / "environment.json", snapshot=root, repo="repo", revision="claimed", runner=self.fake_runner)
            (root / "model-00001-of-00001.safetensors").write_text("weights")
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "wrong.safetensors"}}))
            with self.assertRaises(ValueError):
                capture(Path(tmp) / "environment.json", snapshot=root, repo="repo", revision="actual", runner=self.fake_runner)

    def test_invalid_snapshot_arguments_fail(self):
        with self.assertRaises(ValueError):
            capture(Path(tempfile.gettempdir()) / "environment-invalid.json", snapshot=Path("/does/not/exist"), repo=None, revision=None)
        with self.assertRaises(SystemExit):
            main(["--output", "/tmp/environment-invalid.json", "--full-hash"])

    def test_command_result_redacts_nothing_sensitive(self):
        result = command_result(["missing-command"], lambda argv: (127, "", "No such file"))
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("environ", json.dumps(result).lower())


if __name__ == "__main__":
    unittest.main()
