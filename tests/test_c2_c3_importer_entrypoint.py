import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class ImporterEntrypointTests(unittest.TestCase):
    def test_file_path_entrypoint_imports_package_from_repo_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.json"
            output = root / "imported.json"
            raw.write_text(json.dumps({}))
            completed = subprocess.run(
                [sys.executable, "bench/c2_c3_importer.py", str(raw), "--output", str(output)],
                cwd=REPO,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertNotIn("ModuleNotFoundError", completed.stdout)
            self.assertEqual(completed.returncode, 1)
            self.assertTrue(output.exists())
            self.assertIn("accepted", json.loads(output.read_text()))


if __name__ == "__main__":
    unittest.main()
