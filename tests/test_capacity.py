import json
import tempfile
import unittest
from pathlib import Path

from bench.capacity import (
    LENGTHS, CONCURRENCIES, STRATEGIES, SPECULATION_D, balanced_ratio,
    max_mamba_cache_size, worksheet, main, _units,
)


class CapacityTests(unittest.TestCase):
    def test_hand_calculation_and_units(self):
        # (4 + 8) * 153.9 MB / (100,000 * 32.8 KB) = 0.56304878...
        self.assertAlmostEqual(balanced_ratio("extra_buffer_lazy", "float32", "FP8", 100_000, "DSpark"), 0.5630487805)
        self.assertEqual(max_mamba_cache_size("extra_buffer", 4), 20)

    def test_validation_and_unresolved_methods(self):
        with self.assertRaises(ValueError):
            balanced_ratio("bad", "float32", "FP8", 1, "none")
        with self.assertRaises(ValueError):
            balanced_ratio("extra_buffer_lazy", "float32", "FP8", 0, "none")
        self.assertIsNone(balanced_ratio("extra_buffer_lazy", "float32", "FP8", 1000, "MTP"))
        self.assertEqual(SPECULATION_D["DSpark"], 8)

    def test_decimal_and_binary_units_are_unambiguous(self):
        units = _units(1_000_000)
        self.assertEqual(units["MB"], 1)
        self.assertAlmostEqual(units["MiB"], 1_000_000 / 1024**2)
        self.assertEqual(units["kB"], 1_000)
        self.assertAlmostEqual(units["KiB"], 1_000_000 / 1024)

    def test_grid_completeness_and_profile_shapes(self):
        data = worksheet()
        self.assertEqual(len(data["rows"]), 7 * 3 * 2 * 2 * 4 * 4)
        self.assertEqual(sorted({r["average_total_request_tokens"] for r in data["rows"]}), list(LENGTHS))
        self.assertEqual(sorted({r["concurrency"] for r in data["rows"]}), list(CONCURRENCIES))
        self.assertEqual(set(data["profiles"]), {"ordinary_coding", "long_repo_coding", "near_native_context"})
        self.assertEqual(data["profiles"]["long_repo_coding"]["input_tokens"], 100_000)
        native = data["profiles"]["near_native_context"]
        self.assertEqual(native["input_tokens"] + native["output_tokens"], 262_144)

    def test_cli_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            main(["--json", str(root / "worksheet.json"), "--markdown", str(root / "worksheet.md")])
            payload = json.loads((root / "worksheet.json").read_text())
            self.assertEqual(len(payload["rows"]), 1344)
            self.assertIn("UNRESOLVED", (root / "worksheet.md").read_text())
            self.assertIn("Predicted KV/GDN", (root / "worksheet.md").read_text())


if __name__ == "__main__":
    unittest.main()
