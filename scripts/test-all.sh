#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
cd "$repo"
bash "$repo/scripts/test-launcher-validation.sh"
bash "$repo/scripts/validate-scaffold.sh"
python3 -m unittest discover -s "$repo/tests" -v
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
python3 "$repo/bench/benchmark_contract.py" manifest "$tmpdir/phase7-minimum.generated.json"
cmp -s "$repo/bench/phase7-minimum.json" "$tmpdir/phase7-minimum.generated.json"
python3 "$repo/bench/benchmark_contract.py" validate "$repo/bench/phase7-minimum.json"
python3 "$repo/bench/environment_capture.py" --output "$tmpdir/environment.json" --disk-path "$tmpdir" >/dev/null
test -s "$tmpdir/environment.json"
test -s "$tmpdir/source-compatibility.md"
python3 "$repo/bench/capacity.py" --json "$tmpdir/capacity.json" --markdown "$tmpdir/capacity.md" >/dev/null
python3 - "$tmpdir/capacity.json" <<'PY'
import json, sys
assert len(json.load(open(sys.argv[1]))["rows"]) == 1344
PY
git -C "$repo" diff --check
echo "all scaffold, launcher, smoke, and whitespace checks passed"
