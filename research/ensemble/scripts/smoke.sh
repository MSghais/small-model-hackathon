#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

echo "== JEPA ensemble demo (tiny) =="
uv run --package ensemble python -m ensemble.jepa_ensemble tiny

echo ""
echo "== World ensemble demo (tiny) =="
uv run --package ensemble python -m ensemble.world_ensemble tiny

echo ""
echo "== JEPA harness (toy) =="
uv run --package ensemble python -m ensemble.eval.jepa_harness \
  --llm tiny --toy --limit 10 --n_drafts 4

echo ""
echo "== World harness (toy) =="
uv run --package ensemble python -m ensemble.eval.world_harness \
  --llm tiny --toy --limit 10 --n_drafts 4

echo ""
echo "All smoke checks passed."
