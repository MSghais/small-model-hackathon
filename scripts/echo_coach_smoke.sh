#!/usr/bin/env bash
# Smoke test for EchoCoach analysis (no GPU / no ASR models).
set -euo pipefail
cd "$(dirname "$0")/.."

FIXTURE="libs/echocoach/tests/fixtures/silence_2s.wav"
if [[ ! -f "$FIXTURE" ]]; then
  uv run python libs/echocoach/tests/make_fixture.py
fi

uv run pytest libs/echocoach/tests/test_fillers.py libs/echocoach/tests/test_pace.py libs/echocoach/tests/test_coach_parse.py -q
echo "EchoCoach smoke tests passed."
