"""Deprecated shim — use `ensemble.eval.jepa_harness` instead."""

from ensemble.eval.jepa_harness import run, parse_args

if __name__ == "__main__":
    run(parse_args())
