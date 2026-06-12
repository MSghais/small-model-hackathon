"""Backward-compatible entrypoint. Prefer: uv run --package slm-evals slm-benchmark"""

from slm_evals.run_benchmark import main

if __name__ == "__main__":
    main()
