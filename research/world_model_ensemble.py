"""Deprecated shim — use `ensemble.world_ensemble` instead."""

from ensemble.world_ensemble import (
    WorldEnsemble,
    demo,
    hf_segment_sequences,
    toy_segment_sequences,
)

__all__ = [
    "WorldEnsemble",
    "toy_segment_sequences",
    "hf_segment_sequences",
    "demo",
]

if __name__ == "__main__":
    import sys

    demo(sys.argv[1] if len(sys.argv) > 1 else "tiny")
