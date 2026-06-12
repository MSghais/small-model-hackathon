"""Deprecated shim — use `ensemble.jepa_ensemble` instead."""

from ensemble.backends import HFBackend, LLMBackend, TinyBackend, make_backend
from ensemble.bridge import Bridge
from ensemble.jepa import JEPA
from ensemble.jepa_ensemble import Ensemble, demo_hf, demo_tiny, segment_pairs_from_texts
from ensemble.memory import Embedder, Router, VectorStore

__all__ = [
    "LLMBackend",
    "HFBackend",
    "TinyBackend",
    "make_backend",
    "Embedder",
    "JEPA",
    "Bridge",
    "VectorStore",
    "Router",
    "Ensemble",
    "segment_pairs_from_texts",
    "demo_tiny",
    "demo_hf",
]
