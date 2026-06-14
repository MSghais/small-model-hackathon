"""Hugging Face Gradio SDK entry point (ZeroGPU / Gradio Spaces)."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _src in (
    "apps/gradio-space/src",
    "libs/inference/src",
    "libs/researchmind/src",
    "libs/agent/src",
    "libs/echocoach/src",
):
    _path = str(_ROOT / _src)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from gradio_space.server import main

if __name__ == "__main__":
    main()
