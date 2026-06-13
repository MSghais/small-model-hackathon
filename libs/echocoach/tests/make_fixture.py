"""Generate a short silent WAV fixture for smoke tests."""

from pathlib import Path

import numpy as np
import soundfile as sf

OUT = Path(__file__).resolve().parent / "fixtures" / "silence_2s.wav"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    audio = np.zeros(32_000, dtype=np.float32)  # 2s @ 16kHz
    sf.write(OUT, audio, 16_000)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
