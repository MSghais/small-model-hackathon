"""Matplotlib charts for filler and pace visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from echocoach.models import FillerAnalysis, PaceAnalysis


def render_filler_chart(analysis: FillerAnalysis, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not analysis.counts:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(["(none)"], [0], color="#4a90d9")
        ax.set_title("Filler words")
        ax.set_ylabel("Count")
    else:
        labels = list(analysis.counts.keys())
        values = [analysis.counts[k] for k in labels]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(labels, values, color="#e67e22")
        ax.set_title("Filler words")
        ax.set_ylabel("Count")
        plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_pace_timeline(
    transcript: str,
    duration_seconds: float,
    pace: PaceAnalysis,
    out_path: Path,
    *,
    window_seconds: float = 10.0,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    words = transcript.split()
    if not words or duration_seconds <= 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot([0, max(duration_seconds, 1)], [0, 0], color="#4a90d9")
        ax.set_title("Words per minute (rolling)")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("WPM")
        ax.axhspan(pace.target_low, pace.target_high, alpha=0.15, color="green", label="Target band")
        ax.legend(loc="upper right")
    else:
        n_windows = max(1, int(duration_seconds / window_seconds) + (1 if duration_seconds % window_seconds else 0))
        times: list[float] = []
        wpms: list[float] = []
        words_per_sec = len(words) / duration_seconds
        for i in range(n_windows):
            start = i * window_seconds
            end = min((i + 1) * window_seconds, duration_seconds)
            window_dur = end - start
            if window_dur <= 0:
                continue
            approx_words = words_per_sec * window_dur
            wpm = approx_words / (window_dur / 60.0)
            times.append((start + end) / 2)
            wpms.append(wpm)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(times, wpms, marker="o", color="#4a90d9", linewidth=2)
        ax.axhspan(pace.target_low, pace.target_high, alpha=0.15, color="green", label="Target band")
        ax.set_title("Words per minute (by segment)")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("WPM")
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def build_charts(
    transcript: str,
    duration_seconds: float,
    fillers: FillerAnalysis,
    pace: PaceAnalysis,
    output_dir: Path,
    run_id: str,
) -> tuple[Path, Path]:
    filler_path = output_dir / f"{run_id}_fillers.png"
    pace_path = output_dir / f"{run_id}_pace.png"
    render_filler_chart(fillers, filler_path)
    render_pace_timeline(transcript, duration_seconds, pace, pace_path)
    return filler_path, pace_path
