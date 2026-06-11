"""
SLM Agentic Benchmark Suite
============================
Run BFCL, τ-bench, GAIA, SWE-bench, and internal evals
against a local HuggingFace model checkpoint.

Usage:
    python run_benchmark.py --model ./path/to/model --benchmarks bfcl tau_bench
    python run_benchmark.py --model ./path/to/model --benchmarks all --max-samples 50
    python run_benchmark.py --config configs/experiment_001.yaml
"""

import argparse
import sys
from pathlib import Path

# Make sure sub-packages are importable
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_loader import load_model
from utils.reporter import Reporter
from utils.config_loader import load_config, build_config_from_args
from benchmarks.bfcl import BFCLBenchmark
from benchmarks.tau_bench import TauBenchmark
from benchmarks.gaia import GAIABenchmark
from benchmarks.swe_bench import SWEBenchmark

BENCHMARK_REGISTRY = {
    "bfcl":      BFCLBenchmark,
    "tau_bench":  TauBenchmark,
    "gaia":       GAIABenchmark,
    "swe_bench":  SWEBenchmark,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SLM Agentic Benchmark Suite — HuggingFace backend"
    )
    parser.add_argument(
        "--model", type=str,
        help="Path to local HuggingFace model directory (or HF Hub ID)"
    )
    parser.add_argument(
        "--benchmarks", nargs="+",
        choices=list(BENCHMARK_REGISTRY.keys()) + ["all"],
        default=["all"],
        help="Which benchmarks to run (default: all)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional YAML config file (overrides other flags)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap number of samples per benchmark (useful for quick smoke tests)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to write results (default: ./results)"
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Name tag for this run (auto-generated from timestamp if omitted)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device map for HF: 'auto', 'cpu', 'cuda', 'cuda:0' etc."
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16", "int8", "int4"],
        help="Model dtype / quantization level"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Max tokens to generate per inference call"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0.0 = greedy)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config (YAML wins over CLI flags if provided) ──────────────────
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = build_config_from_args(args)

    print(f"\n{'='*60}")
    print(f"  SLM Benchmark Suite")
    print(f"  Model : {cfg['model_path']}")
    print(f"  Runs  : {', '.join(cfg['benchmarks'])}")
    print(f"  Out   : {cfg['output_dir']}")
    print(f"{'='*60}\n")

    # ── Load model once, reuse across all benchmarks ─────────────────────────
    print("⏳ Loading model …")
    model_bundle = load_model(
        model_path=cfg["model_path"],
        device=cfg["device"],
        dtype=cfg["dtype"],
    )
    print(f"✅ Model loaded — {model_bundle['param_count']:.2f}B parameters\n")

    # ── Run benchmarks ────────────────────────────────────────────────────────
    reporter = Reporter(
        output_dir=cfg["output_dir"],
        experiment_name=cfg["experiment_name"],
        model_path=cfg["model_path"],
    )

    benchmark_names = (
        list(BENCHMARK_REGISTRY.keys())
        if "all" in cfg["benchmarks"]
        else cfg["benchmarks"]
    )

    all_results = {}
    for name in benchmark_names:
        print(f"▶  Running benchmark: {name.upper()}")
        print(f"   {'─'*50}")

        bench_cls = BENCHMARK_REGISTRY[name]
        bench = bench_cls(
            model_bundle=model_bundle,
            max_samples=cfg.get("max_samples"),
            max_new_tokens=cfg.get("max_new_tokens", 512),
            temperature=cfg.get("temperature", 0.0),
            benchmark_cfg=cfg.get("benchmark_overrides", {}).get(name, {}),
        )

        result = bench.run()
        all_results[name] = result

        print(f"   Score : {result['score']:.2%}")
        print(f"   Passed: {result['passed']} / {result['total']}")
        print()

    # ── Write outputs ─────────────────────────────────────────────────────────
    paths = reporter.save(all_results)
    print(f"\n{'='*60}")
    print(f"  Results saved:")
    for fmt, path in paths.items():
        print(f"    {fmt:<8} → {path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
