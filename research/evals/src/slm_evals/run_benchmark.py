"""
SLM Agentic Benchmark Suite
============================
Run BFCL, τ-bench, GAIA, SWE-bench against a local HuggingFace model checkpoint.

Usage:
    uv run --package slm-evals slm-benchmark --model ./path/to/model --benchmarks bfcl
    uv run --package slm-evals python -m slm_evals.run_benchmark --model ./path/to/model
    uv run --package slm-evals slm-benchmark --config configs/experiment_001.yaml
"""

from __future__ import annotations

import argparse
import sys

from slm_evals.benchmarks.bfcl import BFCLBenchmark
from slm_evals.benchmarks.gaia import GAIABenchmark
from slm_evals.benchmarks.swe_bench import SWEBenchmark
from slm_evals.benchmarks.tau_bench import TauBenchmark
from slm_evals.utils.config_loader import build_config_from_args, load_config
from slm_evals.utils.model_loader import load_model
from slm_evals.utils.reporter import Reporter

BENCHMARK_REGISTRY = {
    "bfcl": BFCLBenchmark,
    "tau_bench": TauBenchmark,
    "gaia": GAIABenchmark,
    "swe_bench": SWEBenchmark,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SLM Agentic Benchmark Suite — HuggingFace backend"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to local HuggingFace model directory (or HF Hub ID)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "hf"],
        help="Model loader backend (HuggingFace transformers)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=list(BENCHMARK_REGISTRY.keys()) + ["all"],
        default=["all"],
        help="Which benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file (overrides other flags)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of samples per benchmark (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to write results (default: ./results)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name tag for this run (auto-generated from timestamp if omitted)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device map for HF: 'auto', 'cpu', 'cuda', 'cuda:0' etc.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16", "int8", "int4"],
        help="Model dtype / quantization level",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per inference call",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="Show agentic benchmark keys and preset suites from eval_profiles.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_benchmarks:
        from slm_evals.lm_eval.profiles import format_agentic_benchmarks

        print(format_agentic_benchmarks())
        return

    if args.config:
        cfg = load_config(args.config)
    else:
        if not args.model:
            print("error: --model is required unless --config is provided", file=sys.stderr)
            sys.exit(2)
        cfg = build_config_from_args(args)

    print(f"\n{'='*60}")
    print("  SLM Benchmark Suite")
    print(f"  Model : {cfg['model_path']} ({cfg.get('model_type', 'auto')})")
    print(f"  Runs  : {', '.join(cfg['benchmarks'])}")
    print(f"  Out   : {cfg['output_dir']}")
    print(f"{'='*60}\n")

    print("⏳ Loading model …")
    model_bundle = load_model(
        model_path=cfg["model_path"],
        device=cfg["device"],
        dtype=cfg["dtype"],
        model_type=cfg.get("model_type", "auto"),
    )
    print(f"✅ Model loaded — {model_bundle['param_count']:.2f}B parameters\n")

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

    paths = reporter.save(all_results)
    print(f"\n{'='*60}")
    print("  Results saved:")
    for fmt, path in paths.items():
        print(f"    {fmt:<8} → {path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
