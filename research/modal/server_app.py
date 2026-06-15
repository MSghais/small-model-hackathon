"""
Long-lived Modal GPU worker — reuse one warm container for many finetune / eval runs.

Deploy once (enables min_containers warm pool across separate CLI invocations):
    modal deploy research/modal/server_app.py

Default: keep a GPU worker alive for several hours (blocks local terminal):
    modal run research/modal/server_app.py
    modal run research/modal/server_app.py --hours 6

Detached keep-alive (local terminal free):
    modal run -d research/modal/server_app.py --hours 6

Run the skill-matrix pipeline on the warm worker (separate terminal, same
container when deployed) — per-profile baselines -> finetune -> eval -> gate -> publish:
    modal run research/modal/server_app.py --job math-lora --max-steps 20
    modal run research/modal/server_app.py --category science
    modal run research/modal/server_app.py --pipeline --no-publish
    modal run research/modal/server_app.py --eval-only --job math-lora
    modal run research/modal/server_app.py --publish-only --job math-lora
    modal run research/modal/server_app.py --cmd "uv run python research/finetune.py --help"

Stop deployed app:
    modal app stop slm-gpu-worker
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal

# Make `_common` importable both locally (sibling file) and in the Modal
# container, where the entrypoint lands at /root but the repo is baked into the
# image at /repo (see add_local_dir in _common.py).
for _candidate in (Path(__file__).resolve().parent, Path("/repo/research/modal")):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from _common import (
    BASE_MODEL_ID,
    DEFAULT_GPU,
    DEFAULT_KEEPALIVE_HOURS,
    DEFAULT_SCALEDOWN_WINDOW,
    DEFAULT_WORKER_TIMEOUT,
    FINETUNE_VOL_PATH,
    HF_CACHE_PATH,
    LM_EVAL_OUTPUT,
    apply_defaults,
    build_finetune_cmd,
    build_lm_eval_cmd,
    check_gate_files,
    commit_volumes,
    config_for_profile,
    finetune_vol,
    hf_cache_vol,
    hf_secret,
    image,
    load_experiments,
    prepare_jobs,
    publish_adapter_files,
    pull_artifacts,
    reload_volumes,
    repo_env,
)

APP_NAME = "slm-gpu-worker"

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=DEFAULT_GPU,
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        FINETUNE_VOL_PATH: finetune_vol,
    },
    secrets=[hf_secret],
    timeout=DEFAULT_WORKER_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    min_containers=1,
    max_containers=1,  # single warm container; serialize work, never sprawl
)
class GpuWorker:
    """Single warm GPU container for sequential finetune / lm-eval / shell commands."""

    @modal.enter()
    def startup(self) -> None:
        reload_volumes()
        print(
            f"GpuWorker ready (HF cache={HF_CACHE_PATH}, finetune vol={FINETUNE_VOL_PATH})"
        )

    @modal.method()
    def ping(self) -> dict[str, str]:
        return {"status": "ok", "app": APP_NAME}

    @modal.method()
    def keep_alive(self, hours: float = DEFAULT_KEEPALIVE_HOURS) -> dict[str, Any]:
        """Hold this container open; cheap heartbeat so scaledown_window stays fresh."""
        deadline = time.time() + hours * 3600
        ticks = 0
        while time.time() < deadline:
            remaining = int(deadline - time.time())
            if ticks % 5 == 0:
                print(f"keep_alive: {remaining}s remaining")
            time.sleep(60)
            ticks += 1
        return {"status": "done", "hours": hours}

    @modal.method()
    def exec_cmd(self, argv: list[str], cwd: str = "/repo") -> dict[str, Any]:
        """Run an arbitrary command in the repo (same env as finetune.py)."""
        print("Running:", " ".join(argv))
        proc = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            env=repo_env(),
            capture_output=True,
            text=True,
        )
        commit_volumes()
        return {
            "argv": argv,
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    @modal.method()
    def finetune(self, job: dict[str, Any]) -> dict[str, Any]:
        """Fine-tune one dataset job via research/finetune.py."""
        name = job["name"]
        out_dir = f"{FINETUNE_VOL_PATH}/{name}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        cmd = build_finetune_cmd(job, out_dir)
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd="/repo", check=True, env=repo_env())

        commit_volumes()

        results_path = Path(out_dir) / "training_results.json"
        payload = json.loads(results_path.read_text())
        payload["job_name"] = name
        payload["output_dir"] = out_dir
        return payload

    @modal.method()
    def lm_eval(
        self,
        *,
        experiment_name: str,
        config: str = "research/evals/configs/lm_eval_smoke.yaml",
        preset: str | None = None,
        model_path: str | None = None,
        adapter_path: str | None = None,
        compare_to: str | None = None,
    ) -> dict[str, Any]:
        """Run slm-lm-eval on base model or finetuned checkpoint."""
        # Pick up adapters committed by another container (e.g. a separate
        # eval-only invocation) — the warm container's mount may predate them.
        reload_volumes()

        if adapter_path:
            adapter_dir = Path(adapter_path)
            adapter_cfg = adapter_dir / "adapter_config.json"
            if not adapter_cfg.is_file():
                raise FileNotFoundError(
                    f"LoRA adapter not visible at {adapter_path} "
                    f"(missing {adapter_cfg.name})."
                )

        cmd = build_lm_eval_cmd(
            experiment_name=experiment_name,
            config=config,
            preset=preset,
            model_path=model_path,
            adapter_path=adapter_path,
            compare_to=compare_to,
        )
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd="/repo", check=False, env=repo_env())

        commit_volumes()

        out_root = Path(LM_EVAL_OUTPUT) / experiment_name
        results_json = out_root / "results.json"
        summary_md = out_root / "summary.md"
        comparison_md = out_root / "comparison.md"

        return {
            "experiment_name": experiment_name,
            "config": config,
            "preset": preset,
            "model_path": model_path,
            "adapter_path": adapter_path,
            "compare_to": compare_to,
            "results_json": str(results_json),
            "summary_md": str(summary_md),
            "comparison_md": str(comparison_md) if comparison_md.is_file() else None,
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0,
        }

    @modal.method()
    def check_gate(
        self,
        *,
        candidate_results_path: str,
        baseline_results_path: str | None,
        goals: dict[str, Any],
    ) -> dict[str, Any]:
        """Check a candidate's lm-eval results against `goals` (Hub publish gate)."""
        return check_gate_files(
            candidate_results_path=candidate_results_path,
            baseline_results_path=baseline_results_path,
            goals=goals,
        )

    @modal.method()
    def publish_adapter(
        self,
        *,
        job: dict[str, Any],
        adapter_dir: str,
        gate_result: dict[str, Any],
        candidate_results_path: str,
        baseline_results_path: str | None,
    ) -> dict[str, Any]:
        """Write a model card and push the adapter to the Hub, but only if the gate passed."""
        return publish_adapter_files(
            job=job,
            adapter_dir=adapter_dir,
            gate_result=gate_result,
            candidate_results_path=candidate_results_path,
            baseline_results_path=baseline_results_path,
        )

    @modal.method()
    def run_pipeline(
        self,
        *,
        job_names: list[str] | None = None,
        category: str | None = None,
        max_steps: int | None = None,
        train: bool = True,
        eval_only: bool = False,
        publish: bool = True,
    ) -> dict[str, Any]:
        """Per-profile baselines -> finetune -> eval -> gate -> publish (same container)."""
        spec = load_experiments()
        defaults = spec.get("defaults", {})
        jobs = spec.get("finetune", [])

        if job_names:
            jobs = [j for j in jobs if j.get("name") in job_names]
            if not jobs:
                raise ValueError(f"No matching jobs in experiments.yaml: {job_names}")
        if category:
            jobs = [j for j in jobs if j.get("category") == category]
            if not jobs:
                raise ValueError(f"No jobs with category {category!r}")
        if not jobs:
            raise ValueError("No jobs matched job_names/category")

        preset = defaults.get("preset", "minicpm5-1b")
        prepared: list[dict[str, Any]] = []
        for raw in jobs:
            merged = apply_defaults(raw, defaults)
            if max_steps is not None:
                merged["max_steps"] = max_steps
            prepared.append(merged)

        profiles = sorted({j.get("eval_profile", "compare_study") for j in prepared})

        baselines_ok: dict[str, bool] = {}
        if not eval_only:
            for profile in profiles:
                result = self.lm_eval.local(
                    experiment_name=f"{preset}__baseline__{profile}",
                    config=config_for_profile(profile),
                    preset=preset,
                )
                baselines_ok[profile] = bool(result.get("ok"))

        train_results: dict[str, dict[str, Any]] = {}
        if train and not eval_only:
            for j in prepared:
                train_results[j["name"]] = self.finetune.local(j)

        rows: list[dict[str, Any]] = []
        for j in prepared:
            job_name = j["name"]
            profile = j.get("eval_profile", "compare_study")
            train_payload = train_results.get(job_name)
            adapter_path = (
                train_payload["output_dir"]
                if train_payload
                else f"{FINETUNE_VOL_PATH}/{job_name}"
            )

            baseline_path = f"{LM_EVAL_OUTPUT}/{preset}__baseline__{profile}/results.json"
            compare_to = baseline_path if baselines_ok.get(profile) else None

            exp_name = f"{job_name}__{profile}"
            eval_result = self.lm_eval.local(
                experiment_name=exp_name,
                config=config_for_profile(profile),
                model_path=BASE_MODEL_ID,
                adapter_path=adapter_path,
                compare_to=compare_to,
            )

            row: dict[str, Any] = {
                "name": job_name,
                "category": j.get("category"),
                "profile": profile,
                "eval": eval_result,
            }

            gate_result: dict[str, Any] | None = None
            if j.get("goals"):
                if eval_result.get("ok"):
                    gate_result = self.check_gate.local(
                        candidate_results_path=eval_result["results_json"],
                        baseline_results_path=baseline_path,
                        goals=j["goals"],
                    )
                row["gate"] = gate_result

            if j.get("publish") and publish and gate_result is not None:
                row["publish"] = self.publish_adapter.local(
                    job=j,
                    adapter_dir=adapter_path,
                    gate_result=gate_result,
                    candidate_results_path=eval_result["results_json"],
                    baseline_results_path=baseline_path,
                )

            rows.append(row)

        return {"jobs": rows}


def _worker() -> GpuWorker:
    """Prefer deployed warm worker; fall back to ephemeral cls for first deploy."""
    try:
        cls = modal.Cls.from_name(APP_NAME, "GpuWorker")
        return cls()
    except modal.exception.NotFoundError:
        return GpuWorker()


@app.local_entrypoint()
def main(
    serve: bool = True,
    hours: float = DEFAULT_KEEPALIVE_HOURS,
    cmd: str | None = None,
    job: str | None = None,
    category: str | None = None,
    max_steps: int | None = None,
    eval_only: bool = False,
    pipeline: bool = False,
    publish: bool = True,
    publish_only: bool = False,
    pull: bool = True,
    ping: bool = False,
):
    """
    GPU worker CLI.

    With no task flags, keeps one container alive (default). With --job/--category,
    --cmd, --eval-only, --pipeline, or --publish-only, runs that task on the warm
    worker instead. --pipeline (and --job/--category/--eval-only) run the skill-matrix
    pipeline: per-profile baselines -> finetune -> eval -> gate -> publish.

    Examples:
        modal deploy research/modal/server_app.py
        modal run research/modal/server_app.py
        modal run research/modal/server_app.py --pipeline --job math-lora --max-steps 20
        modal run research/modal/server_app.py --pipeline --category science --no-publish
        modal run research/modal/server_app.py --eval-only --job math-lora
        modal run research/modal/server_app.py --publish-only --job math-lora
        modal run research/modal/server_app.py --cmd "uv run python research/finetune.py --help"
    """
    has_task = bool(cmd or job or category or eval_only or pipeline or publish_only or ping)
    if has_task:
        serve = False

    worker = _worker()

    if ping:
        print(json.dumps(worker.ping.remote(), indent=2))
        return

    if cmd:
        argv = shlex.split(cmd)
        result = worker.exec_cmd.remote(argv)
        if result.get("stdout"):
            print(result["stdout"], end="")
        if result.get("stderr"):
            print(result["stderr"], end="", file=__import__("sys").stderr)
        if not result.get("ok"):
            raise SystemExit(result.get("exit_code", 1))
        return

    if publish_only:
        if not job:
            raise SystemExit("--publish-only requires --job")
        defaults, prepared = prepare_jobs(job=job)
        j = prepared[0]
        if not j.get("goals") or not j.get("publish"):
            raise SystemExit(f"Job {job!r} needs `goals` and `publish` in experiments.yaml")

        preset = defaults.get("preset", "minicpm5-1b")
        profile = j.get("eval_profile", "compare_study")
        adapter_path = f"{FINETUNE_VOL_PATH}/{job}"
        candidate_results_path = f"{LM_EVAL_OUTPUT}/{job}__{profile}/results.json"
        baseline_results_path = f"{LM_EVAL_OUTPUT}/{preset}__baseline__{profile}/results.json"

        gate_result = worker.check_gate.remote(
            candidate_results_path=candidate_results_path,
            baseline_results_path=baseline_results_path,
            goals=j["goals"],
        )
        print(json.dumps(gate_result, indent=2))

        result = worker.publish_adapter.remote(
            job=j,
            adapter_dir=adapter_path,
            gate_result=gate_result,
            candidate_results_path=candidate_results_path,
            baseline_results_path=baseline_results_path,
        )
        print(json.dumps(result, indent=2))
        return

    if pipeline or job or category or eval_only:
        job_names = [job] if job else None
        result = worker.run_pipeline.remote(
            job_names=job_names,
            category=category,
            max_steps=max_steps,
            train=not eval_only,
            eval_only=eval_only,
            publish=publish,
        )
        print(json.dumps(result, indent=2))

        if pull:
            for row in result.get("jobs", []):
                pull_artifacts(row["name"], f"{row['name']}__{row['profile']}")
        return

    if serve:
        print(
            f"Keeping GpuWorker alive for {hours}h "
            f"(deploy with `modal deploy` so other terminals reuse this container)"
        )
        worker.ping.remote()
        result = worker.keep_alive.remote(hours=hours)
        print(json.dumps(result, indent=2))
        return

    raise SystemExit(
        "Nothing to do. Use default serve mode, or pass --job, --category, --cmd, "
        "--pipeline, --eval-only, --publish-only, or --ping."
    )
