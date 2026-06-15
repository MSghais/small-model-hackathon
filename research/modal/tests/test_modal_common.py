from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.modal._common import (  # noqa: E402
    COMMON_ENV,
    build_finetune_cmd,
    build_lm_eval_cmd,
    prepare_jobs,
    split_csv,
)


def test_build_lm_eval_cmd_accepts_runtime_overrides():
    cmd = build_lm_eval_cmd(
        experiment_name="exp",
        config="cfg.yaml",
        preset="minicpm5-1b",
        tasks=["arc_easy", "hellaswag"],
        limit=5,
        num_fewshot=1,
        batch_size="2",
        device="cuda",
        dtype="float16",
        seed=7,
    )

    assert cmd[-15:] == [
        "--tasks",
        "arc_easy",
        "hellaswag",
        "--limit",
        "5",
        "--num-fewshot",
        "1",
        "--batch-size",
        "2",
        "--device",
        "cuda",
        "--dtype",
        "float16",
        "--seed",
        "7",
    ]


def test_prepare_jobs_filters_and_applies_finetune_overrides():
    _, jobs = prepare_jobs(
        sector="math",
        profiles=["math"],
        max_steps=3,
        max_samples=11,
        finetune_overrides={"lr": 1e-4, "lora_r": 8, "dataset_split": "train[:11]"},
    )

    assert [job["name"] for job in jobs] == ["math-lora"]
    job = jobs[0]
    assert job["max_steps"] == 3
    assert job["max_samples"] == 11
    assert job["dataset_split"] == "train[:11]"
    assert job["args"]["lr"] == 1e-4
    assert job["args"]["lora_r"] == 8

    cmd = build_finetune_cmd(job, "/tmp/out")
    assert "--max_steps" in cmd
    assert cmd[cmd.index("--max_steps") + 1] == "3"
    assert "--lr" in cmd
    assert cmd[cmd.index("--lr") + 1] == "0.0001"
    assert "--lora_r" in cmd
    assert cmd[cmd.index("--lora_r") + 1] == "8"


def test_split_csv_trims_empty_values():
    assert split_csv(" math, science ,,code ") == ["math", "science", "code"]
    assert split_csv(None) is None


def test_common_env_redirects_xet_logs_off_hf_cache_volume():
    assert COMMON_ENV["HF_XET_LOG_DEST"] == "/tmp/xet-logs/"
