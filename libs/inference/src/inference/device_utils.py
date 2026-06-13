"""CUDA / CPU device selection and OOM helpers for inference backends."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Iterator, Literal

DevicePreference = Literal["auto", "cuda", "cpu"]


@dataclass(frozen=True)
class DevicePlan:
    device: str
    torch_dtype_name: str
    device_map: str | dict[str, int] | None
    label: str


def inference_device_preference() -> str:
    return os.environ.get("INFERENCE_DEVICE", "auto").strip().lower()


def is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "cuda out of memory" in msg or "cudaoom" in msg.replace("_", "")


def clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    torch.cuda.synchronize()


def iter_inference_device_plans() -> Iterator[DevicePlan]:
    """Yield load plans: each CUDA device (if allowed), then CPU."""
    pref = inference_device_preference()
    if pref == "cpu":
        yield DevicePlan("cpu", "float32", None, "cpu")
        return

    try:
        import torch
    except ImportError:
        yield DevicePlan("cpu", "float32", None, "cpu")
        return

    if pref in ("cuda", "auto") and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            yield DevicePlan(
                f"cuda:{index}",
                "float16",
                {"": index},
                f"cuda:{index} ({name})",
            )

    if pref in ("auto", "cpu"):
        yield DevicePlan("cpu", "float32", None, "cpu")
