from inference.device_utils import is_cuda_oom, iter_inference_device_plans


def test_is_cuda_oom_matches_pytorch_message() -> None:
    exc = RuntimeError(
        "CUDA out of memory. Tried to allocate 384.00 MiB. "
        "GPU 0 has a total capacity of 3.68 GiB of which 350.19 MiB is free."
    )
    assert is_cuda_oom(exc)


def test_is_cuda_oom_rejects_other_errors() -> None:
    assert not is_cuda_oom(RuntimeError("disk full"))


def test_iter_inference_device_plans_includes_cpu(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    plans = list(iter_inference_device_plans())
    assert len(plans) == 1
    assert plans[0].device == "cpu"
