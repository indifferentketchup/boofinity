from boofinity.args import EngineArgs
from boofinity.primitives import Device, InferenceEngine


def test_EngineArgs_no_input():
    EngineArgs()


def test_engine_args():
    args = EngineArgs(
        model_name_or_path="michaelfeil/bge-small-en-v1.5",
        batch_size=64,
        revision=None,
        trust_remote_code=True,
        engine="torch",
        model_warmup=False,
        vector_disk_cache_path="",
        device="cpu",
        lengths_via_tokenize=False,
    )

    assert args.model_name_or_path == "michaelfeil/bge-small-en-v1.5"
    assert args.batch_size == 64
    assert args.revision is None
    assert args.trust_remote_code
    assert args.engine == InferenceEngine.torch
    assert not args.model_warmup
    assert args.vector_disk_cache_path == ""
    assert args.device == Device.cpu
    assert not args.lengths_via_tokenize


def test_multiargs():
    EngineArgs.from_env()


def test_enable_webgpu_ep_defaults_false():
    """Task 9.1: the WebGPU EP opt-in defaults to False."""
    args = EngineArgs(model_name_or_path="michaelfeil/bge-small-en-v1.5")
    assert args.enable_webgpu_ep is False


def test_enable_webgpu_ep_explicit_true():
    args = EngineArgs(
        model_name_or_path="michaelfeil/bge-small-en-v1.5",
        enable_webgpu_ep=True,
    )
    assert args.enable_webgpu_ep is True


def test_from_env_picks_up_webgpu_flag(monkeypatch):
    """Task 9.1: BOOFINITY_WEBGPU_EP=true is wired through from_env()."""
    from boofinity.env import MANAGER

    monkeypatch.setenv("BOOFINITY_WEBGPU_EP", "true")
    # The MANAGER caches env reads; drop the cached value so the new env is read.
    MANAGER.__dict__.pop("webgpu_ep", None)
    try:
        args_list = EngineArgs.from_env()
        assert all(a.enable_webgpu_ep is True for a in args_list)
    finally:
        MANAGER.__dict__.pop("webgpu_ep", None)
