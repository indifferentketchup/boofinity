# SPDX-License-Identifier: MIT
"""_try_fetch_json must treat 'not found' as 'not this model type' (return None)
while letting real network/auth errors propagate instead of masking them."""
from __future__ import annotations

import importlib

import pytest

from boofinity.args import EngineArgs
from boofinity.primitives import Device, InferenceEngine

# The module and a function inside it share the name `select_model`, so a plain
# `import ... as sm` binds the function; import the module explicitly.
sm = importlib.import_module("boofinity.inference.select_model")


def _args(path: str) -> EngineArgs:
    return EngineArgs(
        engine=InferenceEngine.torch,
        model_name_or_path=path,
        batch_size=4,
        device=Device.cpu,
        model_warmup=False,
    )


def test_repository_not_found_returns_none(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import RepositoryNotFoundError

    def boom(*a, **k):
        raise RepositoryNotFoundError("repo missing")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    assert sm._try_fetch_json(_args("org/does-not-exist"), "config.json") is None


def test_entry_not_found_returns_none(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    def boom(*a, **k):
        raise EntryNotFoundError("file missing")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    assert sm._try_fetch_json(_args("org/repo"), "config.json") is None


def test_http_error_propagates(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    def boom(*a, **k):
        raise HfHubHTTPError("500 server error")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    with pytest.raises(HfHubHTTPError):
        sm._try_fetch_json(_args("org/repo"), "config.json")


def test_local_malformed_json_returns_none(tmp_path):
    (tmp_path / "config.json").write_text("{ not valid json ")
    assert sm._try_fetch_json(_args(str(tmp_path)), "config.json") is None


def test_local_missing_file_returns_none(tmp_path):
    assert sm._try_fetch_json(_args(str(tmp_path)), "config.json") is None
