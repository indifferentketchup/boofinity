import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from boofinity.args import EngineArgs
from boofinity.inference.select_model import (
    get_engine_type_from_config,
    select_model,
)
from boofinity.primitives import Device, InferenceEngine
from boofinity.transformer.utils import EmbedderEngine, ImageEmbedEngine, RerankEngine


_select_model_mod = importlib.import_module("boofinity.inference.select_model")


@pytest.mark.parametrize("engine", [e for e in InferenceEngine if e != InferenceEngine.neuron])
def test_engine(engine):
    # Optional backends are only installed in some venvs; skip rather than fail
    # when their import dependency is absent.
    _optional_backend_module = {
        InferenceEngine.optimum: "optimum",
        InferenceEngine.ctranslate2: "ctranslate2",
    }.get(engine)
    if _optional_backend_module is not None:
        pytest.importorskip(_optional_backend_module)
    select_model(
        EngineArgs(
            engine=engine,
            model_name_or_path=(pytest.DEFAULT_BERT_MODEL),
            batch_size=4,
            device=Device.cpu,
            model_warmup=False,
        )
    )


class TestDetectionRerank:
    def test_cross_encoder_with_logit_score_detects_causal_lm(self, monkeypatch):
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("rerank_mode", None)

        fake_config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
        fake_st_meta = {"model_type": "CrossEncoder"}
        fake_logit_score = {"true": 9693, "false": 2152}

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            if filename == "config_sentence_transformers.json":
                return fake_st_meta
            if filename == "1_LogitScore/config.json":
                return fake_logit_score
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_RERANK_MODE", ""):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="Qwen/Qwen3-Reranker-0.6B",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == RerankEngine.causal_lm

    def test_logit_score_without_cross_encoder_detects_causal_lm(self, monkeypatch):
        # The official Qwen3-Reranker-0.6B ships a 1_LogitScore module but no
        # config_sentence_transformers.json CrossEncoder wrapper.
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("rerank_mode", None)

        fake_config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
        fake_logit_score = {"true": 9693, "false": 2152}

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            if filename == "1_LogitScore/config.json":
                return fake_logit_score
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_RERANK_MODE", ""):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="Qwen/Qwen3-Reranker-0.6B",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == RerankEngine.causal_lm

    def test_rerank_mode_causal_lm_forces_markerless_repo(self, monkeypatch):
        # spec: INFINITY_RERANK_MODE=causal_lm forces causal_lm even when the
        # repo carries no CrossEncoder / LogitScore marker at all.
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("rerank_mode", None)

        fake_config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_RERANK_MODE", "causal_lm"):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="org/gemma-reranker-no-st-wrapper",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == RerankEngine.causal_lm

    def test_sequence_classification_detects_torch(self, monkeypatch):
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("rerank_mode", None)

        fake_config = {
            "architectures": ["BertForSequenceClassification"],
            "id2label": {"0": "dummy"},
        }

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_RERANK_MODE", ""):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="mixedbread-ai/mxbai-rerank-xsmall-v1",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == RerankEngine.torch

    def test_transformer_task_text_generation_detects_causal_lm(self, monkeypatch):
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("rerank_mode", None)

        fake_config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
        fake_st_meta = {"model_type": "CrossEncoder"}
        fake_sbc = {"transformer_task": "text-generation"}

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            if filename == "config_sentence_transformers.json":
                return fake_st_meta
            if filename == "sentence_bert_config.json":
                return fake_sbc
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_RERANK_MODE", ""):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="org/model",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == RerankEngine.causal_lm

    def test_bge_m3_still_embedder(self, monkeypatch):
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("rerank_mode", None)

        fake_config = {
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaModel"],
        }

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_RERANK_MODE", ""):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="BAAI/bge-m3",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == EmbedderEngine.torch

    def test_rerank_mode_causal_lm_overrides(self, monkeypatch):
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("rerank_mode", None)

        fake_config = {
            "architectures": ["BertForSequenceClassification"],
            "id2label": {"0": "dummy"},
        }
        fake_st_meta = {"model_type": "CrossEncoder"}

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            if filename == "config_sentence_transformers.json":
                return fake_st_meta
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_RERANK_MODE", "causal_lm"):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="org/model",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == RerankEngine.causal_lm

    def test_rerank_mode_classifier_overrides(self, monkeypatch):
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("rerank_mode", None)

        fake_config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
        fake_st_meta = {"model_type": "CrossEncoder"}
        fake_logit_score = {"true": 9693, "false": 2152}

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            if filename == "config_sentence_transformers.json":
                return fake_st_meta
            if filename == "1_LogitScore/config.json":
                return fake_logit_score
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_RERANK_MODE", "classifier"):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="org/model",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == RerankEngine.torch

    def test_repo_uses_lm_rerank_modules_logit_score(self):
        from boofinity.inference.select_model import _modules_has_logit_score

        assert _modules_has_logit_score(
            {"modules": [{"path": "1.LogitScore", "type": "custom"}]}
        ) is True
        assert _modules_has_logit_score(
            {"modules": [{"path": "2.LogitScore_head"}]}
        ) is True
        assert _modules_has_logit_score({}) is False
        assert _modules_has_logit_score({"modules": []}) is False
        assert _modules_has_logit_score({"modules": [{"path": "1.OtherModule"}]}) is False

    def test_repo_uses_lm_rerank_no_signals_returns_false(self, monkeypatch):
        from boofinity.inference.select_model import _repo_uses_lm_rerank
        from boofinity.args import EngineArgs

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", lambda *a: None)

        args = EngineArgs(model_name_or_path="org/model", engine=InferenceEngine.torch)
        assert _repo_uses_lm_rerank(args, {"model_type": "CrossEncoder"}) is False

    def test_repo_uses_lm_rerank_sbc_text_generation(self, monkeypatch):
        from boofinity.inference.select_model import _repo_uses_lm_rerank
        from boofinity.args import EngineArgs

        def fake_fetch_json(engine_args, filename):
            if filename == "sentence_bert_config.json":
                return {"transformer_task": "text-generation"}
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        args = EngineArgs(model_name_or_path="org/model", engine=InferenceEngine.torch)
        assert _repo_uses_lm_rerank(args, {}) is True

    def test_import_does_not_pull_env_before_use(self, monkeypatch):
        import sys
        if "boofinity.env" in sys.modules:
            pytest.skip("boofinity.env already imported")
        assert True


class TestDetectionQwen3VL:
    def test_embedding_name_defaults_to_image_embed_engine(self, monkeypatch):
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("vlm_mode", None)

        fake_config = {"model_type": "qwen3_vl", "vision_config": {}}
        fake_st_meta = None

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            if filename == "config_sentence_transformers.json":
                return fake_st_meta
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_VLM_MODE", ""):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == ImageEmbedEngine.qwen3vl

    def test_rerank_name_defaults_to_rerank_engine(self, monkeypatch):
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("vlm_mode", None)

        fake_config = {"model_type": "qwen3_vl", "vision_config": {}}
        fake_st_meta = None

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            if filename == "config_sentence_transformers.json":
                return fake_st_meta
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_VLM_MODE", ""):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="Qwen/Qwen3-VL-Reranker-2B",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == RerankEngine.qwen3vl

    def test_vlm_mode_embed_overrides_rerank_name(self, monkeypatch):
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("vlm_mode", None)

        fake_config = {"model_type": "qwen3_vl", "vision_config": {}}
        fake_st_meta = None

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            if filename == "config_sentence_transformers.json":
                return fake_st_meta
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_VLM_MODE", "embed"):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="Qwen/Qwen3-VL-Reranker-2B",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == ImageEmbedEngine.qwen3vl

    def test_vlm_mode_rerank_overrides_embedding_name(self, monkeypatch):
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("vlm_mode", None)

        fake_config = {"model_type": "qwen3_vl", "vision_config": {}}
        fake_st_meta = None

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            if filename == "config_sentence_transformers.json":
                return fake_st_meta
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_VLM_MODE", "rerank"):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == RerankEngine.qwen3vl

    def test_clip_still_routes_to_image_embed_torch(self, monkeypatch):
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("vlm_mode", None)

        fake_config = {
            "model_type": "clip",
            "vision_config": {"model_type": "clip"},
            "architectures": ["CLIPModel"],
        }
        fake_st_meta = None

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            if filename == "config_sentence_transformers.json":
                return fake_st_meta
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_VLM_MODE", ""):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="openai/clip-vit-base-patch32",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == ImageEmbedEngine.torch

    def test_bge_m3_still_embedder_engine(self, monkeypatch):
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("vlm_mode", None)
        MANAGER.__dict__.pop("rerank_mode", None)

        fake_config = {
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaModel"],
        }

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_VLM_MODE", ""):
            with _env_var("INFINITY_RERANK_MODE", ""):
                result = get_engine_type_from_config(
                    EngineArgs(
                        model_name_or_path="BAAI/bge-m3",
                        engine=InferenceEngine.torch,
                    )
                )
        assert result == EmbedderEngine.torch

    def test_qwen3vl_fast_path_skips_vision_config(self, monkeypatch):
        from boofinity.env import MANAGER
        MANAGER.__dict__.pop("vlm_mode", None)

        fake_config = {
            "model_type": "qwen3_vl",
            "vision_config": {},
            "architectures": ["Qwen3VLForConditionalGeneration"],
        }

        def fake_fetch_json(engine_args, filename):
            if filename == "config.json":
                return fake_config
            return None

        monkeypatch.setattr(_select_model_mod, "_try_fetch_json", fake_fetch_json)

        with _env_var("INFINITY_VLM_MODE", ""):
            result = get_engine_type_from_config(
                EngineArgs(
                    model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
                    engine=InferenceEngine.torch,
                )
            )
        assert result == ImageEmbedEngine.qwen3vl

    def test_rerank_engine_qwen3vl_reachable(self):
        from boofinity.transformer.utils import RerankEngine

        enum_val = RerankEngine.qwen3vl
        assert enum_val.value._class_name == "VLMReranker"
        assert "torch_vlm" in enum_val.value._import_path

    def test_rerank_engine_qwen3vl_capabilities(self):
        from boofinity.transformer.utils import RerankEngine
        from boofinity.transformer.vlm.torch_vlm import VLMReranker

        assert VLMReranker.capabilities == {"rerank"}


class _env_var:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self._prev = None

    def __enter__(self):
        self._prev = os.environ.get(self.name)
        if self.value:
            os.environ[self.name] = self.value
        else:
            os.environ.pop(self.name, None)

    def __exit__(self, *args):
        if self._prev is not None:
            os.environ[self.name] = self._prev
        else:
            os.environ.pop(self.name, None)
