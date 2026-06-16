import copy
import json
import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from boofinity.args import EngineArgs
from boofinity.primitives import Device, InferenceEngine
from boofinity.transformer.crossencoder.lm_torch import (
    CausalLMReranker,
    LMProfile,
    _QWEN3_RERANKER_PREFIX,
    _QWEN3_RERANKER_SUFFIX,
    _QWEN3_YES_ID,
    _QWEN3_NO_ID,
    _QWEN3_MAX_LENGTH,
    _QWEN3_RERANKER_DEFAULT_INSTRUCTION,
)


class TestLMProfile:
    def test_qwen3_reranker_constants(self):
        profile = LMProfile.qwen3_reranker()
        assert profile.yes_id == _QWEN3_YES_ID
        assert profile.no_id == _QWEN3_NO_ID
        assert profile.max_length == _QWEN3_MAX_LENGTH
        assert "<|im_start|>" in profile.prefix
        assert "<|im_start|>assistant\n" in profile.suffix
        assert profile.default_instruction == _QWEN3_RERANKER_DEFAULT_INSTRUCTION

    def test_from_repo_with_logit_score_config(self):
        fake_config = {
            "true": 9999,
            "false": 1111,
            "max_length": 4096,
            "default_instruction": "Custom instruction",
        }

        with patch(
            "boofinity.transformer.crossencoder.lm_torch.AutoTokenizer"
        ) as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.convert_tokens_to_ids.return_value = None
            mock_tok.unk_token_id = 0
            mock_tok_cls.from_pretrained.return_value = mock_tok

            with patch(
                "boofinity.transformer.crossencoder.lm_torch._fetch_logit_score_config",
                return_value=fake_config,
            ):
                profile = LMProfile.from_repo("org/model", revision=None)
                assert profile.yes_id == 9999
                assert profile.no_id == 1111
                assert profile.max_length == 4096
                assert profile.default_instruction == "Custom instruction"

    def test_from_repo_fallback_when_file_missing(self):
        with patch(
            "boofinity.transformer.crossencoder.lm_torch._fetch_logit_score_config",
            side_effect=OSError("not found"),
        ):
            profile = LMProfile.from_repo("org/model", revision=None)
            assert profile.yes_id == _QWEN3_YES_ID
            assert profile.no_id == _QWEN3_NO_ID
            assert profile.max_length == _QWEN3_MAX_LENGTH


class FakeTokenizer:
    def __init__(self):
        self.padding_side = "left"
        self.pad_token_id = 0
        self.unk_token_id = None

    def encode(self, text, add_special_tokens=True, truncation=True, max_length=None):
        return [1] * min(len(text), max_length or 9999)

    def pad(self, all_input_ids, padding=True, return_tensors="pt"):
        import torch
        max_len = max(len(d["input_ids"]) for d in all_input_ids)
        batch_ids = []
        batch_mask = []
        for d in all_input_ids:
            ids = d["input_ids"]
            pad_len = max_len - len(ids)
            padded = [self.pad_token_id] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)
            batch_ids.append(padded)
            batch_mask.append(mask)
        return {
            "input_ids": torch.tensor(batch_ids),
            "attention_mask": torch.tensor(batch_mask),
        }

    def batch_encode_plus(self, sentences, **kwargs):
        class FakeEncoding:
            def __init__(self, tokens):
                self.tokens = tokens

        return type(
            "Result",
            (),
            {"encodings": [FakeEncoding(list(s)) for s in sentences]},
        )()

    def convert_tokens_to_ids(self, token):
        return None


class TestCausalLMReranker:
    def test_constructor_stores_profile_fields(self):
        reranker = CausalLMReranker.__new__(CausalLMReranker)
        reranker.tokenizer = FakeTokenizer()
        reranker.model = MagicMock()
        reranker._infinity_tokenizer = FakeTokenizer()
        reranker.profile = LMProfile.qwen3_reranker()
        reranker._prefix_ids = [1, 2, 3]
        reranker._suffix_ids = [7, 8, 9]
        assert reranker.profile.yes_id == _QWEN3_YES_ID
        assert reranker.profile.no_id == _QWEN3_NO_ID
        assert reranker.profile.max_length == _QWEN3_MAX_LENGTH
        assert isinstance(reranker.profile.default_instruction, str)
        assert len(reranker.profile.default_instruction) > 0

    def test_encode_pre_left_padded_batch(self):
        reranker = CausalLMReranker.__new__(CausalLMReranker)
        tokenizer = FakeTokenizer()
        reranker.tokenizer = tokenizer
        reranker.model = MagicMock()
        reranker.model.device = "cpu"
        reranker.profile = LMProfile.qwen3_reranker()
        reranker._prefix_ids = [100]
        reranker._suffix_ids = [200]
        reranker._infinity_tokenizer = tokenizer

        pairs = [("query text", "document text")] * 2
        result = reranker.encode_pre(pairs)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape[0] == 2
        assert result["input_ids"][:, -1].tolist() == [200, 200]
        assert result["attention_mask"].sum(dim=1).tolist() == [
            result["input_ids"].shape[1],
            result["input_ids"].shape[1],
        ]

    def test_encode_core_shape_and_dtype(self):
        import torch

        reranker = CausalLMReranker.__new__(CausalLMReranker)
        reranker.profile = LMProfile.qwen3_reranker()
        fake_model = MagicMock()
        B, T, V = 2, 10, 50000
        fake_logits = torch.randn(B, T, V)
        fake_model.return_value = type("Out", (), {"logits": fake_logits})()
        reranker.model = fake_model

        features = {
            "input_ids": torch.randint(0, V, (B, T)),
            "attention_mask": torch.ones(B, T),
        }
        result = reranker.encode_core(features)
        assert result.shape == (B, 2)
        assert result.dtype == torch.float32
        assert str(result.device) == "cpu"

    def test_encode_post_returns_logit(self):
        import torch

        reranker = CausalLMReranker.__new__(CausalLMReranker)
        out = torch.tensor([[-2.0, 2.0], [0.0, 0.0], [2.0, -2.0]])
        result = reranker.encode_post(out)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)
        assert result[0] > result[2]
        assert abs(result[1]) < 0.01

    def test_tokenize_lengths_list_of_str(self):
        reranker = CausalLMReranker.__new__(CausalLMReranker)
        tokenizer = FakeTokenizer()
        reranker._infinity_tokenizer = tokenizer
        lengths = reranker.tokenize_lengths(["short", "much longer text here"])
        assert isinstance(lengths, list)
        assert len(lengths) == 2
        assert all(isinstance(v, int) for v in lengths)
        assert lengths[0] < lengths[1]

    def test_no_quant_embedding_decorator_on_encode_post(self):
        import inspect

        source = inspect.getsource(CausalLMReranker.encode_post)
        assert "@quant_embedding_decorator" not in source

    def test_warmup_inherited_from_base(self):
        from boofinity.transformer.abstract import BaseCrossEncoder

        assert hasattr(BaseCrossEncoder, "warmup")
        with pytest.raises(TypeError):
            CausalLMReranker()

    def test_import_succeeds(self):
        assert CausalLMReranker is not None
        assert LMProfile is not None

    @pytest.mark.parametrize(
        "text",
        [
            "short text",
            "longer text with more characters",
        ],
    )
    def test_tokenize_lengths_returns_int_list(self, text):
        reranker = CausalLMReranker.__new__(CausalLMReranker)
        tokenizer = FakeTokenizer()
        reranker._infinity_tokenizer = tokenizer
        result = reranker.tokenize_lengths([text])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], int)
        assert result[0] > 0


class TestCausalLMRerankerNeedsNetwork:
    @pytest.mark.needs_network
    def test_real_qwen3_reranker(self):
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            pytest.skip("HF_HUB_OFFLINE=1")
        try:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
        except Exception:
            pytest.skip("Qwen/Qwen3-Reranker-0.6B not available")

        args = EngineArgs(
            model_name_or_path="Qwen/Qwen3-Reranker-0.6B",
            engine=InferenceEngine.torch,
            device=Device.cpu,
            dtype="float32",
            model_warmup=False,
        )
        reranker = CausalLMReranker(engine_args=args)

        query = "Where is Munich?"
        documents = [
            "Berlin is the capital of Germany.",
            "Munich is in Germany.",
            "Paris is in France.",
            "Munich is famous for its beer.",
        ]
        pairs = [(query, doc) for doc in documents]
        pre = reranker.encode_pre(pairs)
        core = reranker.encode_core(pre)
        scores = reranker.encode_post(core)

        assert len(scores) == 4
        assert isinstance(scores, list)
        assert all(isinstance(v, float) for v in scores)
        top_idx = scores.index(max(scores))
        assert "Munich" in documents[top_idx]


class TestEnvRerankMode:
    def test_rerank_mode_default(self):
        from boofinity.env import MANAGER

        MANAGER.__dict__.pop("rerank_mode", None)
        with _env_var("BOOFINITY_RERANK_MODE", ""):
            assert MANAGER.rerank_mode == "auto"

    def test_rerank_mode_causal_lm(self):
        from boofinity.env import MANAGER

        MANAGER.__dict__.pop("rerank_mode", None)
        with _env_var("BOOFINITY_RERANK_MODE", "causal_lm"):
            assert MANAGER.rerank_mode == "causal_lm"

    def test_rerank_mode_classifier(self):
        from boofinity.env import MANAGER

        MANAGER.__dict__.pop("rerank_mode", None)
        with _env_var("BOOFINITY_RERANK_MODE", "classifier"):
            assert MANAGER.rerank_mode == "classifier"

    def test_rerank_mode_bogus_falls_back_to_auto(self):
        from boofinity.env import MANAGER

        MANAGER.__dict__.pop("rerank_mode", None)
        with _env_var("BOOFINITY_RERANK_MODE", "bogus"):
            assert MANAGER.rerank_mode == "auto"


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
