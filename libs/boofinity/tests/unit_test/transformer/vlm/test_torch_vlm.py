# SPDX-License-Identifier: MIT
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


class TestVLMResolveDtype:
    def test_pascal_forces_fp16(self):
        from boofinity.transformer.vlm.dtype import vlm_resolve_dtype
        from boofinity.primitives import Device, Dtype

        args = MagicMock()
        args.device = Device.cuda
        args.dtype = Dtype.auto
        import torch
        dtype = vlm_resolve_dtype(args, (6, 1))
        assert dtype == torch.float16

    def test_ampere_auto_bf16(self, monkeypatch):
        from boofinity.transformer.vlm.dtype import vlm_resolve_dtype
        from boofinity.primitives import Device, Dtype

        monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: True)

        args = MagicMock()
        args.device = Device.cuda
        args.dtype = Dtype.auto
        import torch
        dtype = vlm_resolve_dtype(args, (8, 0))
        assert dtype == torch.bfloat16

    def test_cpu_auto_fp32(self):
        from boofinity.transformer.vlm.dtype import vlm_resolve_dtype
        from boofinity.primitives import Device, Dtype

        args = MagicMock()
        args.device = Device.cpu
        args.dtype = Dtype.auto
        import torch
        dtype = vlm_resolve_dtype(args, (0, 0))
        assert dtype == torch.float32

    def test_explicit_dtype_honored(self):
        from boofinity.transformer.vlm.dtype import vlm_resolve_dtype
        from boofinity.primitives import Device, Dtype

        args = MagicMock()
        args.device = Device.cuda
        args.dtype = Dtype.float32
        import torch
        dtype = vlm_resolve_dtype(args, (8, 0))
        assert dtype == torch.float32


class TestMMEmbeddingSingle:
    def test_text_only(self):
        from boofinity.primitives import MMEmbeddingSingle

        s = MMEmbeddingSingle(text="hello")
        assert s.to_input() == "hello"
        assert s.str_repr() == "hello"

    def test_image_only(self):
        from boofinity.primitives import MMEmbeddingSingle
        from PIL import Image

        img = Image.new("RGB", (256, 256))
        s = MMEmbeddingSingle(image=img)
        assert s.to_input() is img
        # str_repr is a bounded length proxy, not a multi-KB throwaway string.
        rep = s.str_repr()
        assert rep == "img:256x256"
        assert len(rep) < 64

    def test_image_only_str_repr_is_bounded_for_large_images(self):
        from boofinity.primitives import MMEmbeddingSingle
        from PIL import Image

        img = Image.new("RGB", (4096, 4096))
        s = MMEmbeddingSingle(image=img)
        rep = s.str_repr()
        assert len(rep) < 64
        assert "4096" in rep

    def test_text_and_image(self):
        from boofinity.primitives import MMEmbeddingSingle
        from PIL import Image

        img = Image.new("RGB", (256, 256))
        s = MMEmbeddingSingle(text="hello", image=img)
        result = s.to_input()
        assert isinstance(result, tuple)
        assert result[0] == "hello"
        assert result[1] is img

    def test_empty_raises(self):
        from boofinity.primitives import MMEmbeddingSingle

        s = MMEmbeddingSingle()
        with pytest.raises(ValueError):
            s.to_input()


class TestMMEmbeddingInner:
    @pytest.mark.anyio
    async def test_get_result(self):
        import asyncio
        from boofinity.primitives import MMEmbeddingInner, MMEmbeddingSingle

        single = MMEmbeddingSingle(text="hello")
        future = asyncio.get_event_loop().create_future()
        inner = MMEmbeddingInner(content=single, future=future)
        fake = np.array([1.0, 2.0], dtype=np.float32)
        await inner.complete(fake)
        result = await inner.get_result()
        np.testing.assert_array_equal(result, fake)

    def test_get_inner_item_routes(self):
        from boofinity.primitives import MMEmbeddingInner, MMEmbeddingSingle, get_inner_item

        assert get_inner_item(MMEmbeddingSingle) == MMEmbeddingInner


class TestVLMEmbedderStub:
    @pytest.fixture
    def vlm_stub(self):
        import torch
        from boofinity.args import EngineArgs
        from boofinity.primitives import Device, Dtype

        from boofinity.transformer.vlm.torch_vlm import VLMEmbedder

        engine_args = EngineArgs(
            model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
            device=Device.cpu,
            dtype=Dtype.float32,
            model_warmup=False,
        )
        object.__setattr__(engine_args, "_matryoshka_dim", None)

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.image_processor = MagicMock()
        mock_qwen_vl_utils = MagicMock()

        vlm = VLMEmbedder.__new__(VLMEmbedder)
        vlm._default_instruction = "Represent the user's input."
        vlm.engine_args = engine_args
        vlm.model = mock_model
        vlm.processor = mock_processor
        vlm._qwen_vl_utils = mock_qwen_vl_utils
        vlm.max_length = 8192
        vlm._image_patch_size = 16
        vlm._warmup_image_size = (256, 256)
        vlm.capabilities = {"embed", "image_embed"}

        return vlm, mock_model, mock_processor, mock_qwen_vl_utils

    def test_encode_core_pooling_and_norm(self, vlm_stub):
        import torch
        vlm, mock_model, _, _ = vlm_stub

        B, T, D = 2, 10, 128
        hidden = torch.randn(B, T, D)

        def fake_forward(**features):
            result = MagicMock()
            # Qwen3VLForConditionalGeneration exposes the final decoder layer via
            # hidden_states[-1] (with output_hidden_states=True), not last_hidden_state.
            result.hidden_states = [hidden]
            return result

        mock_model.side_effect = fake_forward
        mock_model.device = torch.device("cpu")

        attn = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                             [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
        features = {"attention_mask": attn, "pixel_values": torch.randn(B, 3, 256, 256)}

        result = vlm.encode_core(features)

        assert result.shape == (B, D)
        assert result.dtype == torch.float32
        norms = result.norm(p=2, dim=1)
        for n in norms:
            assert abs(n.item() - 1.0) < 1e-5

        assert mock_model.called

    def test_encode_core_last_non_pad(self, vlm_stub):
        import torch
        import torch.nn.functional as F
        vlm, mock_model, _, _ = vlm_stub

        B, T, D = 1, 5, 4
        hidden = torch.zeros(B, T, D)
        hidden[0, 2] = 9.0
        hidden[0, 3] = 8.0
        hidden[0, 4] = 7.0

        def fake_forward(**features):
            result = MagicMock()
            result.hidden_states = [hidden]
            return result

        mock_model.side_effect = fake_forward
        mock_model.device = torch.device("cpu")

        attn = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
        features = {"attention_mask": attn}

        result = vlm.encode_core(features)
        assert result.shape == (1, D)
        idx = attn.sum(dim=1) - 1
        expected = hidden[torch.arange(hidden.size(0)), idx]
        assert torch.allclose(result, F.normalize(expected.float(), p=2, dim=1))

    def test_encode_post_matryoshka(self, vlm_stub):
        import torch
        vlm, _, _, _ = vlm_stub

        object.__setattr__(vlm.engine_args, "_matryoshka_dim", 4)
        B, D = 2, 128
        tensor = torch.randn(B, D)
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)

        result = vlm.encode_post(tensor)

        assert len(result) == B
        for r in result:
            assert isinstance(r, np.ndarray)
            assert r.shape == (4,)
            norm = np.linalg.norm(r)
            assert abs(norm - 1.0) < 1e-5

    def test_encode_post_no_matryoshka(self, vlm_stub):
        import torch
        vlm, _, _, _ = vlm_stub

        B, D = 2, 128
        tensor = torch.randn(B, D)
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)

        result = vlm.encode_post(tensor)

        assert len(result) == B
        for r in result:
            assert isinstance(r, np.ndarray)
            assert r.shape == (D,)

    def test_encode_post_dim_zero_skips(self, vlm_stub):
        import torch
        vlm, _, _, _ = vlm_stub

        object.__setattr__(vlm.engine_args, "_matryoshka_dim", 0)
        B, D = 2, 128
        tensor = torch.randn(B, D)
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)

        result = vlm.encode_post(tensor)

        assert len(result) == B
        for r in result:
            assert isinstance(r, np.ndarray)
            assert r.shape == (D,)

    def test_encode_pre_default_instruction(self, vlm_stub):
        vlm, _, mock_processor, mock_qwen = vlm_stub

        mock_qwen.process_vision_info.return_value = []
        mock_processor.apply_chat_template.return_value = (
            "<|im_start|>system\nRepresent the user's input.<|im_end|>"
        )
        mock_inputs = MagicMock()
        mock_inputs.items.return_value = [("attention_mask", MagicMock())]
        mock_processor.return_value = mock_inputs

        vlm.model.device = "cpu"

        items = ["hello"]
        features = vlm.encode_pre(items)

        assert mock_processor.apply_chat_template.called
        assert mock_processor.call_args[1]["do_resize"] is False
        assert mock_processor.call_args[1]["padding"] is True
        assert mock_processor.call_args[1]["return_tensors"] == "pt"

    def test_encode_pre_with_image(self, vlm_stub):
        from PIL import Image
        vlm, _, mock_processor, mock_qwen = vlm_stub

        mock_qwen.process_vision_info.return_value = ["fake_image_tensor"]
        mock_processor.apply_chat_template.return_value = (
            "<|im_start|>system\nRepresent the user's input.<|im_end|>"
        )
        mock_inputs = MagicMock()
        mock_inputs.items.return_value = [("attention_mask", MagicMock())]
        mock_processor.return_value = mock_inputs

        vlm.model.device = "cpu"

        img = Image.new("RGB", (256, 256))
        items = [("What is this?", img)]
        features = vlm.encode_pre(items)

        assert mock_qwen.process_vision_info.called
        assert mock_processor.called

    def test_min_embedding_import(self):
        from boofinity.primitives import (
            MMEmbeddingInner,
            MMEmbeddingSingle,
            get_inner_item,
        )

        assert get_inner_item(MMEmbeddingSingle) == MMEmbeddingInner


class TestMMItem:
    def test_text_only(self):
        from boofinity.primitives import MMItem

        item = MMItem(text="hello")
        assert item.text == "hello"
        assert item.image is None

    def test_image_only(self):
        from boofinity.primitives import MMItem
        from PIL import Image

        img = Image.new("RGB", (256, 256))
        item = MMItem(image=img)
        assert item.text is None
        assert item.image is img

    def test_both(self):
        from boofinity.primitives import MMItem
        from PIL import Image

        img = Image.new("RGB", (256, 256))
        item = MMItem(text="hello", image=img)
        assert item.text == "hello"
        assert item.image is img

    def test_empty_ok(self):
        from boofinity.primitives import MMItem

        item = MMItem()
        assert item.text is None
        assert item.image is None


class TestReRankMMSingle:
    def test_text_only_query_and_doc(self):
        from boofinity.primitives import MMItem, ReRankMMSingle

        s = ReRankMMSingle(
            query=MMItem(text="q"), document=MMItem(text="d")
        )
        result = s.to_input()
        assert isinstance(result, tuple)
        assert result[0].text == "q"
        assert result[1].text == "d"
        assert s.str_repr() == "qd"

    def test_image_only_document(self):
        from boofinity.primitives import MMItem, ReRankMMSingle
        from PIL import Image

        img = Image.new("RGB", (256, 256))
        s = ReRankMMSingle(
            query=MMItem(text="q"),
            document=MMItem(image=img),
        )
        result = s.to_input()
        assert result[0].text == "q"
        assert result[1].image is img

    def test_both_query_and_both_doc(self):
        from boofinity.primitives import MMItem, ReRankMMSingle
        from PIL import Image

        img = Image.new("RGB", (256, 256))
        s = ReRankMMSingle(
            query=MMItem(text="q", image=img),
            document=MMItem(text="d", image=img),
        )
        result = s.to_input()
        assert result[0].text == "q"
        assert result[0].image is img
        assert result[1].text == "d"


class TestReRankMMInner:
    @pytest.mark.anyio
    async def test_get_result(self):
        import asyncio
        from boofinity.primitives import MMItem, ReRankMMInner, ReRankMMSingle

        single = ReRankMMSingle(
            query=MMItem(text="q"), document=MMItem(text="d")
        )
        future = asyncio.get_event_loop().create_future()
        inner = ReRankMMInner(content=single, future=future)
        await inner.complete(0.85)
        result = await inner.get_result()
        assert result == 0.85

    def test_get_inner_item_routes(self):
        from boofinity.primitives import (
            ReRankMMInner,
            ReRankMMSingle,
            get_inner_item,
        )

        assert get_inner_item(ReRankMMSingle) == ReRankMMInner


class TestRerankProfile:
    def test_qwen3vl_defaults(self):
        from boofinity.transformer.vlm.profiles import RerankProfile

        p = RerankProfile.qwen3_vl_reranker()
        assert p.yes_id == 9693
        assert p.no_id == 2152
        assert p.max_length == 10240
        assert len(p.default_instruction) > 0
        assert "user" in p.prefix
        assert "assistant" in p.suffix

    def test_from_repo_config(self, monkeypatch):
        from boofinity.transformer.vlm.profiles import RerankProfile

        class FakeConfig:
            yes_token_id = 42
            no_token_id = 43

        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda *a, **kw: FakeConfig(),
        )

        p = RerankProfile.from_repo("test/model")
        assert p.yes_id == 42
        assert p.no_id == 43

    def test_from_repo_missing_fallsback_to_defaults(self, monkeypatch):
        from boofinity.transformer.vlm.profiles import RerankProfile

        class FakeConfigNoTokens:
            pass

        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda *a, **kw: FakeConfigNoTokens(),
        )

        class FakeTok:
            unk_token_id = 0

            def convert_tokens_to_ids(self, t):
                return self.unk_token_id

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *a, **kw: FakeTok(),
        )

        p = RerankProfile.from_repo("test/model")
        assert p.yes_id == 9693
        assert p.no_id == 2152


class TestBaseCrossEncoderMM:
    def test_import_and_capabilities(self):
        from boofinity.transformer.abstract import BaseCrossEncoderMM

        assert BaseCrossEncoderMM.capabilities == {"rerank"}

    def test_encode_pre_is_abstract(self):
        from boofinity.transformer.abstract import BaseCrossEncoderMM
        import inspect

        assert inspect.isabstract(BaseCrossEncoderMM)
        assert hasattr(BaseCrossEncoderMM, "encode_pre")
        assert getattr(BaseCrossEncoderMM.encode_pre, "__isabstractmethod__", False)


class TestVLMRerankerStub:
    @pytest.fixture
    def reranker_stub(self):
        import torch
        from boofinity.args import EngineArgs
        from boofinity.primitives import Device, Dtype
        from boofinity.transformer.vlm.torch_vlm import VLMReranker

        engine_args = EngineArgs(
            model_name_or_path="Qwen/Qwen3-VL-Reranker-2B",
            device=Device.cpu,
            dtype=Dtype.float32,
            model_warmup=False,
        )

        reranker = VLMReranker.__new__(VLMReranker)
        reranker.engine_args = engine_args
        reranker.capabilities = {"rerank"}

        V, D = 100, 128
        fake_lm_weight = torch.randn(V, D)
        yes_id = 50
        no_id = 60
        fake_lm_weight[yes_id] = torch.ones(D) * 3.0
        fake_lm_weight[no_id] = torch.ones(D) * 1.0

        fake_model = MagicMock()
        fake_model.lm_head = MagicMock()
        fake_model.lm_head.weight = fake_lm_weight
        fake_model.device = torch.device("cpu")
        fake_model.dtype = torch.float32

        class FakeBackbone:
            def __call__(self, **inputs):
                B = inputs.get("input_ids", torch.zeros(1, 1)).shape[0]
                hidden = torch.randn(B, 10, D)
                result = MagicMock()
                result.last_hidden_state = hidden
                return result

        fake_model.model = FakeBackbone()
        reranker.model = fake_model

        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.padding_side = "left"
        mock_processor.image_processor = MagicMock()
        reranker.processor = mock_processor
        reranker.tokenizer = mock_processor.tokenizer

        mock_qwen = MagicMock()
        reranker._qwen_vl_utils = mock_qwen

        from boofinity.transformer.vlm.profiles import RerankProfile
        reranker.profile = RerankProfile(
            prefix="<|im_start|>system\nTest.<|im_end|>\n<|im_start|>user\n",
            suffix="<|im_end|>\n<|im_start|>assistant\n",
            yes_id=yes_id,
            no_id=no_id,
            max_length=10240,
            default_instruction="Test instruction",
        )
        reranker._prefix_ids = [1, 2, 3]
        reranker._suffix_ids = [4, 5]
        reranker._image_patch_size = 16
        reranker._warmup_image_size = (256, 256)

        import torch.nn as nn
        reranker.score_head = nn.Linear(D, 1, bias=False)
        with torch.no_grad():
            diff = (fake_lm_weight[yes_id] - fake_lm_weight[no_id]).detach().float().unsqueeze(0)
            reranker.score_head.weight.copy_(diff)

        ls = MagicMock()
        ls.device_placement = "cpu"
        ls.quantization_dtype = None
        object.__setattr__(engine_args, "_loading_strategy", ls)

        return reranker, fake_model, mock_processor, mock_qwen

    def test_score_head_weight_equals_diff(self, reranker_stub):
        import torch
        reranker, fake_model, _, _ = reranker_stub
        yes = reranker.profile.yes_id
        no = reranker.profile.no_id
        expected = (
            (fake_model.lm_head.weight[yes] - fake_model.lm_head.weight[no])
            .detach()
            .float()
            .unsqueeze(0)
        )
        assert torch.allclose(
            reranker.score_head.weight.data, expected, atol=1e-5
        )
        assert reranker.score_head.bias is None

    def test_encode_pre_returns_list_of_dicts(self, reranker_stub):
        import torch
        from boofinity.primitives import MMItem
        reranker, _, mock_processor, mock_qwen = reranker_stub

        mock_qwen.process_vision_info.return_value = []
        mock_processor.apply_chat_template.return_value = "<|im_start|>test"

        fake_inputs = MagicMock()
        fake_inputs.items.return_value = [("input_ids", torch.tensor([[1, 2, 3]]))]
        mock_processor.return_value = fake_inputs

        pairs = [
            (MMItem(text="q"), MMItem(text="d")),
        ]

        features_list = reranker.encode_pre(pairs)
        assert len(features_list) == 1
        assert isinstance(features_list[0], dict)
        assert mock_processor.call_args[1]["do_resize"] is False
        assert mock_processor.call_args[1]["max_length"] == 10240

    def test_encode_core_per_pair_loop(self, reranker_stub):
        import torch
        reranker, fake_model, _, _ = reranker_stub

        class CallCountingBackbone:
            def __init__(self, D):
                self.call_count = 0
                self.D = D

            def __call__(self, **inputs):
                self.call_count += 1
                B = inputs.get("input_ids", torch.zeros(1, 1)).shape[0]
                hidden = torch.randn(B, 10, self.D)
                result = MagicMock()
                result.last_hidden_state = hidden
                return result

        D = 128
        backbone = CallCountingBackbone(D)
        fake_model.model = backbone

        B = 3
        features_list = [
            {"input_ids": torch.randint(0, 100, (1, 5)), "attention_mask": torch.ones(1, 5)}
            for _ in range(B)
        ]

        result = reranker.encode_core(features_list)
        assert backbone.call_count == B
        assert result.shape == (B, 1)
        assert result.dtype == torch.float32

    def test_encode_post_sigmoid(self, reranker_stub):
        import torch
        reranker, _, _, _ = reranker_stub

        out = torch.tensor([[0.7], [-0.5], [0.1]])
        scores = reranker.encode_post(out)
        assert len(scores) == 3
        for s in scores:
            assert 0.0 < s < 1.0
        expected = torch.sigmoid(torch.tensor(0.7)).item()
        assert abs(scores[0] - expected) < 1e-5

    def test_encode_post_raw(self, reranker_stub):
        import torch
        reranker, _, _, _ = reranker_stub

        out = torch.tensor([[1.2], [-0.5], [0.1]], dtype=torch.float32)
        scores = reranker.encode_post_raw(out)
        assert len(scores) == 3
        assert abs(scores[0] - 1.2) < 1e-6
        assert abs(scores[1] - (-0.5)) < 1e-6
        assert abs(scores[2] - 0.1) < 1e-6

    def test_tokenize_lengths(self, reranker_stub):
        from boofinity.primitives import MMItem, ReRankMMSingle
        reranker, _, _, _ = reranker_stub

        class FakeEnc:
            def __init__(self, ids):
                self.input_ids = ids

            def __getitem__(self, key):
                return getattr(self, key)

        def fake_tokenizer(text=None, **kwargs):
            return FakeEnc([[1] * (len(t) // 2 + 4) for t in text])

        reranker.tokenizer = fake_tokenizer

        pairs = [
            (MMItem(text="hello"), MMItem(text="world")),
            (MMItem(text="a longer text here"), MMItem(text="short")),
        ]
        lengths = reranker.tokenize_lengths(pairs)
        assert len(lengths) == 2
        assert lengths[1] > lengths[0]
