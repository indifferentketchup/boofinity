# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest


class TestMMEmbeddingItem:
    def test_text_only_accepted(self):
        from boofinity.fastapi_schemas.pymodels import MMEmbeddingItem

        item = MMEmbeddingItem(text="hello")
        assert item.text == "hello"
        assert item.image is None

    def test_image_url_only_accepted(self):
        from boofinity.fastapi_schemas.pymodels import MMEmbeddingItem

        item = MMEmbeddingItem(image="https://example.com/image.png")
        assert item.text is None
        assert str(item.image).startswith("https://")

    def test_image_data_uri_accepted(self):
        from boofinity.fastapi_schemas.pymodels import MMEmbeddingItem

        item = MMEmbeddingItem(
            image="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        assert item.text is None
        assert item.image is not None

    def test_both_accepted(self):
        from boofinity.fastapi_schemas.pymodels import MMEmbeddingItem

        item = MMEmbeddingItem(text="What is this?", image="https://example.com/image.png")
        assert item.text == "What is this?"
        assert item.image is not None

    def test_both_none_rejected(self):
        from boofinity.fastapi_schemas.pymodels import MMEmbeddingItem
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MMEmbeddingItem(text=None, image=None)

    def test_empty_model_rejected(self):
        from boofinity.fastapi_schemas.pymodels import MMEmbeddingItem
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MMEmbeddingItem()


class TestMMEmbeddingInput:
    def test_minimal_input(self):
        from boofinity.fastapi_schemas.pymodels import MMEmbeddingInput, MMEmbeddingItem

        inp = MMEmbeddingInput(input=[MMEmbeddingItem(text="hi")])
        assert len(inp.input) == 1
        assert inp.input[0].text == "hi"
        assert inp.dimensions == 0

    def test_empty_list_rejected(self):
        from boofinity.fastapi_schemas.pymodels import MMEmbeddingInput
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MMEmbeddingInput(input=[])

    def test_dimensions_is_zero_default(self):
        from boofinity.fastapi_schemas.pymodels import MMEmbeddingInput, MMEmbeddingItem

        inp = MMEmbeddingInput(input=[MMEmbeddingItem(text="hi")])
        assert inp.dimensions == 0


class TestMMEmbeddingResult:
    def test_to_embeddings_response_shape(self):
        from boofinity.fastapi_schemas.pymodels import MMEmbeddingResult
        from boofinity.fastapi_schemas.pymodels import EmbeddingEncodingFormat
        from boofinity.args import EngineArgs
        from boofinity.primitives import InferenceEngine, Device, Dtype
        import numpy as np

        engine_args = EngineArgs(
            model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
            engine=InferenceEngine.torch,
            device=Device.cpu,
        )
        object.__setattr__(engine_args, "served_model_name", "Qwen3-VL-Embedding-2B")

        embeddings = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        result = MMEmbeddingResult.to_embeddings_response(
            embeddings=embeddings,
            engine_args=engine_args,
            usage=2,
            encoding_format=EmbeddingEncodingFormat.float,
        )
        assert result["model"] == "Qwen3-VL-Embedding-2B"
        assert len(result["data"]) == 2
        assert result["data"][0]["object"] == "embedding"
        assert len(result["data"][0]["embedding"]) == 3
        assert result["usage"]["prompt_tokens"] == 2
        assert result["usage"]["total_tokens"] == 2


class TestMMReRankItem:
    def test_text_only_accepted(self):
        from boofinity.fastapi_schemas.pymodels import MMReRankItem

        item = MMReRankItem(text="hello")
        assert item.text == "hello"
        assert item.image is None

    def test_image_url_only_accepted(self):
        from boofinity.fastapi_schemas.pymodels import MMReRankItem

        item = MMReRankItem(image="https://example.com/image.png")
        assert item.text is None
        assert str(item.image).startswith("https://")

    def test_image_data_uri_accepted(self):
        from boofinity.fastapi_schemas.pymodels import MMReRankItem

        item = MMReRankItem(
            image="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        assert item.text is None
        assert item.image is not None

    def test_both_accepted(self):
        from boofinity.fastapi_schemas.pymodels import MMReRankItem

        item = MMReRankItem(text="What is this?", image="https://example.com/image.png")
        assert item.text == "What is this?"
        assert item.image is not None

    def test_both_none_rejected(self):
        from boofinity.fastapi_schemas.pymodels import MMReRankItem
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MMReRankItem(text=None, image=None)

    def test_empty_model_rejected(self):
        from boofinity.fastapi_schemas.pymodels import MMReRankItem
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MMReRankItem()


class TestMMReRankInput:
    def test_minimal_input(self):
        from boofinity.fastapi_schemas.pymodels import MMReRankInput, MMReRankItem

        inp = MMReRankInput(
            query=MMReRankItem(text="q"),
            documents=[MMReRankItem(text="d")],
        )
        assert inp.query.text == "q"
        assert len(inp.documents) == 1
        assert inp.raw_scores is False

    def test_empty_documents_rejected(self):
        from boofinity.fastapi_schemas.pymodels import MMReRankInput, MMReRankItem
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MMReRankInput(query=MMReRankItem(text="q"), documents=[])

    def test_top_n_positive(self):
        from boofinity.fastapi_schemas.pymodels import MMReRankInput, MMReRankItem

        inp = MMReRankInput(
            query=MMReRankItem(text="q"),
            documents=[MMReRankItem(text="d")],
            top_n=5,
        )
        assert inp.top_n == 5

    def test_top_n_negative_rejected(self):
        from boofinity.fastapi_schemas.pymodels import MMReRankInput, MMReRankItem
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MMReRankInput(
                query=MMReRankItem(text="q"),
                documents=[MMReRankItem(text="d")],
                top_n=0,
            )


class TestMMReRankResult:
    def test_to_rerank_response_shape(self):
        from boofinity.fastapi_schemas.pymodels import MMReRankResult
        from boofinity.primitives import RerankReturnType

        score1 = RerankReturnType(relevance_score=0.9, index=0, document="doc0")
        score2 = RerankReturnType(relevance_score=0.1, index=1, document="doc1")
        result = MMReRankResult.to_rerank_response(
            scores=[score1, score2],
            model="test-model",
            usage=10,
            return_documents=False,
        )
        assert result["model"] == "test-model"
        assert len(result["results"]) == 2
        assert result["results"][0]["relevance_score"] == 0.9
        assert result["results"][0]["index"] == 0
        assert "document" not in result["results"][0]
        assert result["usage"]["prompt_tokens"] == 10

    def test_to_rerank_response_with_documents(self):
        from boofinity.fastapi_schemas.pymodels import MMReRankResult
        from boofinity.primitives import RerankReturnType

        score = RerankReturnType(relevance_score=0.5, index=0, document="doc0")
        result = MMReRankResult.to_rerank_response(
            scores=[score],
            model="test-model",
            usage=5,
            return_documents=True,
        )
        assert "document" in result["results"][0]
        assert result["results"][0]["document"] == "doc0"
