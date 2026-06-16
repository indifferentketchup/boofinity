import asyncio
import base64
import json
import os
import pathlib
import random
import sys
import time
from unittest import TestCase

import numpy as np
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from boofinity import create_server
from boofinity.args import EngineArgs
from boofinity.primitives import InferenceEngine

PREFIX = ""
MODEL_NAME = "dummy-number-1"
MODEL_NAME_2 = "dummy-number-2"
BATCH_SIZE = 16

PATH_OPENAPI = pathlib.Path(__file__).parent.parent.parent.parent.parent.joinpath(
    "docs", "assets", "openapi.json"
)

app = create_server(
    url_prefix=PREFIX,
    engine_args_list=[
        EngineArgs(
            model_name_or_path=MODEL_NAME,
            batch_size=BATCH_SIZE,
            engine=InferenceEngine.debugengine,
        ),
        EngineArgs(
            model_name_or_path=MODEL_NAME_2,
            batch_size=BATCH_SIZE,
            engine=InferenceEngine.debugengine,
        ),
    ],
)


@pytest.fixture()
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client, LifespanManager(app):
        yield client


@pytest.mark.anyio
async def test_model_route(client):
    response = await client.get(f"{PREFIX}/models")
    assert response.status_code == 200
    rdata = response.json()
    assert "data" in rdata
    assert rdata["data"][0].get("id", "") == MODEL_NAME
    assert rdata["data"][1].get("id", "") == MODEL_NAME_2
    assert isinstance(rdata["data"][0].get("stats"), dict)

    # ready test
    respnse_health = await client.get("/health")
    assert respnse_health.status_code == 200
    assert "unix" in respnse_health.json()


@pytest.mark.anyio
async def test_embedding_max_length(client):
    # TOO long
    for model_name in [MODEL_NAME, MODEL_NAME_2]:
        input = "%_" * 4097 * 15
        response = await client.post(
            f"{PREFIX}/embeddings", json=dict(input=input, model=model_name)
        )
        assert response.status_code == 422, f"{response.status_code}, {response.text}"
        # works
        input = "%_" * 4096 * 15
        response = await client.post(
            f"{PREFIX}/embeddings", json=dict(input=input, model=model_name)
        )
        assert response.status_code == 200, f"{response.status_code}, {response.text}"
        assert response.json()["model"] == model_name


@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.anyio
async def test_encoding_base_64(client, model_name):
    input = "Hello World"
    response = await client.post(
        f"{PREFIX}/embeddings",
        json=dict(input=input, model=model_name, encoding_format="float"),
    )
    assert response.status_code == 200
    response_base64 = await client.post(
        f"{PREFIX}/embeddings",
        json=dict(input=input, model=model_name, encoding_format="base64"),
    )
    assert response_base64.status_code == 200
    embedding = response.json()["data"][0]["embedding"]
    embedding_base64 = response_base64.json()["data"][0]["embedding"]
    embedding_base64 = np.frombuffer(base64.b64decode(embedding_base64), dtype=np.float32).tolist()
    assert embedding_base64 == embedding


@pytest.mark.anyio
async def test_embedding(client):
    possible_inputs = [
        ["This is a test sentence."],
        ["This is a test sentence.", "This is another test sentence."],
    ]
    for inp in possible_inputs:
        response = await client.post(f"{PREFIX}/embeddings", json=dict(input=inp, model=MODEL_NAME))
        assert response.status_code == 200, f"{response.status_code}, {response.text}"
        rdata = response.json()
        assert "data" in rdata and isinstance(rdata["data"], list)
        assert all("embedding" in d for d in rdata["data"])
        assert len(rdata["data"]) == len(inp)
        for embedding, sentence in zip(rdata["data"], inp):
            assert len(sentence) == embedding["embedding"][0]


@pytest.mark.anyio
async def test_batch_embedding(client, get_sts_bechmark_dataset):
    sentences = []
    for d in get_sts_bechmark_dataset:
        for item in d:
            sentences.append(item.texts[0])
    random.shuffle(sentences)
    sentences = sentences

    async def _post_batch(inputs):
        return await client.post(f"{PREFIX}/embeddings", json=dict(input=inputs, model=MODEL_NAME))

    _request_size = BATCH_SIZE // 2
    tasks = [
        _post_batch(inputs=sentences[sl : sl + _request_size])
        for sl in range(0, len(sentences), _request_size)
    ]
    start = time.perf_counter()
    _responses = await asyncio.gather(*tasks)
    end = time.perf_counter()
    time_api = end - start

    responses = []
    for response in _responses:
        responses.extend(response.json()["data"])
    for i in range(len(responses)):
        responses[i] = responses[i]["embedding"]

    print(time_api)


@pytest.mark.skipif(sys.platform != "linux", reason="Only check on linux")
@pytest.mark.skipif(not PATH_OPENAPI.exists(), reason="openapi.json does not exist")
@pytest.mark.anyio
async def test_openapi_same_as_docs_file(client):
    assert (
        PATH_OPENAPI.exists()
    ), f"openapi.json file does not exist, it should be in {PATH_OPENAPI.resolve()}"

    openapi_req = await client.get("/openapi.json")
    assert openapi_req.status_code == 200
    openapi_json = openapi_req.json()
    openapi_json_expected = json.loads(PATH_OPENAPI.read_text())
    openapi_json["info"].pop("version")
    openapi_json_expected["info"].pop("version")
    tc = TestCase()
    tc.maxDiff = 100000
    assert openapi_json["openapi"] == openapi_json_expected["openapi"]
    tc.assertDictEqual(openapi_json["info"], openapi_json_expected["info"])
    tc.assertDictEqual(openapi_json["paths"], openapi_json_expected["paths"])
    # tc.assertDictEqual(openapi_json["components"], openapi_json_expected["components"])


@pytest.mark.anyio
async def test_matryoshka_embedding(client):
    matryoshka_dim = 10

    possible_inputs = [
        ["This is a test sentence."],
        ["This is a test sentence.", "This is another test sentence."],
    ]
    for inp in possible_inputs:
        response = await client.post(
            f"{PREFIX}/embeddings",
            json=dict(input=inp, model=MODEL_NAME, dimensions=matryoshka_dim),
        )
        assert response.status_code == 200, f"{response.status_code}, {response.text}"
        rdata = response.json()
        assert "data" in rdata and isinstance(rdata["data"], list)
        assert all("embedding" in d for d in rdata["data"])
        assert len(rdata["data"]) == len(inp)
        for embedding, sentence in zip(rdata["data"], inp):
            assert len(sentence) == embedding["embedding"][0]
            assert len(embedding["embedding"]) == matryoshka_dim


# --- needs-network end-to-end against the real Qwen3-VL-Embedding-2B ---
#
# These boot a second FastAPI app with the real multimodal embed backend.
# They are skipped on the CPU-only dev box (the 2B model is too slow to be a
# useful CPU test) and when HF Hub access is disabled, so importing/collecting
# this module never triggers a model download.

MM_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"

# 1x1 PNG, base64 data URI (avoids a second network fetch for the image input).
_TINY_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _build_mm_app():
    return create_server(
        url_prefix=PREFIX,
        engine_args_list=[
            EngineArgs(
                model_name_or_path=MM_MODEL_NAME,
                engine=InferenceEngine.torch,
                model_warmup=True,
                dtype="auto",
                device="cuda",
            )
        ],
    )


@pytest.mark.needs_network
@pytest.mark.anyio
@pytest.mark.skipif(not _cuda_available(), reason="CUDA host only; 2B VLM too slow on CPU")
async def test_mm_embeddings_real_model_text_and_image():
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        pytest.skip("HF_HUB_OFFLINE=1")
    mm_app = _build_mm_app()
    async with AsyncClient(
        transport=ASGITransport(app=mm_app), base_url="http://test"
    ) as cli, LifespanManager(mm_app):
        response = await cli.post(
            f"{PREFIX}/mm_embeddings",
            json=dict(
                model=MM_MODEL_NAME,
                input=[
                    {"text": "a photo of a cat"},
                    {"text": "what is in this image?", "image": _TINY_PNG_DATA_URI},
                ],
            ),
        )
    assert response.status_code == 200, f"{response.status_code}, {response.text}"
    rdata = response.json()
    assert "data" in rdata and len(rdata["data"]) == 2
    for obj in rdata["data"]:
        emb = obj["embedding"]
        assert len(emb) == 2048
        norm = sum(v * v for v in emb) ** 0.5
        assert abs(norm - 1.0) < 1e-5, f"L2 norm {norm} != 1.0"


@pytest.mark.needs_network
@pytest.mark.anyio
@pytest.mark.skipif(not _cuda_available(), reason="CUDA host only; 2B VLM too slow on CPU")
async def test_mm_embeddings_real_model_smoke_200():
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        pytest.skip("HF_HUB_OFFLINE=1")
    mm_app = _build_mm_app()
    async with AsyncClient(
        transport=ASGITransport(app=mm_app), base_url="http://test"
    ) as cli, LifespanManager(mm_app):
        response = await cli.post(
            f"{PREFIX}/mm_embeddings",
            json=dict(model=MM_MODEL_NAME, input=[{"text": "hello"}]),
        )
    assert response.status_code == 200, f"{response.status_code}, {response.text}"
