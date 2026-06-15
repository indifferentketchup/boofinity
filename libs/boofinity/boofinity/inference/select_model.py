# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import json
from pathlib import Path
from typing import Optional, Union

from boofinity.args import (
    EngineArgs,
)
from boofinity.log_handler import logger
from boofinity.transformer.abstract import BaseCrossEncoder, BaseEmbedder
from boofinity.transformer.utils import (
    AudioEmbedEngine,
    EmbedderEngine,
    ImageEmbedEngine,
    InferenceEngine,
    PredictEngine,
    RerankEngine,
)


def get_engine_type_from_config(
    engine_args: EngineArgs,
) -> Union[EmbedderEngine, RerankEngine, PredictEngine, ImageEmbedEngine, AudioEmbedEngine]:
    """resolved the class of inference engine path from config.json of the repo."""
    if engine_args.engine in [InferenceEngine.debugengine]:
        return EmbedderEngine.from_inference_engine(engine_args.engine)

    config = _try_fetch_json(engine_args, "config.json")
    if config is None:
        return EmbedderEngine.from_inference_engine(engine_args.engine)

    st_meta = _try_fetch_json(
        engine_args, "config_sentence_transformers.json"
    )

    # A qwen3_vl repo can also carry CrossEncoder sentence-transformers metadata;
    # the multimodal backend must win over the text CrossEncoder routing.
    if config.get("model_type") == "qwen3_vl":
        return _resolve_qwen3vl_embed_or_rerank(engine_args)

    if st_meta and st_meta.get("model_type") == "CrossEncoder":
        return _resolve_rerank_engine(engine_args, st_meta)

    if any("SequenceClassification" in arch for arch in config.get("architectures", [])):
        id2label = config.get("id2label", {"0": "dummy"})
        if len(id2label) < 2:
            return RerankEngine.from_inference_engine(engine_args.engine)
        else:
            return PredictEngine.from_inference_engine(engine_args.engine)
    if config.get("vision_config"):
        return ImageEmbedEngine.from_inference_engine(engine_args.engine)
    if config.get("audio_config") and "clap" in config.get("model_type", "").lower():
        return AudioEmbedEngine.from_inference_engine(engine_args.engine)

    else:
        return EmbedderEngine.from_inference_engine(engine_args.engine)


def _mode() -> str:
    from boofinity.env import MANAGER

    return MANAGER.rerank_mode


def _resolve_rerank_engine(engine_args: EngineArgs, st_meta: dict) -> RerankEngine:
    if _mode() == "causal_lm":
        return RerankEngine.causal_lm
    if _mode() == "classifier":
        return RerankEngine.torch
    if _repo_uses_lm_rerank(engine_args, st_meta):
        return RerankEngine.causal_lm
    return RerankEngine.torch


def _repo_uses_lm_rerank(engine_args: EngineArgs, st_meta: dict) -> bool:
    if st_meta.get("model_type") == "CrossEncoder":
        modules = _try_fetch_json(engine_args, "modules.json") or {}
        if _modules_has_logit_score(modules):
            return True
        if _try_fetch_json(engine_args, "1_LogitScore/config.json"):
            return True
    sbc = _try_fetch_json(engine_args, "sentence_bert_config.json") or {}
    if sbc.get("transformer_task") == "text-generation":
        return True
    return False


def _modules_has_logit_score(modules) -> bool:
    # sentence-transformers modules.json is a JSON list of module dicts; some
    # variants nest it under a "modules" key.
    if isinstance(modules, dict):
        entries = modules.get("modules", []) or []
    elif isinstance(modules, list):
        entries = modules
    else:
        entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path", "") or ""
        mtype = entry.get("type", "") or ""
        if "LogitScore" in path or mtype.endswith("LogitScore"):
            return True
    return False


def _resolve_qwen3vl_embed_or_rerank(
    engine_args: EngineArgs,
) -> Union[ImageEmbedEngine, RerankEngine]:
    from boofinity.env import MANAGER

    vlm_mode = MANAGER.vlm_mode
    if vlm_mode == "embed":
        return ImageEmbedEngine.qwen3vl
    if vlm_mode == "rerank":
        return RerankEngine.qwen3vl

    last_segment = engine_args.model_name_or_path.rstrip("/").rsplit("/", 1)[-1].lower()
    if "rerank" in last_segment:
        return RerankEngine.qwen3vl
    return ImageEmbedEngine.qwen3vl


def _try_fetch_json(engine_args: EngineArgs, filename: str) -> Optional[dict]:
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        if Path(engine_args.model_name_or_path).is_dir():
            config_path = Path(engine_args.model_name_or_path) / filename
            if not config_path.is_file():
                return None
            with open(config_path, "r") as f:
                return json.load(f)
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            filename=filename,
        )
        with open(config_path, "r") as f:
            return json.load(f)
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        EntryNotFoundError,
        RepositoryNotFoundError,
    ):
        # "this file/repo does not exist" means "not this model type", so fall
        # through. Network, auth, and rate-limit errors propagate instead of
        # being silently treated as a missing config.
        return None


def select_model(
    engine_args: EngineArgs,
) -> tuple[list[Union[BaseCrossEncoder, BaseEmbedder]], float, float]:
    """based on engine args, fully instantiates the Engine."""
    logger.info(
        f"model=`{engine_args.model_name_or_path}` selected, "
        f"using engine=`{engine_args.engine.value}`"
        f" and device=`{engine_args.device.resolve()}`"
    )
    # engine_args.update_loading_strategy()

    unloaded_engine = get_engine_type_from_config(engine_args)

    engine_replicas = []
    min_inference_t = 4e-3
    max_inference_t = 4e-3

    # TODO: Can be parallelized
    for device_map in engine_args._loading_strategy.device_mapping:  # type: ignore
        engine_args_copy = engine_args.copy()
        engine_args_copy._loading_strategy.device_placement = device_map
        loaded_engine = unloaded_engine.value(engine_args=engine_args_copy)

        if engine_args.model_warmup:
            # size one, warm up warm start timings.
            # loaded_engine.warmup(batch_size=engine_args.batch_size, n_tokens=1)
            # size one token
            min_inference_t = min(
                min(loaded_engine.warmup(batch_size=1, n_tokens=1)[1] for _ in range(10)),
                min_inference_t,
            )
            loaded_engine.warmup(batch_size=engine_args.batch_size, n_tokens=1)
            emb_per_sec_short, max_inference_temp, log_msg = loaded_engine.warmup(
                batch_size=engine_args.batch_size, n_tokens=1
            )
            max_inference_t = max(max_inference_temp, max_inference_t)

            logger.info(log_msg)
            # now warm up with max_token, max batch size
            loaded_engine.warmup(batch_size=engine_args.batch_size, n_tokens=512)
            emb_per_sec, _, log_msg = loaded_engine.warmup(
                batch_size=engine_args.batch_size, n_tokens=512
            )
            logger.info(log_msg)
            logger.info(
                f"model warmed up, between {emb_per_sec:.2f}-{emb_per_sec_short:.2f}"
                f" embeddings/sec at batch_size={engine_args.batch_size}"
            )
        engine_replicas.append(loaded_engine)
    assert len(engine_replicas) > 0, "No engine replicas were loaded"

    return engine_replicas, min_inference_t, max_inference_t
