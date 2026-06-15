# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from boofinity._optional_imports import CHECK_TORCH, CHECK_TRANSFORMERS
from boofinity.args import EngineArgs
from boofinity.transformer.abstract import BaseCrossEncoder

if CHECK_TORCH.is_available and CHECK_TRANSFORMERS.is_available:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]
else:

    class AutoTokenizer:  # type: ignore[no-redef]
        pass

    class AutoModelForCausalLM:  # type: ignore[no-redef]
        pass


if TYPE_CHECKING:
    from torch import Tensor

_QWEN3_RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_QWEN3_RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN3_RERANKER_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

_QWEN3_YES_ID = 9693
_QWEN3_NO_ID = 2152
_QWEN3_MAX_LENGTH = 8192


@dataclass
class LMProfile:
    prefix: str
    suffix: str
    yes_id: int
    no_id: int
    max_length: int
    default_instruction: str
    _fallback: bool = field(default=False)

    @staticmethod
    def qwen3_reranker() -> "LMProfile":
        return LMProfile(
            prefix=_QWEN3_RERANKER_PREFIX,
            suffix=_QWEN3_RERANKER_SUFFIX,
            yes_id=_QWEN3_YES_ID,
            no_id=_QWEN3_NO_ID,
            max_length=_QWEN3_MAX_LENGTH,
            default_instruction=_QWEN3_RERANKER_DEFAULT_INSTRUCTION,
        )

    @staticmethod
    def from_repo(
        model_name_or_path: str, revision: Optional[str] = None
    ) -> "LMProfile":
        try:
            profile_data = _fetch_logit_score_config(model_name_or_path, revision)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, revision=revision, trust_remote_code=True
            )
            true_id = _resolve_token_id(tokenizer, profile_data.get("true"), "yes")
            false_id = _resolve_token_id(tokenizer, profile_data.get("false"), "no")
            max_length = int(profile_data.get("max_length", _QWEN3_MAX_LENGTH))
            instruction = profile_data.get(
                "default_instruction", _QWEN3_RERANKER_DEFAULT_INSTRUCTION
            )
            if isinstance(instruction, str) and instruction.strip():
                default_instruction = instruction
            else:
                default_instruction = _QWEN3_RERANKER_DEFAULT_INSTRUCTION
            return LMProfile(
                prefix=_QWEN3_RERANKER_PREFIX,
                suffix=_QWEN3_RERANKER_SUFFIX,
                yes_id=true_id,
                no_id=false_id,
                max_length=max_length,
                default_instruction=default_instruction,
            )
        except Exception:
            return LMProfile.qwen3_reranker()


def _fetch_logit_score_config(
    model_name_or_path: str, revision: Optional[str] = None
) -> dict:
    if Path(model_name_or_path).is_dir():
        config_path = Path(model_name_or_path) / "1_LogitScore" / "config.json"
        if config_path.is_file():
            with open(config_path, "r") as f:
                return json.load(f)
        raise FileNotFoundError(str(config_path))
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(
        model_name_or_path,
        revision=revision,
        filename="1_LogitScore/config.json",
        repo_type="model",
    )
    with open(config_path, "r") as f:
        return json.load(f)


def _resolve_token_id(tokenizer, profile_value, fallback_token: str) -> int:
    if profile_value is not None:
        return int(profile_value)
    token_id = tokenizer.convert_tokens_to_ids(fallback_token)
    if token_id is not None and token_id != tokenizer.unk_token_id:
        return int(token_id)
    return getattr(
        LMProfile.qwen3_reranker(),
        "yes_id" if fallback_token == "yes" else "no_id",
    )


class CausalLMReranker(BaseCrossEncoder):
    capabilities = {"rerank"}

    def __init__(self, *, engine_args: EngineArgs):
        CHECK_TORCH.mark_required()
        CHECK_TRANSFORMERS.mark_required()

        ls = engine_args._loading_strategy
        assert ls is not None

        self.tokenizer = AutoTokenizer.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.padding_side != "left":
            raise RuntimeError(
                "CausalLMReranker requires padding_side='left'"
            )

        model_kwargs = {}
        if ls.loading_dtype is not None:
            model_kwargs["dtype"] = ls.loading_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            **model_kwargs,
        )
        self.model.to(ls.device_placement)
        self.model.eval()

        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)

        self.profile = LMProfile.from_repo(
            engine_args.model_name_or_path, revision=engine_args.revision
        )

        self._prefix_ids = self.tokenizer.encode(
            self.profile.prefix, add_special_tokens=False
        )
        self._suffix_ids = self.tokenizer.encode(
            self.profile.suffix, add_special_tokens=False
        )

    def encode_pre(self, pairs: list[tuple[str, str]]) -> dict[str, "Tensor"]:
        prefix_ids = self._prefix_ids
        suffix_ids = self._suffix_ids
        max_body_length = self.profile.max_length - len(prefix_ids) - len(suffix_ids)

        all_input_ids = []
        for q, d in pairs:
            body = (
                f"<Instruct>: {self.profile.default_instruction}\n"
                f"<Query>: {q}\n"
                f"<Document>: {d}"
            )
            body_ids = self.tokenizer.encode(
                body,
                add_special_tokens=False,
                truncation=True,
                max_length=max_body_length,
            )
            item_ids = prefix_ids + body_ids + suffix_ids
            all_input_ids.append({"input_ids": item_ids})

        padded = self.tokenizer.pad(all_input_ids, padding=True, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in padded.items()}

    def encode_core(self, features: dict[str, "Tensor"]) -> "Tensor":
        with torch.no_grad():
            out = self.model(**features, return_dict=True)
            last = out.logits[:, -1, :]
            pair = torch.stack(
                [last[:, self.profile.no_id], last[:, self.profile.yes_id]], dim=1
            )
            return pair.detach().to("cpu", torch.float32)

    def encode_post(self, out_features) -> list[float]:
        true_logit = out_features[:, 1]
        false_logit = out_features[:, 0]
        logits = true_logit - false_logit
        return logits.to(torch.float32).numpy().tolist()

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        tks = self._infinity_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=False,
            truncation="longest_first",
        ).encodings
        return [len(t.tokens) for t in tks]
