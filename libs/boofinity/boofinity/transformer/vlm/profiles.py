# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


_DEFAULT_SYSTEM = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
)
_DEFAULT_USER_PREFIX = "<|im_start|>user\n"
_DEFAULT_ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

_DEFAULT_YES_ID = 9693
_DEFAULT_NO_ID = 2152
_DEFAULT_MAX_LENGTH = 10240


@dataclass
class RerankProfile:
    prefix: str
    suffix: str
    yes_id: int
    no_id: int
    max_length: int
    default_instruction: str

    @staticmethod
    def qwen3_vl_reranker() -> "RerankProfile":
        return RerankProfile(
            prefix=(_DEFAULT_SYSTEM + _DEFAULT_USER_PREFIX),
            suffix=_DEFAULT_ASSISTANT_SUFFIX,
            yes_id=_DEFAULT_YES_ID,
            no_id=_DEFAULT_NO_ID,
            max_length=_DEFAULT_MAX_LENGTH,
            default_instruction=_DEFAULT_INSTRUCTION,
        )

    @staticmethod
    def from_repo(
        model_name_or_path: str, revision: Optional[str] = None
    ) -> "RerankProfile":
        builtin = RerankProfile.qwen3_vl_reranker()
        try:
            from transformers import AutoConfig  # type: ignore

            base_config: dict = {
                "pretrained_model_name_or_path": model_name_or_path,
                "trust_remote_code": True,
            }
            if revision is not None:
                base_config["revision"] = revision
            config = AutoConfig.from_pretrained(**base_config)
            if hasattr(config, "yes_token_id") and config.yes_token_id is not None:
                builtin.yes_id = int(config.yes_token_id)
            if hasattr(config, "no_token_id") and config.no_token_id is not None:
                builtin.no_id = int(config.no_token_id)
        except (OSError, ValueError, ImportError):
            pass

        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                revision=revision,
            )
            yes = tok.convert_tokens_to_ids("yes")
            no = tok.convert_tokens_to_ids("no")
            if yes is not None and yes != tok.unk_token_id:
                builtin.yes_id = int(yes)
            if no is not None and no != tok.unk_token_id:
                builtin.no_id = int(no)
        except (OSError, ValueError, ImportError):
            pass

        return builtin
