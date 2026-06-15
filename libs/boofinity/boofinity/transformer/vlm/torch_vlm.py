# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

from boofinity._optional_imports import (
    CHECK_PIL,
    CHECK_QWEN_VL_UTILS,
    CHECK_TORCH,
    CHECK_TRANSFORMERS,
)
from boofinity.args import EngineArgs
from boofinity.transformer.abstract import BaseCrossEncoderMM, BaseTIMM
from boofinity.transformer.quantization.interface import (
    quant_embedding_decorator,
    quant_interface,
)
from boofinity.transformer.vlm.dtype import vlm_resolve_dtype

if CHECK_TORCH.is_available:
    import torch
    import torch.nn as nn  # noqa: N812
    import torch.nn.functional as F  # noqa: N812
if CHECK_TRANSFORMERS.is_available:
    from transformers import AutoConfig  # type: ignore
if CHECK_PIL.is_available:
    from PIL import Image

if TYPE_CHECKING:
    from PIL.Image import Image as ImageClass

    from boofinity.primitives import MMItem

_DEFAULT_INSTRUCTION = "Represent the user's input."
_DEFAULT_MAX_LENGTH = 8192
_WARMUP_IMAGE_SIZE = (256, 256)


def _resolve_local_model_dir(engine_args: "EngineArgs") -> str:
    """Return a local directory for the model.

    Loading a Qwen3-VL repo by hub id triggers a transformers chat-template
    resolution path that fails on repos shipping an `additional_chat_templates`
    directory. Resolving to the local snapshot dir takes the local-glob path
    instead, which loads the real .jinja files correctly.
    """
    import os

    path = engine_args.model_name_or_path
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(path, revision=engine_args.revision)


class VLMReranker(BaseCrossEncoderMM):
    """Qwen3-VL multimodal rerank backend.

    Loads Qwen3VLForConditionalGeneration + Qwen3VLProcessor and produces
    per-pair relevance scores via a one-output score head built from
    lm_head.weight[yes_id] - lm_head.weight[no_id] and sigmoid.

    Forward: model.model(**inputs).last_hidden_state[:, -1] -> score_head -> sigmoid.
    """

    capabilities = {"rerank"}

    def __init__(self, *, engine_args: EngineArgs):
        CHECK_TORCH.mark_required()
        CHECK_TRANSFORMERS.mark_required()
        CHECK_PIL.mark_required()
        CHECK_QWEN_VL_UTILS.mark_required()

        import qwen_vl_utils  # noqa: F811

        self._qwen_vl_utils = qwen_vl_utils

        from boofinity.transformer.vlm.profiles import RerankProfile

        self.engine_args = engine_args
        ls = engine_args._loading_strategy
        assert ls is not None

        device_capability = (0, 0)
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()

        resolved_dtype = vlm_resolve_dtype(engine_args, device_capability)

        base_config = dict(
            pretrained_model_name_or_path=_resolve_local_model_dir(engine_args),
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        )

        config = AutoConfig.from_pretrained(**base_config)
        model_type = (getattr(config, "model_type", "") or "").lower()
        if model_type != "qwen3_vl":
            raise ValueError(
                f"VLMReranker requires model_type qwen3_vl, got {model_type}"
            )

        from transformers import (  # type: ignore
            Qwen3VLForConditionalGeneration,
            Qwen3VLProcessor,
        )

        self.processor = Qwen3VLProcessor.from_pretrained(**base_config)
        self.processor.tokenizer.padding_side = "left"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            **base_config,
            dtype=resolved_dtype,
            attn_implementation="sdpa",
        )
        self.model = self.model.to(ls.device_placement)  # type: ignore
        self.model.eval()

        self.profile = RerankProfile.from_repo(
            engine_args.model_name_or_path, revision=engine_args.revision
        )

        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.padding_side != "left":
            raise RuntimeError(
                "VLMReranker requires padding_side='left'"
            )

        V, D = self.model.lm_head.weight.shape  # type: ignore
        yes = self.profile.yes_id
        no = self.profile.no_id
        self.score_head = nn.Linear(D, 1, bias=False)
        with torch.no_grad():
            diff = (
                (self.model.lm_head.weight[yes] - self.model.lm_head.weight[no])  # type: ignore
                .detach()
            )
            self.score_head.weight.copy_(diff.to(torch.float32).unsqueeze(0))
        self.score_head = self.score_head.to(self.model.device).to(
            self.model.dtype  # type: ignore
        )

        self._prefix_ids = self.tokenizer.encode(
            self.profile.prefix, add_special_tokens=False
        )
        self._suffix_ids = self.tokenizer.encode(
            self.profile.suffix, add_special_tokens=False
        )

        self._warmup_image_size = _WARMUP_IMAGE_SIZE

        if hasattr(self.processor.image_processor, "patch_size"):
            self._image_patch_size = self.processor.image_processor.patch_size
        elif hasattr(self.processor.image_processor, "merge_size"):
            self._image_patch_size = self.processor.image_processor.merge_size
        else:
            self._image_patch_size = 16

        if ls.quantization_dtype is not None:
            self.model = quant_interface(
                self.model, engine_args.dtype, device=engine_args.device
            )

        if engine_args.compile:
            self.model = torch.compile(self.model, dynamic=True)

        self.mock_image = Image.new("RGB", _WARMUP_IMAGE_SIZE, color="black")

    @property
    def embedding_dtype(self):
        from boofinity.primitives import EmbeddingDtype

        return self.engine_args.embedding_dtype if hasattr(self, "engine_args") else EmbeddingDtype.float32

    def encode_pre(
        self, pairs: list[tuple["MMItem", "MMItem"]]
    ) -> list[dict]:
        features_list = []
        for q_item, d_item in pairs:
            q_text = q_item.text if q_item.text else ""
            d_text = d_item.text if d_item.text else ""

            messages: list[dict] = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.profile.default_instruction}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"<Instruct>: {self.profile.default_instruction}\n<Query>: {q_text}\n<Document>: {d_text}"}
                    ],
                },
            ]

            if q_item.image is not None:
                messages[1]["content"].append({"type": "image", "image": q_item.image})
            if d_item.image is not None:
                messages[1]["content"].append({"type": "image", "image": d_item.image})

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            vision_info = self._qwen_vl_utils.process_vision_info(
                messages, image_patch_size=self._image_patch_size
            )
            # process_vision_info returns a (image_inputs, video_inputs) tuple;
            # the processor wants just the list of images (or None).
            if isinstance(vision_info, (tuple, list)):
                image_inputs = vision_info[0] if vision_info else None
            else:
                image_inputs = vision_info
            images = image_inputs if image_inputs else None

            inputs = self.processor(
                text=text,
                images=images,
                padding=True,
                truncation=True,
                max_length=self.profile.max_length,
                do_resize=False,
                return_tensors="pt",
            )
            device = self.model.device
            features_list.append({k: v.to(device) for k, v in inputs.items()})

        return features_list

    def encode_core(self, features_list: list[dict]) -> torch.Tensor:
        scores = []
        for f in features_list:
            with torch.no_grad():
                out = self.model.model(**f, return_dict=True)
                last = out.last_hidden_state[:, -1, :]
                score = self.score_head(last.to(self.score_head.weight.dtype))
                scores.append(score.detach().cpu().float())
        return torch.cat(scores, dim=0)

    @quant_embedding_decorator()
    def encode_post(self, out: torch.Tensor) -> list[float]:
        probs = torch.sigmoid(out.squeeze(-1))
        return probs.numpy().astype("float32").tolist()

    def encode_post_raw(self, out: torch.Tensor) -> list[float]:
        logits = out.squeeze(-1).cpu().float()
        return logits.tolist()

    def tokenize_lengths(self, pairs) -> list[int]:
        texts = []
        for pair in pairs:
            if isinstance(pair, tuple):
                q, d = pair
                q_text = q.text if q.text else ""
                d_text = d.text if d.text else ""
            else:
                q_text = pair.query.text or ""
                d_text = pair.document.text or ""
            texts.append(q_text + d_text)

        encodings = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.profile.max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [len(t) for t in encodings["input_ids"]]


class VLMEmbedder(BaseTIMM):
    """Qwen3-VL multimodal image-embed backend.

    Loads Qwen3VLForConditionalGeneration + Qwen3VLProcessor and produces
    instruction-aware (text, image) embeddings via last-non-pad token pooling
    and L2 normalisation.
    """

    capabilities = {"embed", "image_embed"}

    def __init__(self, *, engine_args: EngineArgs):
        CHECK_TORCH.mark_required()
        CHECK_TRANSFORMERS.mark_required()
        CHECK_PIL.mark_required()
        CHECK_QWEN_VL_UTILS.mark_required()

        import qwen_vl_utils  # noqa: F811

        self._qwen_vl_utils = qwen_vl_utils

        self.engine_args = engine_args
        ls = engine_args._loading_strategy
        assert ls is not None

        device_capability = (0, 0)
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()

        resolved_dtype = vlm_resolve_dtype(engine_args, device_capability)

        base_config = dict(
            pretrained_model_name_or_path=_resolve_local_model_dir(engine_args),
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        )

        config = AutoConfig.from_pretrained(**base_config)
        model_type = (getattr(config, "model_type", "") or "").lower()
        if model_type != "qwen3_vl":
            raise ValueError(
                f"VLMEmbedder requires model_type qwen3_vl, got {model_type}"
            )

        from transformers import (  # type: ignore
            Qwen3VLForConditionalGeneration,
            Qwen3VLProcessor,
        )

        self.processor = Qwen3VLProcessor.from_pretrained(**base_config)
        self.processor.tokenizer.padding_side = "right"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            **base_config,
            dtype=resolved_dtype,
            attn_implementation="sdpa",
        )
        self.model = self.model.to(ls.device_placement)  # type: ignore
        self.model.eval()

        self._default_instruction = _DEFAULT_INSTRUCTION
        self._warmup_image_size = _WARMUP_IMAGE_SIZE

        if hasattr(self.processor.image_processor, "patch_size"):
            self._image_patch_size = self.processor.image_processor.patch_size
        elif hasattr(self.processor.image_processor, "merge_size"):
            self._image_patch_size = self.processor.image_processor.merge_size
        else:
            self._image_patch_size = 16

        max_length = _DEFAULT_MAX_LENGTH
        if hasattr(self.model.config, "max_length"):
            max_length = self.model.config.max_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            max_length = self.model.config.max_position_embeddings
        elif hasattr(self.model.config, "text_config") and hasattr(
            self.model.config.text_config, "max_length"
        ):
            max_length = self.model.config.text_config.max_length
        self.max_length = max_length

        if ls.quantization_dtype is not None:
            self.model = quant_interface(
                self.model, engine_args.dtype, device=engine_args.device
            )

        if engine_args.compile:
            # not recommended on Pascal; honoured if explicitly requested
            self.model = torch.compile(self.model, dynamic=True)

        self.mock_image = Image.new("RGB", _WARMUP_IMAGE_SIZE, color="black")

    def encode_pre(
        self, items: list[Union[str, ImageClass, tuple[str, ImageClass]]]
    ) -> dict[str, Any]:
        messages_list = []
        for item in items:
            if isinstance(item, tuple):
                text, image = item
            elif isinstance(item, str):
                text, image = item, None
            else:
                text, image = None, item

            user_content: list[dict[str, Any]] = []
            if text is not None:
                user_content.append({"type": "text", "text": text})
            if image is not None:
                user_content.append({"type": "image", "image": image})

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self._default_instruction}
                    ],
                },
                {"role": "user", "content": user_content},
            ]
            messages_list.append(messages)

        texts = []
        images_list = []
        for messages in messages_list:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            vision_info = self._qwen_vl_utils.process_vision_info(
                messages, image_patch_size=self._image_patch_size
            )
            images_list.append(vision_info[0] if vision_info else None)

        has_images = any(img is not None for img in images_list)
        if has_images:
            flat_images = [img for img in images_list if img is not None]
        else:
            flat_images = None

        inputs = self.processor(
            text=texts,
            images=flat_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            do_resize=False,
            return_tensors="pt",
        )
        device = self.model.device
        return {k: v.to(device) for k, v in inputs.items()}

    def encode_core(self, features: dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(**features, return_dict=True, output_hidden_states=True)
            # Qwen3VLForConditionalGeneration is a causal LM: take the final
            # decoder layer's hidden states for last-token pooling.
            hidden = out.hidden_states[-1]
            attn = features["attention_mask"]
            idx = attn.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
            pooled = F.normalize(pooled, p=2, dim=1)
            return pooled.detach().cpu().float()

    @quant_embedding_decorator()
    def encode_post(
        self, out_features: torch.Tensor
    ) -> list[Any]:
        embeddings = out_features.float()

        matryoshka_dim = None
        if hasattr(self.engine_args, "_matryoshka_dim"):
            matryoshka_dim = self.engine_args._matryoshka_dim
        if matryoshka_dim is not None:
            matryoshka_dim = int(matryoshka_dim)

        if matryoshka_dim is not None and matryoshka_dim > 0:
            embeddings = embeddings[:, :matryoshka_dim]
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return [row.numpy() for row in embeddings]

    def tokenize_lengths(self, items: list[str]) -> list[int]:
        texts = []
        images = []
        for item in items:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self._default_instruction}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item},
                        {"type": "image", "image": self.mock_image},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            vision_info = self._qwen_vl_utils.process_vision_info(
                messages, image_patch_size=self._image_patch_size
            )
            images.append(vision_info[0] if vision_info else None)

        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            do_resize=False,
            return_tensors="pt",
        )
        return [len(t) for t in inputs["input_ids"]]
