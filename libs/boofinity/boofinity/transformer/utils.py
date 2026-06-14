# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from enum import Enum
from importlib import import_module
from typing import Callable

from boofinity.primitives import InferenceEngine

__all__ = [
    "length_tokenizer",
    "get_lengths_with_tokenize",
]


class _LazyBackend:
    __slots__ = ("_import_path", "_class_name", "_cls")

    def __init__(self, import_path: str, class_name: str):
        self._import_path = import_path
        self._class_name = class_name
        self._cls = None

    def __call__(self, *args, **kwargs):
        if self._cls is None:
            mod = import_module(self._import_path)
            self._cls = getattr(mod, self._class_name)
        return self._cls(*args, **kwargs)


class EmbedderEngine(Enum):
    torch = _LazyBackend(
        "boofinity.transformer.embedder.sentence_transformer",
        "SentenceTransformerPatched",
    )
    ctranslate2 = _LazyBackend(
        "boofinity.transformer.embedder.ct2", "CT2SentenceTransformer"
    )
    debugengine = _LazyBackend(
        "boofinity.transformer.embedder.dummytransformer", "DummyTransformer"
    )
    optimum = _LazyBackend(
        "boofinity.transformer.embedder.optimum", "OptimumEmbedder"
    )
    neuron = _LazyBackend(
        "boofinity.transformer.embedder.neuron", "NeuronOptimumEmbedder"
    )

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.torch:
            return EmbedderEngine.torch
        elif engine == InferenceEngine.ctranslate2:
            return EmbedderEngine.ctranslate2
        elif engine == InferenceEngine.debugengine:
            return EmbedderEngine.debugengine
        elif engine == InferenceEngine.optimum:
            return EmbedderEngine.optimum
        elif engine == InferenceEngine.neuron:
            return EmbedderEngine.neuron
        else:
            raise NotImplementedError(f"EmbedderEngine for {engine} not implemented")


class RerankEngine(Enum):
    torch = _LazyBackend(
        "boofinity.transformer.crossencoder.torch", "CrossEncoderPatched"
    )
    optimum = _LazyBackend(
        "boofinity.transformer.crossencoder.optimum", "OptimumCrossEncoder"
    )

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.torch:
            return RerankEngine.torch
        elif engine == InferenceEngine.optimum:
            return RerankEngine.optimum
        else:
            raise NotImplementedError(f"RerankEngine for {engine} not implemented")


class ImageEmbedEngine(Enum):
    torch = _LazyBackend(
        "boofinity.transformer.vision.torch_vision", "TIMM"
    )

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.torch:
            return ImageEmbedEngine.torch
        else:
            raise NotImplementedError(f"ImageEmbedEngine for {engine} not implemented")


class AudioEmbedEngine(Enum):
    torch = _LazyBackend(
        "boofinity.transformer.audio.torch", "TorchAudioModel"
    )

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.torch:
            return AudioEmbedEngine.torch
        else:
            raise NotImplementedError(f"AudioEmbedEngine for {engine} not implemented")


class PredictEngine(Enum):
    torch = _LazyBackend(
        "boofinity.transformer.classifier.torch", "SentenceClassifier"
    )
    optimum = _LazyBackend(
        "boofinity.transformer.classifier.optimum", "OptimumClassifier"
    )

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.torch:
            return PredictEngine.torch
        elif engine == InferenceEngine.optimum:
            return PredictEngine.optimum
        else:
            raise NotImplementedError(f"PredictEngine for {engine} not implemented")


def length_tokenizer(
    _sentences: list[str],
) -> list[int]:
    return [len(i) for i in _sentences]


def get_lengths_with_tokenize(
    _sentences: list[str], tokenize: Callable = length_tokenizer
) -> tuple[list[int], int]:
    _lengths = tokenize(_sentences)
    return _lengths, sum(_lengths)
