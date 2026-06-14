# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

"""
Data-driven ORT provider policy.

Maps a HardwareCapability to an ordered ONNX Runtime provider list and dtype.
No TensorRT. Import-light: torch is never imported at module level (JD-007).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from boofinity.hardware.capability import HardwareCapability

logger = logging.getLogger(__name__)

# Pinned ORT version (matches .venv-batch2).
PINNED_ORT_VERSION: str = "1.26.0"

# CUDA/cuDNN compatibility matrix for the pinned version.
# Source: onnxruntime.ai CUDA EP requirements page (captured 2026-06-13)
# and PyPI metadata (nvidia-cudnn-cu12~=9.0). The docs table lists through
# 1.20.x; 1.26.0 was confirmed via GitHub release v1.26.0 (2026-05-08) and
# PyPI requires_dist.
@dataclass(frozen=True)
class CudaCudnnRow:
    """One row of the ORT CUDA/cuDNN compatibility matrix."""

    ort_version: str
    cuda_major: int
    cudnn_major: int
    pypi_available: bool
    arch_floor: Tuple[int, int]  # minimum compute capability (major, minor)


CUDA_CUDNN_MATRIX: List[CudaCudnnRow] = [
    CudaCudnnRow(
        ort_version="1.26.0",
        cuda_major=12,
        cudnn_major=9,
        pypi_available=True,
        arch_floor=(7, 0),  # sm_70 Volta
    ),
]


@dataclass(frozen=True)
class ProviderPlan:
    """Resolved provider list, dtype, and decision notes."""

    providers: List[str]
    dtype: str
    notes: List[str]


def _arch_floor() -> Tuple[int, int]:
    """Minimum compute capability for the pinned ORT wheel, from the matrix.

    Returns (0, 0) when no matrix row matches, which disables arch gating
    rather than guessing a floor.
    """
    for row in CUDA_CUDNN_MATRIX:
        if row.ort_version == PINNED_ORT_VERSION:
            return row.arch_floor
    return (0, 0)


def _resolve_cuda_dtype() -> str:
    """Resolve CUDA dtype: bfloat16 if supported, else float16.

    Mirrors tests/parity/common.py resolve_dtype exactly (V2/JD-001).
    Called lazily inside provider_plan, never at module import (JD-007).
    """
    try:
        import torch  # noqa: WPS433
        if torch.cuda.is_bf16_supported():
            return "bfloat16"
    except Exception:
        pass
    return "float16"


def provider_plan(cap: HardwareCapability) -> ProviderPlan:
    """Map a HardwareCapability to an ORT provider plan.

    Provider selection is by list membership, never position or length.
    No TensorRT entry anywhere.
    """
    notes: List[str] = []

    # Determine usable CUDA: torch says CUDA is available AND not a ROCm
    # masquerade (amd_rocm_detected with no CUDAExecutionProvider in ORT).
    usable_cuda = cap.cuda_available
    if cap.amd_rocm_detected and "CUDAExecutionProvider" not in cap.onnxruntime_providers:
        usable_cuda = False
        notes.append(
            "ROCm detected but CUDAExecutionProvider absent; treating CUDA as not usable"
        )

    # Arch floor check: a GPU below the wheel's minimum arch gets CPU plan.
    # Floor comes from the matrix, not a hardcoded literal. A None compute
    # capability (nvidia-smi parse miss) is never compared: it is left as
    # usable rather than crashing on `None < tuple`.
    floor = _arch_floor()
    floor_name = f"sm_{floor[0]}{floor[1]}"

    if usable_cuda and cap.physical_gpus:
        for gpu in cap.physical_gpus:
            cc = gpu.compute_capability
            if cc is not None and cc < floor:
                usable_cuda = False
                notes.append(
                    f"GPU {gpu.name} sm_{cc[0]}{cc[1]} below arch floor"
                    f" {floor_name}; falling back to CPU"
                )
                break

    # Also check torch-view GPUs if no physical GPUs were probed.
    if usable_cuda and not cap.physical_gpus and cap.gpus:
        for gpu in cap.gpus:
            cc = gpu.compute_capability
            if cc is not None and cc < floor:
                usable_cuda = False
                notes.append(
                    f"GPU {gpu.name} sm_{cc[0]}{cc[1]} below arch floor"
                    f" {floor_name}; falling back to CPU"
                )
                break

    if usable_cuda:
        dtype = _resolve_cuda_dtype()
        return ProviderPlan(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            dtype=dtype,
            notes=notes,
        )

    return ProviderPlan(
        providers=["CPUExecutionProvider"],
        dtype="float32",
        notes=notes,
    )
