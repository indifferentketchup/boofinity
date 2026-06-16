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


# Keyed PER ORT WHEEL: each row carries its own arch_floor, because the floor
# is a property of the wheel build, not the CUDA toolkit major. The stock ORT
# 1.26 CUDA 12 wheel floors at sm_70 and ships NO Pascal sm_61 kernels, so a
# CUDA 12 row here must NOT silently re-assert a Pascal-admitting floor. A lower
# floor for Pascal would require an explicitly pinned, verified wheel recorded
# as its own row (correction folded from validation V6).
CUDA_CUDNN_MATRIX: List[CudaCudnnRow] = [
    CudaCudnnRow(
        ort_version="1.26.0",
        cuda_major=12,
        cudnn_major=9,
        pypi_available=True,
        arch_floor=(7, 0),  # sm_70 Volta; stock wheel ships no sm_61 kernels
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
    except Exception as exc:  # noqa: BLE001
        # torch absent or CUDA not initialized: fall back to float16. Logged
        # at debug so the degrade is traceable instead of silently swallowed.
        logger.debug("bf16 probe failed, defaulting CUDA dtype to float16: %s", exc)
    return "float16"


def _amd_plan(cap: HardwareCapability, notes: List[str]) -> ProviderPlan:
    """MIGraphX-primary AMD provider plan.

    Upstream ONNX Runtime removed the ROCm execution provider at release 1.23;
    MIGraphXExecutionProvider is the supported AMD provider. AMD's downstream
    onnxruntime-rocm wheels may still expose ROCMExecutionProvider, so it is
    inserted as an optional fallback only when the running wheel reports it.
    Falls back to CPU when no MIGraphX provider is present.
    """
    providers = cap.onnxruntime_providers
    if "MIGraphXExecutionProvider" not in providers:
        notes.append(
            "AMD ROCm detected but MIGraphXExecutionProvider absent; no usable"
            " AMD ONNX provider, falling back to CPU"
        )
        return ProviderPlan(
            providers=["CPUExecutionProvider"],
            dtype="float32",
            notes=notes,
        )

    plan: List[str] = ["MIGraphXExecutionProvider"]
    if "ROCMExecutionProvider" in providers:
        plan.append("ROCMExecutionProvider")
    plan.append("CPUExecutionProvider")
    notes.append("AMD ROCm: MIGraphXExecutionProvider leads ONNX plan")
    return ProviderPlan(providers=plan, dtype=_resolve_cuda_dtype(), notes=notes)


def _webgpu_plan(
    cap: HardwareCapability, base: ProviderPlan, notes: List[str]
) -> ProviderPlan:
    """WebGPU (Vulkan-via-Dawn on Linux) plan when explicitly requested.

    Experimental and default off. When WebGpuExecutionProvider is present the
    plan leads with it and keeps CPU as the tail; when absent the base plan is
    returned unchanged with a note that WebGPU was requested but unavailable.
    No VulkanExecutionProvider string is ever emitted (none exists upstream).
    """
    if "WebGpuExecutionProvider" not in cap.onnxruntime_providers:
        notes.append(
            "WebGPU EP requested but WebGpuExecutionProvider absent; using"
            " normal plan"
        )
        return ProviderPlan(providers=base.providers, dtype=base.dtype, notes=notes)
    notes.append("WebGPU EP requested and present (experimental, Vulkan via Dawn)")
    return ProviderPlan(
        providers=["WebGpuExecutionProvider", "CPUExecutionProvider"],
        dtype=base.dtype,
        notes=notes,
    )


def provider_plan(
    cap: HardwareCapability, enable_webgpu_ep: bool = False
) -> ProviderPlan:
    """Map a HardwareCapability to an ORT provider plan.

    Provider selection is by list membership, never position or length.
    No TensorRT entry anywhere. ``enable_webgpu_ep`` is the default-off opt-in
    for the experimental WebGPU (Vulkan) provider.
    """
    notes: List[str] = []

    # ROCm routes to the MIGraphX-primary plan, not the NVIDIA CUDA branch.
    if cap.amd_rocm_detected:
        plan = _amd_plan(cap, notes)
        if enable_webgpu_ep:
            return _webgpu_plan(cap, plan, notes)
        return plan

    usable_cuda = cap.cuda_available

    # Arch floor check: a GPU below the wheel's minimum arch gets CPU plan.
    # Floor comes from the matrix, not a hardcoded literal. A None compute
    # capability (nvidia-smi parse miss) is never compared: it is left as
    # usable rather than crashing on `None < tuple`. Pascal sm_61 is below the
    # stock 1.26 wheel's sm_70 floor, so it resolves to the CPU ONNX plan;
    # Pascal GPU serving is expected via the torch CUDA 12 path, not stock ONNX.
    floor = _arch_floor()
    floor_name = f"sm_{floor[0]}{floor[1]}"

    if usable_cuda and cap.physical_gpus:
        for gpu in cap.physical_gpus:
            cc = gpu.compute_capability
            if cc is not None and cc < floor:
                usable_cuda = False
                notes.append(
                    f"GPU {gpu.name} sm_{cc[0]}{cc[1]} below matched wheel"
                    f" compute-capability floor {floor_name}; ONNX falls back"
                    " to CPU (Pascal GPU serving uses the torch CUDA 12 path)"
                )
                break

    # Also check torch-view GPUs if no physical GPUs were probed.
    if usable_cuda and not cap.physical_gpus and cap.gpus:
        for gpu in cap.gpus:
            cc = gpu.compute_capability
            if cc is not None and cc < floor:
                usable_cuda = False
                notes.append(
                    f"GPU {gpu.name} sm_{cc[0]}{cc[1]} below matched wheel"
                    f" compute-capability floor {floor_name}; ONNX falls back"
                    " to CPU (Pascal GPU serving uses the torch CUDA 12 path)"
                )
                break

    if usable_cuda and "CUDAExecutionProvider" in cap.onnxruntime_providers:
        plan = ProviderPlan(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            dtype=_resolve_cuda_dtype(),
            notes=notes,
        )
    else:
        plan = ProviderPlan(
            providers=["CPUExecutionProvider"],
            dtype="float32",
            notes=notes,
        )

    if enable_webgpu_ep:
        return _webgpu_plan(cap, plan, notes)
    return plan
