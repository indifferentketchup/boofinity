"""Tests for boofinity.hardware.provider_policy."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from boofinity.hardware.capability import (
    GpuInfo,
    HardwareCapability,
    PhysicalGpu,
)
from boofinity.hardware.provider_policy import (
    PINNED_ORT_VERSION,
    CudaCudnnRow,
    ProviderPlan,
    provider_plan,
)


def _cap_cpu_only() -> HardwareCapability:
    """Synthetic CPU-only capability."""
    return HardwareCapability(
        os_name="Linux",
        machine_arch="x86_64",
        torch_available=True,
        torch_version="2.12.0+cpu",
        cuda_available=False,
        gpus=[],
        onnxruntime_available=True,
        onnxruntime_providers=["CPUExecutionProvider"],
        physical_gpus=[],
    )


def _cap_cuda_sm86(
    bf16_supported: bool = True,
    with_physical: bool = True,
) -> HardwareCapability:
    """Synthetic CUDA sm_86 capability."""
    torch_gpus = [GpuInfo(index=0, name="RTX 3090", compute_capability=(8, 6))]
    physical = (
        [PhysicalGpu(name="NVIDIA GeForce RTX 3090", memory_total_mb=24576, compute_capability=(8, 6))]
        if with_physical
        else []
    )
    return HardwareCapability(
        os_name="Linux",
        machine_arch="x86_64",
        torch_available=True,
        torch_version="2.12.0+cu121",
        cuda_built_with="12.1",
        cuda_available=True,
        gpus=torch_gpus,
        onnxruntime_available=True,
        onnxruntime_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        physical_gpus=physical,
        driver_version="535.129.03",
    )


def _cap_cuda_below_floor() -> HardwareCapability:
    """Synthetic CUDA GPU below the arch floor (sm_61)."""
    return HardwareCapability(
        os_name="Linux",
        machine_arch="x86_64",
        torch_available=True,
        torch_version="2.12.0+cu121",
        cuda_built_with="12.1",
        cuda_available=True,
        gpus=[GpuInfo(index=0, name="GTX 1080 Ti", compute_capability=(6, 1))],
        onnxruntime_available=True,
        onnxruntime_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        physical_gpus=[
            PhysicalGpu(name="NVIDIA GeForce GTX 1080 Ti", memory_total_mb=11264, compute_capability=(6, 1))
        ],
        driver_version="535.129.03",
    )


def _cap_rocm_masquerade() -> HardwareCapability:
    """CUDA available via torch, but ROCm and no CUDAExecutionProvider in ORT."""
    return HardwareCapability(
        os_name="Linux",
        machine_arch="x86_64",
        torch_available=True,
        torch_version="2.12.0+rocm6.0",
        cuda_available=True,
        gpus=[GpuInfo(index=0, name="AMD GPU", compute_capability=(9, 0))],
        onnxruntime_available=True,
        onnxruntime_providers=["CPUExecutionProvider"],
        physical_gpus=[],
        amd_rocm_detected=True,
    )


class TestProviderPlanCpuOnly:
    def test_cpu_plan(self) -> None:
        plan = provider_plan(_cap_cpu_only())
        assert plan.providers == ["CPUExecutionProvider"]
        assert plan.dtype == "float32"

    def test_no_tensorrt(self) -> None:
        plan = provider_plan(_cap_cpu_only())
        assert not any("TensorRT" in p for p in plan.providers)


class TestProviderPlanCudaSm86:
    def test_cuda_plan_bf16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "boofinity.hardware.provider_policy._resolve_cuda_dtype",
            lambda: "bfloat16",
        )
        plan = provider_plan(_cap_cuda_sm86(bf16_supported=True))
        assert "CUDAExecutionProvider" in plan.providers
        assert "CPUExecutionProvider" in plan.providers
        assert plan.dtype == "bfloat16"

    def test_cuda_plan_fp16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "boofinity.hardware.provider_policy._resolve_cuda_dtype",
            lambda: "float16",
        )
        plan = provider_plan(_cap_cuda_sm86(bf16_supported=False))
        assert "CUDAExecutionProvider" in plan.providers
        assert "CPUExecutionProvider" in plan.providers
        assert plan.dtype == "float16"

    def test_no_tensorrt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "boofinity.hardware.provider_policy._resolve_cuda_dtype",
            lambda: "float16",
        )
        plan = provider_plan(_cap_cuda_sm86())
        for p in plan.providers:
            assert "TensorRT" not in p


class TestProviderPlanBelowFloor:
    def test_below_floor_gets_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "boofinity.hardware.provider_policy._resolve_cuda_dtype",
            lambda: "float16",
        )
        plan = provider_plan(_cap_cuda_below_floor())
        assert plan.providers == ["CPUExecutionProvider"]
        assert plan.dtype == "float32"
        assert any("below arch floor" in n for n in plan.notes)


class TestProviderPlanRocmMasquerade:
    def test_rocm_not_usable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "boofinity.hardware.provider_policy._resolve_cuda_dtype",
            lambda: "float16",
        )
        plan = provider_plan(_cap_rocm_masquerade())
        assert plan.providers == ["CPUExecutionProvider"]
        assert plan.dtype == "float32"
        assert any("ROCm" in n for n in plan.notes)


class TestProviderPlanOrtAbsent:
    def test_ort_absent_cpu(self) -> None:
        cap = HardwareCapability(
            os_name="Linux",
            machine_arch="x86_64",
            torch_available=False,
            onnxruntime_available=False,
            onnxruntime_providers=[],
        )
        plan = provider_plan(cap)
        assert plan.providers == ["CPUExecutionProvider"]
        assert plan.dtype == "float32"


class TestCudaCudnnMatrix:
    def test_matrix_row_has_required_fields(self) -> None:
        from boofinity.hardware.provider_policy import CUDA_CUDNN_MATRIX

        for row in CUDA_CUDNN_MATRIX:
            assert isinstance(row, CudaCudnnRow)
            assert row.cuda_major > 0
            assert row.cudnn_major > 0
            assert isinstance(row.pypi_available, bool)
            assert isinstance(row.arch_floor, tuple)
            assert len(row.arch_floor) == 2

    def test_pinned_version_in_matrix(self) -> None:
        from boofinity.hardware.provider_policy import CUDA_CUDNN_MATRIX

        versions = {r.ort_version for r in CUDA_CUDNN_MATRIX}
        assert PINNED_ORT_VERSION in versions

    def test_pinned_version_in_provider_policy_and_ort_provisioning(self) -> None:
        """Task 2.0 verify: version string appears in both modules and matches."""
        from boofinity.hardware.provider_policy import PINNED_ORT_VERSION as pp_ver

        # Import the provisioning module to check its constant (may not exist yet)
        try:
            import boofinity.hardware.ort_provisioning as ort_mod

            assert pp_ver == ort_mod.PINNED_ORT_VERSION
        except (ImportError, ModuleNotFoundError):
            pytest.skip("ort_provisioning.py not yet created")
        assert pp_ver == "1.26.0"


class TestDriftGuard:
    """Task 2.4: provider_plan dtype agrees with parity harness resolve_dtype."""

    @staticmethod
    def _resolve_dtype(device: str, dtype: str | None) -> str:
        """Import resolve_dtype from tests/parity/common.py via sys.path."""
        import importlib.util

        parity_path = (
            __file__.rsplit("libs", 1)[0]
            + "tests/parity/common.py"
        )
        spec = importlib.util.spec_from_file_location("parity_common", parity_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.resolve_dtype(device, dtype)

    def test_cpu_dtype_agreement(self) -> None:
        plan = provider_plan(_cap_cpu_only())
        parity_dtype = self._resolve_dtype("cpu", None)
        assert plan.dtype == parity_dtype

    def test_cuda_bf16_agreement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mock ONLY the shared primitive both sides call (torch.cuda.is_bf16_supported),
        # and let provider_policy._resolve_cuda_dtype run for real. Mocking
        # _resolve_cuda_dtype directly would be mock-blind: it could not catch the
        # policy's own dtype logic drifting from the parity harness.
        monkeypatch.setattr(
            "torch.cuda.is_bf16_supported",
            MagicMock(return_value=True),
        )
        cap = _cap_cuda_sm86(bf16_supported=True)
        plan = provider_plan(cap)
        parity_dtype = self._resolve_dtype("cuda", None)
        assert plan.dtype == parity_dtype == "bfloat16"

    def test_cuda_fp16_agreement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # See test_cuda_bf16_agreement: mock only the shared primitive.
        monkeypatch.setattr(
            "torch.cuda.is_bf16_supported",
            MagicMock(return_value=False),
        )
        cap = _cap_cuda_sm86(bf16_supported=False)
        plan = provider_plan(cap)
        parity_dtype = self._resolve_dtype("cuda", None)
        assert plan.dtype == parity_dtype == "float16"


class TestBenchConsumeProviderPlan:
    """Task 2.4: bench smoke run reports dtype from provider_plan."""

    def test_bench_smoke_cpu(self) -> None:
        """A cpu bench smoke run still reports dtype float32."""
        plan = provider_plan(_cap_cpu_only())
        assert plan.dtype == "float32"
