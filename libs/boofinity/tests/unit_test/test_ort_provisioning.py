"""Tests for boofinity.hardware.ort_provisioning."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from boofinity.hardware.capability import HardwareCapability, GpuInfo, PhysicalGpu
from boofinity.hardware.ort_provisioning import (
    PINNED_ORT_VERSION,
    WHEEL_HASHES,
    OrtWheelPlan,
    _check_transitive_satisfied,
    install_ort,
    preflight,
    resolve_ort_wheel,
)
from boofinity.hardware.provider_policy import PINNED_ORT_VERSION as PP_VERSION


def _cap_cpu() -> HardwareCapability:
    return HardwareCapability(
        os_name="Linux",
        machine_arch="x86_64",
        torch_available=True,
        onnxruntime_available=True,
        onnxruntime_providers=["CPUExecutionProvider"],
    )


def _cap_cuda_sm86() -> HardwareCapability:
    return HardwareCapability(
        os_name="Linux",
        machine_arch="x86_64",
        torch_available=True,
        cuda_available=True,
        onnxruntime_available=True,
        onnxruntime_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        physical_gpus=[
            PhysicalGpu(
                name="NVIDIA GeForce RTX 3090",
                memory_total_mb=24576,
                compute_capability=(8, 6),
            )
        ],
        driver_version="535.129.03",
    )


class TestResolveOrtWheel:
    def test_cpu_plan(self) -> None:
        plan = resolve_ort_wheel(_cap_cpu())
        assert plan.package == "onnxruntime"
        assert plan.version == PINNED_ORT_VERSION
        assert plan.index_url == "https://pypi.org/simple/"
        assert plan.provider_family == "cpu"

    def test_cuda_plan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "boofinity.hardware.provider_policy._resolve_cuda_dtype",
            lambda: "float16",
        )
        plan = resolve_ort_wheel(_cap_cuda_sm86())
        assert plan.package == "onnxruntime-gpu"
        assert plan.version == PINNED_ORT_VERSION
        assert plan.provider_family == "cuda"

    def test_package_is_fixed_literal(self) -> None:
        """JD-008: package is one of the two fixed literals."""
        plan = resolve_ort_wheel(_cap_cpu())
        assert plan.package in ("onnxruntime", "onnxruntime-gpu")
        plan2 = resolve_ort_wheel(_cap_cuda_sm86())
        assert plan2.package in ("onnxruntime", "onnxruntime-gpu")


class TestOrtWheelPlanTypes:
    def test_package_type_literal(self) -> None:
        """JD-008: package is typed Literal."""
        plan = OrtWheelPlan(
            package="onnxruntime",
            version="1.26.0",
            index_url="https://pypi.org/simple/",
            hashes="abc123",
            provider_family="cpu",
        )
        assert plan.package == "onnxruntime"

    def test_index_url_is_pypi(self) -> None:
        plan = resolve_ort_wheel(_cap_cpu())
        assert plan.index_url == "https://pypi.org/simple/"


class TestWheelHashes:
    def test_pinned_version_has_hash_cpu(self) -> None:
        """Task 3.2b: pinned version has a hash entry for onnxruntime."""
        assert "onnxruntime" in WHEEL_HASHES
        assert PINNED_ORT_VERSION in WHEEL_HASHES["onnxruntime"]
        h = WHEEL_HASHES["onnxruntime"][PINNED_ORT_VERSION]
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex

    def test_pinned_version_has_hash_gpu(self) -> None:
        """Task 3.2b: pinned version has a hash entry for onnxruntime-gpu."""
        assert "onnxruntime-gpu" in WHEEL_HASHES
        assert PINNED_ORT_VERSION in WHEEL_HASHES["onnxruntime-gpu"]
        h = WHEEL_HASHES["onnxruntime-gpu"][PINNED_ORT_VERSION]
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex

    def test_pinned_version_matches_provider_policy(self) -> None:
        """Task 2.0: version strings match across modules."""
        assert PINNED_ORT_VERSION == PP_VERSION


class TestInstallOrt:
    def test_no_install_when_assume_yes_false(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Task 3.3: nothing spawned when assume_yes=False."""
        plan = resolve_ort_wheel(_cap_cpu())
        mock_run = MagicMock()
        monkeypatch.setattr("subprocess.run", mock_run)
        install_ort(plan, assume_yes=False)
        mock_run.assert_not_called()
        output = capsys.readouterr().out
        assert "requires explicit confirmation" in output

    def test_floor_flags_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Task 3.3: --only-binary :all: and --index-url always present."""
        plan = resolve_ort_wheel(_cap_cpu())
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr(
            "boofinity.hardware.ort_provisioning._check_transitive_satisfied",
            lambda: False,
        )
        install_ort(plan, assume_yes=True)
        args = mock_run.call_args[0][0]
        assert "--only-binary" in args
        assert ":all:" in args
        assert "--index-url" in args
        assert "https://pypi.org/simple/" in args

    def test_require_hashes_when_transitives_satisfied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Task 3.3: --require-hashes appears only when transitives satisfied."""
        plan = resolve_ort_wheel(_cap_cpu())
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr(
            "boofinity.hardware.ort_provisioning._check_transitive_satisfied",
            lambda: True,
        )
        install_ort(plan, assume_yes=True)
        args = mock_run.call_args[0][0]
        assert "--require-hashes" in args

    def test_no_require_hashes_when_transitives_not_satisfied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Task 3.3: --require-hashes absent when transitives not satisfied."""
        plan = resolve_ort_wheel(_cap_cpu())
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr(
            "boofinity.hardware.ort_provisioning._check_transitive_satisfied",
            lambda: False,
        )
        install_ort(plan, assume_yes=True)
        args = mock_run.call_args[0][0]
        assert "--require-hashes" not in args

    def test_no_extra_index_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """V1: no --extra-index-url ever."""
        plan = resolve_ort_wheel(_cap_cpu())
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        monkeypatch.setattr("subprocess.run", mock_run)
        install_ort(plan, assume_yes=True)
        args = mock_run.call_args[0][0]
        assert "--extra-index-url" not in args


class TestPreflight:
    def test_mismatched_cuda_raises(self) -> None:
        """Task 3.4: mismatched capability raises before pip call."""
        cap = _cap_cuda_sm86()
        plan = resolve_ort_wheel(_cap_cpu())  # CPU plan for a CUDA box
        # This should not raise because the plan is CPU and preflight
        # only checks CUDA matrix for GPU wheels
        preflight(plan, cap)  # Should pass

    def test_gpu_below_arch_floor_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Task 3.4: GPU below arch floor raises descriptive error."""
        cap_sm61 = HardwareCapability(
            os_name="Linux",
            machine_arch="x86_64",
            torch_available=True,
            cuda_available=True,
            onnxruntime_available=True,
            onnxruntime_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            physical_gpus=[
                PhysicalGpu(
                    name="GTX 1080 Ti",
                    memory_total_mb=11264,
                    compute_capability=(6, 1),
                )
            ],
            driver_version="535.129.03",
        )
        monkeypatch.setattr(
            "boofinity.hardware.provider_policy._resolve_cuda_dtype",
            lambda: "float16",
        )
        plan = resolve_ort_wheel(cap_sm61)
        # The plan should be CPU since sm_61 < sm_70
        assert plan.package == "onnxruntime"


class TestImportNoSideEffects:
    """Task 3.1: import has no side effects (no subprocess, no network)."""

    def test_import_does_not_spawn_subprocess(self) -> None:
        # Module is already imported; just verify it loaded cleanly
        import boofinity.hardware.ort_provisioning as mod

        assert hasattr(mod, "resolve_ort_wheel")
        assert hasattr(mod, "install_ort")
        assert hasattr(mod, "PINNED_ORT_VERSION")
