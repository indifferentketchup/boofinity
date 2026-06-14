"""Tests for boofinity.hardware.capability.detect()."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from boofinity.hardware.capability import (
    HardwareCapability,
    PhysicalGpu,
    detect,
)


@pytest.fixture(autouse=True)
def _clear_detect_cache():
    """Clear detect() lru_cache before each test to prevent cross-test bleed."""
    detect.cache_clear()
    yield
    detect.cache_clear()


class TestNoGPU:
    """Real path on this CPU-only box: no mocks needed."""

    def test_returns_hardware_capability(self):
        cap = detect()
        assert isinstance(cap, HardwareCapability)

    def test_os_and_arch_are_strings(self):
        cap = detect()
        assert isinstance(cap.os_name, str)
        assert len(cap.os_name) > 0
        assert isinstance(cap.machine_arch, str)
        assert len(cap.machine_arch) > 0

    def test_no_gpu_detected(self):
        cap = detect()
        assert cap.cuda_available is False
        assert cap.gpus == []

    def test_torch_detected(self):
        cap = detect()
        assert cap.torch_available is True
        assert cap.torch_version is not None
        assert cap.cuda_built_with is None  # CPU-only build

    def test_onnxruntime_detected(self):
        cap = detect()
        assert cap.onnxruntime_available is True
        assert "CPUExecutionProvider" in cap.onnxruntime_providers

    def test_schema_version(self):
        cap = detect()
        assert cap.schema_version == HardwareCapability.SCHEMA_VERSION == 1

    def test_no_physical_gpus_on_cpu_box(self):
        cap = detect()
        assert cap.physical_gpus == []
        assert cap.driver_version is None

    def test_amd_rocm_not_detected_on_cpu(self):
        cap = detect()
        assert cap.amd_rocm_detected is False


class TestOneCudaGpuSm86:
    """Simulate a single CUDA GPU with sm_86 (RTX 3090 class)."""

    def test_gpu_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_cc = MagicMock(return_value=(8, 6))
        mock_name = MagicMock(return_value="NVIDIA GeForce RTX 3090")
        mock_count = MagicMock(return_value=1)
        mock_is_avail = MagicMock(return_value=True)

        props = MagicMock()
        props.total_memory = 24_576 * 1024 * 1024  # 24 GiB
        mock_props = MagicMock(return_value=props)

        monkeypatch.setattr("torch.cuda.is_available", mock_is_avail)
        monkeypatch.setattr("torch.cuda.device_count", mock_count)
        monkeypatch.setattr("torch.cuda.get_device_name", mock_name)
        monkeypatch.setattr("torch.cuda.get_device_capability", mock_cc)
        monkeypatch.setattr("torch.cuda.get_device_properties", mock_props)

        cap = detect()
        assert cap.cuda_available is True
        assert len(cap.gpus) == 1
        gpu = cap.gpus[0]
        assert gpu.index == 0
        assert gpu.name == "NVIDIA GeForce RTX 3090"
        assert gpu.compute_capability == (8, 6)
        assert gpu.vram_total_bytes == 24_576 * 1024 * 1024


class TestTorchCudaRaisesOnProbe:
    """When torch.cuda.is_available() raises, detection degrades to CPU-only."""

    def test_cuda_probe_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "torch.cuda.is_available",
            MagicMock(side_effect=RuntimeError("CUDA driver not loaded")),
        )
        cap = detect()
        assert cap.torch_available is True
        assert cap.cuda_available is False
        assert cap.gpus == []

    def test_device_count_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("torch.cuda.is_available", MagicMock(return_value=True))
        monkeypatch.setattr(
            "torch.cuda.device_count",
            MagicMock(side_effect=RuntimeError("no devices")),
        )
        cap = detect()
        assert cap.cuda_available is True
        assert cap.gpus == []


class TestOnnxruntimeAbsent:
    """When onnxruntime is not importable, detection reports it absent."""

    def test_ort_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Block onnxruntime import by setting it to None in sys.modules.
        real_ort = sys.modules.get("onnxruntime")

        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        try:
            cap = detect()
            assert cap.onnxruntime_available is False
            assert cap.onnxruntime_providers == []
        finally:
            if real_ort is not None:
                sys.modules["onnxruntime"] = real_ort
            else:
                sys.modules.pop("onnxruntime", None)


class TestOnnxruntimeWithCudaProvider:
    """When onnxruntime has CUDA provider listed."""

    def test_ort_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_ort = sys.modules.get("onnxruntime")
        fake_ort = ModuleType("onnxruntime")
        fake_ort.get_available_providers = MagicMock(
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
        try:
            cap = detect()
            assert cap.onnxruntime_available is True
            assert "CUDAExecutionProvider" in cap.onnxruntime_providers
            assert "CPUExecutionProvider" in cap.onnxruntime_providers
        finally:
            if real_ort is not None:
                sys.modules["onnxruntime"] = real_ort
            else:
                sys.modules.pop("onnxruntime", None)


class TestResolveNvidiaSmi:
    """Task 1.2: _resolve_nvidia_smi() absolute path resolution."""

    def test_canonical_path_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from boofinity.hardware.capability import _resolve_nvidia_smi

        def fake_exists(path):
            return path == "/usr/bin/nvidia-smi"

        monkeypatch.setattr("os.path.exists", fake_exists)
        assert _resolve_nvidia_smi() == "/usr/bin/nvidia-smi"

    def test_second_canonical_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from boofinity.hardware.capability import _resolve_nvidia_smi

        def fake_exists(path):
            return path == "/usr/local/bin/nvidia-smi"

        monkeypatch.setattr("os.path.exists", fake_exists)
        assert _resolve_nvidia_smi() == "/usr/local/bin/nvidia-smi"

    def test_shutil_which_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from boofinity.hardware.capability import _resolve_nvidia_smi

        monkeypatch.setattr("os.path.exists", lambda _: False)
        monkeypatch.setattr("shutil.which", lambda name, path=None: "/usr/lib/wsl/lib/nvidia-smi")
        result = _resolve_nvidia_smi()
        assert result is not None
        assert "nvidia-smi" in result

    def test_none_when_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from boofinity.hardware.capability import _resolve_nvidia_smi

        monkeypatch.setattr("os.path.exists", lambda _: False)
        monkeypatch.setattr("shutil.which", lambda name, path=None: None)
        assert _resolve_nvidia_smi() is None


class TestProbePhysicalGpus:
    """Task 1.3: _probe_physical_gpus() with mocked nvidia-smi CSV."""

    def test_two_gpu_csv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from boofinity.hardware.capability import _probe_physical_gpus

        monkeypatch.setattr(
            "boofinity.hardware.capability._resolve_nvidia_smi",
            lambda: "/usr/bin/nvidia-smi",
        )
        csv_output = (
            "NVIDIA GeForce RTX 3090, 24576, 8.6, 535.129.03\n"
            "NVIDIA GeForce RTX 3080, 10240, 8.6, 535.129.03"
        )
        monkeypatch.setattr(
            "boofinity.hardware.capability._run",
            lambda cmd, timeout=2.0: csv_output,
        )
        gpus, drv = _probe_physical_gpus()
        assert len(gpus) == 2
        assert gpus[0].name == "NVIDIA GeForce RTX 3090"
        assert gpus[0].memory_total_mb == 24576
        assert gpus[0].compute_capability == (8, 6)
        assert gpus[1].name == "NVIDIA GeForce RTX 3080"
        assert gpus[1].memory_total_mb == 10240
        assert drv == "535.129.03"

    def test_no_nvidia_smi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from boofinity.hardware.capability import _probe_physical_gpus

        monkeypatch.setattr(
            "boofinity.hardware.capability._resolve_nvidia_smi",
            lambda: None,
        )
        gpus, drv = _probe_physical_gpus()
        assert gpus == []
        assert drv is None

    def test_nvidia_smi_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from boofinity.hardware.capability import _probe_physical_gpus

        monkeypatch.setattr(
            "boofinity.hardware.capability._resolve_nvidia_smi",
            lambda: "/usr/bin/nvidia-smi",
        )
        monkeypatch.setattr(
            "boofinity.hardware.capability._run",
            lambda cmd, timeout=2.0: None,
        )
        gpus, drv = _probe_physical_gpus()
        assert gpus == []
        assert drv is None


class TestRocmGuard:
    """Task 1.4: ROCm false-positive guard."""

    def test_rocm_detected_with_amdgpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from boofinity.hardware.capability import _probe_rocm

        def fake_exists(path):
            return path == "/sys/module/amdgpu"

        monkeypatch.setattr("os.path.exists", fake_exists)
        assert _probe_rocm() is True

    def test_rocm_detected_with_kfd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from boofinity.hardware.capability import _probe_rocm

        def fake_exists(path):
            return path == "/dev/kfd"

        monkeypatch.setattr("os.path.exists", fake_exists)
        assert _probe_rocm() is True

    def test_rocm_not_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from boofinity.hardware.capability import _probe_rocm

        monkeypatch.setattr("os.path.exists", lambda _: False)
        assert _probe_rocm() is False

    def test_rocm_field_on_hardware_capability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """amd_rocm_detected is True when /dev/kfd exists, even with cuda_available=True."""
        from boofinity.hardware.capability import detect

        monkeypatch.setattr(
            "boofinity.hardware.capability._probe_rocm",
            lambda: True,
        )
        monkeypatch.setattr(
            "boofinity.hardware.capability._probe_physical_gpus",
            lambda: ([], None),
        )
        # Also mock torch to report cuda_available=True so the raw flag is set.
        monkeypatch.setattr("torch.cuda.is_available", MagicMock(return_value=True))
        monkeypatch.setattr("torch.cuda.device_count", MagicMock(return_value=0))
        # Torch reports CUDA available, but ORT has no CUDAExecutionProvider
        real_ort = sys.modules.get("onnxruntime")
        fake_ort = ModuleType("onnxruntime")
        fake_ort.get_available_providers = MagicMock(
            return_value=["CPUExecutionProvider"]
        )
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
        try:
            cap = detect()
            assert cap.amd_rocm_detected is True
            assert cap.cuda_available is True  # raw torch flag preserved
            assert "CUDAExecutionProvider" not in cap.onnxruntime_providers
        finally:
            if real_ort is not None:
                sys.modules["onnxruntime"] = real_ort
            else:
                sys.modules.pop("onnxruntime", None)


class TestSchemaVersion:
    """Task 1.5: SCHEMA_VERSION class constant and schema_version field."""

    def test_schema_version_constant(self) -> None:
        assert HardwareCapability.SCHEMA_VERSION == 1

    def test_schema_version_field(self) -> None:
        cap = detect()
        assert cap.schema_version == 1

    def test_schema_version_on_constructed(self) -> None:
        cap = HardwareCapability(
            os_name="Linux",
            machine_arch="x86_64",
            torch_available=False,
        )
        assert cap.schema_version == HardwareCapability.SCHEMA_VERSION


class TestMemoization:
    """Task 1.6: detect() is memoized; autouse fixture clears cache."""

    def test_second_call_returns_same_object(self) -> None:
        cap1 = detect()
        cap2 = detect()
        assert cap1 is cap2

    def test_cache_clear_works(self) -> None:
        cap1 = detect()
        detect.cache_clear()
        cap2 = detect()
        assert cap1 is not cap2
