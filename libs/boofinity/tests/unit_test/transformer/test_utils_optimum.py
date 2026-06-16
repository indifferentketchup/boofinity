"""Tests for boofinity.transformer.utils_optimum.device_to_onnx (task 8.4)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from boofinity.primitives import Device


def _patch_providers(monkeypatch: pytest.MonkeyPatch, providers: list[str]) -> None:
    """Install a fake onnxruntime that reports the given provider list.

    Also neutralizes the optimum.onnxruntime availability gate so the routing
    logic can be exercised without the optional optimum dependency installed.
    """
    import boofinity.transformer.utils_optimum as mod

    fake_ort = ModuleType("onnxruntime")
    fake_ort.get_available_providers = MagicMock(return_value=providers)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(mod.CHECK_ONNXRUNTIME, "mark_required", lambda: True)


def _device_to_onnx(device: Device, enable_webgpu_ep: bool = False) -> str:
    from boofinity.transformer.utils_optimum import device_to_onnx

    return device_to_onnx(device, enable_webgpu_ep)


class TestDeviceToOnnxAmd:
    """device_to_onnx prefers MIGraphX over ROCm for the AMD cuda case."""

    def test_cuda_prefers_migraphx_over_rocm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_providers(
            monkeypatch,
            ["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"],
        )
        assert _device_to_onnx(Device.cuda) == "MIGraphXExecutionProvider"

    def test_cuda_rocm_only_falls_back_to_rocm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_providers(
            monkeypatch, ["ROCMExecutionProvider", "CPUExecutionProvider"]
        )
        assert _device_to_onnx(Device.cuda) == "ROCMExecutionProvider"

    def test_cuda_nvidia_uses_cuda_ep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_providers(
            monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        assert _device_to_onnx(Device.cuda) == "CUDAExecutionProvider"


class TestDeviceToOnnxCpu:
    def test_cpu_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_providers(monkeypatch, ["CPUExecutionProvider"])
        assert _device_to_onnx(Device.cpu) == "CPUExecutionProvider"

    def test_cpu_openvino(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_providers(
            monkeypatch, ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        )
        assert _device_to_onnx(Device.cpu) == "OpenVINOExecutionProvider"


class TestDeviceToOnnxWebGpu:
    """Task 9: the enable_webgpu_ep opt-in routes ONNX loads to WebGPU."""

    def test_flag_off_ignores_webgpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_providers(
            monkeypatch, ["WebGpuExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        # Default (flag off): WebGPU present but not selected.
        assert _device_to_onnx(Device.cuda) == "CUDAExecutionProvider"

    def test_flag_on_present_selects_webgpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_providers(
            monkeypatch, ["WebGpuExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        assert _device_to_onnx(Device.cuda, enable_webgpu_ep=True) == "WebGpuExecutionProvider"

    def test_flag_on_auto_selects_webgpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_providers(
            monkeypatch, ["WebGpuExecutionProvider", "CPUExecutionProvider"]
        )
        assert _device_to_onnx(Device.auto, enable_webgpu_ep=True) == "WebGpuExecutionProvider"

    def test_flag_on_absent_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Flag on but the wheel does not expose WebGPU: normal selection.
        _patch_providers(
            monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        assert _device_to_onnx(Device.cuda, enable_webgpu_ep=True) == "CUDAExecutionProvider"

    def test_flag_on_explicit_cpu_is_honored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Explicit device=cpu is honored even with the flag and WebGPU present.
        _patch_providers(
            monkeypatch, ["WebGpuExecutionProvider", "CPUExecutionProvider"]
        )
        assert _device_to_onnx(Device.cpu, enable_webgpu_ep=True) == "CPUExecutionProvider"
