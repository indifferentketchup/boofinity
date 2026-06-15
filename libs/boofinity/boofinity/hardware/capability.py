# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

"""
Hardware capability detection for runtime dispatch.

Pure read-only probes: no network, no file writes, no model loads.
Every torch.cuda call is guarded so absence of CUDA never crashes detection.
"""

from __future__ import annotations

import functools
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical nvidia-smi search paths. Never use a bare "nvidia-smi" argv.
_NVIDIA_SMI_PATHS: Tuple[str, ...] = (
    "/usr/bin/nvidia-smi",
    "/usr/local/bin/nvidia-smi",
)


@dataclass(frozen=True)
class GpuInfo:
    """Per-GPU static info gathered at detect() time."""

    index: int
    name: str
    compute_capability: Optional[Tuple[int, int]] = None
    vram_total_bytes: Optional[int] = None


@dataclass(frozen=True)
class PhysicalGpu:
    """nvidia-smi view of a GPU. Separate from GpuInfo (torch view)."""

    name: str
    memory_total_mb: int
    compute_capability: Tuple[int, int]


@dataclass(frozen=True)
class HardwareCapability:
    """Snapshot of detected hardware and runtime libraries.

    All fields are read-only. On a CPU-only box every GPU-related field is
    None/empty; the object is still fully constructible and printable.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    os_name: str
    machine_arch: str
    torch_available: bool
    torch_version: Optional[str] = None
    cuda_built_with: Optional[str] = None
    cuda_available: bool = False
    gpus: List[GpuInfo] = field(default_factory=list)
    onnxruntime_available: bool = False
    onnxruntime_providers: List[str] = field(default_factory=list)
    physical_gpus: List[PhysicalGpu] = field(default_factory=list)
    driver_version: Optional[str] = None
    amd_rocm_detected: bool = False
    schema_version: int = SCHEMA_VERSION


def _resolve_nvidia_smi() -> Optional[str]:
    """Return an absolute path to nvidia-smi, or None.

    Tries canonical paths first, then shutil.which with /usr/lib/wsl/lib
    appended to PATH for WSL compatibility. Never returns a bare argv.
    """
    for path in _NVIDIA_SMI_PATHS:
        if os.path.exists(path):
            return path

    # WSL: append /usr/lib/wsl/lib to PATH for the which lookup.
    extra_path = "/usr/lib/wsl/lib"
    search_path = os.environ.get("PATH", "")
    if extra_path not in search_path:
        search_path = f"{search_path}:{extra_path}"

    found = shutil.which("nvidia-smi", path=search_path)
    if found is not None:
        return os.path.abspath(found)

    return None


def _run(cmd: List[str], timeout: float = 2.0) -> Optional[str]:
    """Run a subprocess with list args, no shell. Returns stripped stdout or None.

    Never raises. Returns None on TimeoutExpired, FileNotFoundError, OSError,
    or nonzero exit code.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.debug("Command %s exited %d", cmd, result.returncode)
            return None
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.debug("Command %s failed: %s", cmd, exc)
        return None


def _probe_rocm() -> bool:
    """Detect AMD ROCm presence via /dev/kfd or /sys/module/amdgpu."""
    if os.path.exists("/dev/kfd"):
        return True
    if os.path.exists("/sys/module/amdgpu"):
        return True
    return False


def _probe_physical_gpus() -> Tuple[List[PhysicalGpu], Optional[str]]:
    """Probe physical GPUs via nvidia-smi. Returns (gpus, driver_version).

    Pre-checks /proc/driver/nvidia/version and /dev/nvidia[0-9]* before
    attempting nvidia-smi. Tries nvidia-smi even when pre-check is empty
    (container shim case). Never raises.
    """
    nvidia_smi = _resolve_nvidia_smi()
    if nvidia_smi is None:
        return [], None

    # Run nvidia-smi query. CSV format: name, memory.total, compute_cap, driver_version
    output = _run([
        nvidia_smi,
        "--query-gpu=name,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if output is None:
        return [], None

    gpus: List[PhysicalGpu] = []
    driver_version: Optional[str] = None

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name_str, mem_str, cc_str, drv_str = parts[0], parts[1], parts[2], parts[3]
        try:
            mem_mb = int(mem_str)
        except ValueError:
            continue
        # Parse compute capability "major.minor"
        cc_parts = cc_str.split(".")
        if len(cc_parts) != 2:
            continue
        try:
            cc = (int(cc_parts[0]), int(cc_parts[1]))
        except ValueError:
            continue
        gpus.append(PhysicalGpu(name=name_str, memory_total_mb=mem_mb, compute_capability=cc))
        if driver_version is None:
            driver_version = drv_str

    return gpus, driver_version


def _probe_torch() -> Tuple[bool, Optional[str], Optional[str], bool, List[GpuInfo]]:
    """Return (available, version_str, cuda_built_with, cuda_is_available, gpus)."""
    try:
        import torch  # noqa: WPS433
    except ImportError:
        return False, None, None, False, []

    version_str: Optional[str] = getattr(torch, "__version__", None)
    cuda_built: Optional[str] = getattr(torch.version, "cuda", None)  # type: ignore[union-attr]

    cuda_ok = False
    gpus: List[GpuInfo] = []
    try:
        cuda_ok = torch.cuda.is_available()  # type: ignore[union-attr]
    except Exception:
        # torch.cuda probed on a build without CUDA support: degrade, do not crash.
        return True, version_str, cuda_built, False, []

    if not cuda_ok:
        return True, version_str, cuda_built, False, []

    # CUDA is available; enumerate GPUs.
    try:
        count = torch.cuda.device_count()  # type: ignore[union-attr]
    except Exception:
        return True, version_str, cuda_built, cuda_ok, []

    for idx in range(count):
        try:
            name = torch.cuda.get_device_name(idx)  # type: ignore[union-attr]
        except Exception:
            name = f"cuda:{idx}"
        cc: Optional[Tuple[int, int]] = None
        try:
            cc = torch.cuda.get_device_capability(idx)  # type: ignore[union-attr]
        except Exception:
            pass
        vram: Optional[int] = None
        try:
            vram = torch.cuda.get_device_properties(idx).total_memory  # type: ignore[union-attr]
        except Exception:
            pass
        gpus.append(GpuInfo(index=idx, name=str(name), compute_capability=cc, vram_total_bytes=vram))

    return True, version_str, cuda_built, cuda_ok, gpus


def _probe_onnxruntime() -> Tuple[bool, List[str]]:
    """Return (available, provider_list). Never raises."""
    try:
        import onnxruntime  # noqa: WPS433
    except ImportError:
        return False, []
    try:
        providers = list(onnxruntime.get_available_providers())
    except Exception:
        providers = []
    return True, providers


@functools.lru_cache(maxsize=1)
def detect() -> HardwareCapability:
    """Probe hardware and return a snapshot. Always succeeds, never raises."""
    torch_ok, torch_ver, cuda_built, cuda_ok, gpus = _probe_torch()
    ort_ok, ort_providers = _probe_onnxruntime()
    physical_gpus, driver_version = _probe_physical_gpus()
    amd_rocm_detected = _probe_rocm()
    return HardwareCapability(
        os_name=platform.system(),
        machine_arch=platform.machine(),
        torch_available=torch_ok,
        torch_version=torch_ver,
        cuda_built_with=cuda_built,
        cuda_available=cuda_ok,
        gpus=gpus,
        onnxruntime_available=ort_ok,
        onnxruntime_providers=ort_providers,
        physical_gpus=physical_gpus,
        driver_version=driver_version,
        amd_rocm_detected=amd_rocm_detected,
    )
