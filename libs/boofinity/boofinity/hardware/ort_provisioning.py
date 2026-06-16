# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

"""
ORT wheel resolver and optional installer.

Report-only by default; never runs on import. The install path is gated
behind explicit --install-ort --yes and performs preflight validation
before any network or pip call.

Security: unconditional floor is pinned version + single --index-url
https://pypi.org/simple/ + --only-binary :all:. --require-hashes is
conditional on transitive-satisfaction preflight (V1).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

from boofinity.hardware.capability import HardwareCapability, detect
from boofinity.hardware.provider_policy import (
    CUDA_CUDNN_MATRIX,
    PINNED_ORT_VERSION,
    provider_plan,
)

logger = logging.getLogger(__name__)

# Fixed package names. Typed as Literal to enforce at the type level (JD-008).
# Never constructed from input.
PackageType = Literal["onnxruntime", "onnxruntime-gpu"]

# PyPI wheel hashes for the pinned version, cp312 linux_x86_64.
# Fetched 2026-06-13 from https://pypi.org/pypi/<pkg>/<ver>/json
WHEEL_HASHES: dict[str, dict[str, str]] = {
    "onnxruntime": {
        "1.26.0": "9b6dd70599005bd1bf29779f04a91978b92b5e719c11a20068a8f8e535f725b6",
    },
    "onnxruntime-gpu": {
        "1.26.0": "3c01119ed4d9449d60367fa8ccffcd02bd3fe736754284e4b198d131f54edad6",
    },
}

# Transitive dependencies of onnxruntime-gpu that must be pre-satisfied
# for --require-hashes to work (pip refuses hash-checked install unless
# every transitive dep is also pinned and hashed).
_ORT_TRANSITIVE_DEPS = ("flatbuffers", "numpy", "packaging", "protobuf")


@dataclass(frozen=True)
class OrtWheelPlan:
    """Resolved ORT wheel plan. Report-only; never installs."""

    package: PackageType
    version: str
    index_url: str
    hashes: str
    provider_family: str


def _detect_platform_tag() -> str:
    """Return the platform tag for the current interpreter."""
    if platform.system() == "Linux" and platform.machine() == "x86_64":
        return "linux_x86_64"
    return f"{platform.system().lower()}_{platform.machine()}"


def _check_transitive_satisfied() -> bool:
    """Check if all transitive deps of onnxruntime are already importable."""
    for dep in _ORT_TRANSITIVE_DEPS:
        try:
            importlib.metadata.version(dep)
        except importlib.metadata.PackageNotFoundError:
            return False
    return True


def resolve_ort_wheel(cap: HardwareCapability) -> OrtWheelPlan:
    """Resolve which ORT wheel matches the detected hardware.

    Report-only; never installs anything. Package names are fixed literals
    (JD-008), never constructed from input.
    """
    # provider_plan here decides the ORT WHEEL to install (CUDA vs CPU base);
    # the experimental WebGPU (Vulkan) path is a separate `onnxruntime-webgpu`
    # wheel and its runtime selection lives in transformer.utils_optimum.
    # device_to_onnx (gated by EngineArgs.enable_webgpu_ep), so the flag is
    # intentionally not threaded into wheel resolution.
    plan = provider_plan(cap)
    usable_cuda = "CUDAExecutionProvider" in plan.providers

    if usable_cuda:
        pkg: PackageType = "onnxruntime-gpu"
    else:
        pkg = "onnxruntime"

    version = PINNED_ORT_VERSION
    hashes = WHEEL_HASHES.get(pkg, {}).get(version, "")

    return OrtWheelPlan(
        package=pkg,
        version=version,
        index_url="https://pypi.org/simple/",
        hashes=hashes,
        provider_family="cuda" if usable_cuda else "cpu",
    )


def preflight(plan: OrtWheelPlan, cap: HardwareCapability) -> None:
    """Validate preconditions before any pip or network call.

    Raises clear errors on mismatch. Never downloads anything.
    """
    # Platform check
    platform_tag = _detect_platform_tag()
    if platform_tag != "linux_x86_64":
        raise RuntimeError(
            f"Unsupported platform: {platform_tag}. Only linux_x86_64 is supported."
        )

    # CUDA major match check (only for GPU wheel)
    if plan.package == "onnxruntime-gpu":
        # Find the matrix row for the pinned version
        matrix_row = None
        for row in CUDA_CUDNN_MATRIX:
            if row.ort_version == plan.version:
                matrix_row = row
                break
        if matrix_row is None:
            raise RuntimeError(
                f"No CUDA/cuDNN matrix row for ORT {plan.version}"
            )

        # Arch-floor check (the host CUDA-API major is not probed: nvidia-smi
        # --query-gpu has no such field, V3, so only the arch floor is gated).
        # A None compute capability is skipped rather than compared, to avoid
        # `None < tuple` crashing on an nvidia-smi parse miss.
        if cap.physical_gpus:
            for gpu in cap.physical_gpus:
                cc = gpu.compute_capability
                if cc is not None and cc < matrix_row.arch_floor:
                    raise RuntimeError(
                        f"GPU {gpu.name} sm_{cc[0]}{cc[1]} is below the wheel "
                        f"arch floor sm_{matrix_row.arch_floor[0]}"
                        f"{matrix_row.arch_floor[1]}"
                    )

    # Site-packages writable check
    try:
        import site

        sp = site.getsitepackages()
        if sp:
            test_path = os.path.join(sp[0], ".ort_provisioning_test")
            try:
                with open(test_path, "w") as f:
                    f.write("test")
                os.unlink(test_path)
            except OSError:
                raise RuntimeError(
                    f"Site-packages directory {sp[0]} is not writable"
                )
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        logger.warning("Could not verify site-packages writability: %s", exc)

    logger.info("Preflight passed for %s==%s", plan.package, plan.version)


def install_ort(plan: OrtWheelPlan, assume_yes: bool = False) -> None:
    """Install the resolved ORT wheel via pip.

    Unconditional floor: pinned version + single --index-url
    https://pypi.org/simple/ + --only-binary :all:.
    --require-hashes is conditional on transitive-satisfaction preflight (V1).

    NEVER actually runs pip during tests (mock the subprocess).
    """
    preflight(plan, detect())

    # Print plan details for explicit consent surface
    print(f"Package:    {plan.package}")
    print(f"Version:    {plan.version}")
    print(f"Index URL:  {plan.index_url}")
    print(f"Hashes:     {plan.hashes}")
    print(f"Provider:   {plan.provider_family}")

    if not assume_yes:
        print(
            "\nInstall requires explicit confirmation. "
            "Re-run with --install-ort --yes to proceed."
        )
        return

    # Build pip command. Unconditional floor.
    # --isolated ignores environment variables (PIP_EXTRA_INDEX_URL,
    # PIP_INDEX_URL, PIP_TRUSTED_HOST) and pip config files, so the single
    # official --index-url cannot be widened by an inherited env on
    # uncontrolled hardware (dependency confusion).
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--isolated",
        "--only-binary",
        ":all:",
        "--index-url",
        plan.index_url,
        f"{plan.package}=={plan.version}",
    ]

    # Conditional --require-hashes (V1): only when transitives are satisfied
    if _check_transitive_satisfied() and plan.hashes:
        cmd.append("--require-hashes")
        logger.info(
            "Transitives pre-satisfied; installing with --require-hashes"
        )
    else:
        logger.warning(
            "Hash-checked path unavailable: transitives not pre-satisfied. "
            "Installing on unconditional floor without --require-hashes."
        )

    print(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.returncode != 0:
            print(f"pip install failed (exit {result.returncode}):", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(1)
        print("Install successful.")
    except subprocess.TimeoutExpired:
        print("pip install timed out after 300s", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("pip not found", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    """CLI entry point for ort_provisioning."""
    parser = argparse.ArgumentParser(
        description="Resolve and optionally install the correct ORT wheel."
    )
    parser.add_argument(
        "--install-ort",
        action="store_true",
        help="Actually install the resolved wheel (requires --yes).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the install (must be combined with --install-ort).",
    )
    args = parser.parse_args()

    cap = detect()
    plan = resolve_ort_wheel(cap)

    print("Resolved plan:")
    print(f"  Package:    {plan.package}")
    print(f"  Version:    {plan.version}")
    print(f"  Index URL:  {plan.index_url}")
    print(f"  Provider:   {plan.provider_family}")
    if plan.hashes:
        print(f"  Hash ({plan.package}): {plan.hashes}")

    if args.install_ort:
        install_ort(plan, assume_yes=args.yes)
    else:
        print("\nNo install requested. Use --install-ort --yes to install.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
