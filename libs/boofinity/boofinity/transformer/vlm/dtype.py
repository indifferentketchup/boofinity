# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING

from boofinity.primitives import Device, Dtype

if TYPE_CHECKING:
    from boofinity.args import EngineArgs

import torch  # noqa: E402

_TORCH_DTYPE_MAP = {
    Dtype.float32: torch.float32,
    Dtype.float16: torch.float16,
    Dtype.bfloat16: torch.bfloat16,
}


def vlm_resolve_dtype(engine_args: EngineArgs, device_capability: tuple[int, int]) -> torch.dtype:
    if engine_args.device == Device.cuda and device_capability < (8, 0):
        return torch.float16
    if engine_args.dtype == Dtype.auto:
        if engine_args.device == Device.cuda and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if engine_args.device == Device.cpu:
            return torch.float32
        return torch.float16
    return _TORCH_DTYPE_MAP[engine_args.dtype]
