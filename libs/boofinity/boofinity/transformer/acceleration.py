# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from boofinity._optional_imports import CHECK_TORCH

if CHECK_TORCH.is_available:
    import torch

    if torch.cuda.is_available():
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends, "cudnn"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
