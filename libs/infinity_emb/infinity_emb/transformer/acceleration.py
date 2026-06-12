# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from infinity_emb._optional_imports import CHECK_TORCH

if CHECK_TORCH.is_available:
    import torch

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends, "cudnn"):
        # allow TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
