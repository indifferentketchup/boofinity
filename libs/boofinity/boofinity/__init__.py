# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import importlib.metadata

from boofinity.args import EngineArgs  # noqa: E402
from boofinity.engine import AsyncEmbeddingEngine, AsyncEngineArray  # noqa: E402
from boofinity.env import MANAGER  # noqa: E402

# reexports
from boofinity.infinity_server import create_server  # noqa: E402
from boofinity.log_handler import logger  # noqa: E402
from boofinity.sync_engine import SyncEngineArray  # noqa: E402

__version__: str = importlib.metadata.version("boofinity")

__all__ = [
    "__version__",
    "AsyncEmbeddingEngine",
    "AsyncEngineArray",
    "create_server",
    "EngineArgs",
    "logger",
    "MANAGER",
    "SyncEngineArray",
]
