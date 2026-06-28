"""Context construction and prompt assembly.

This module handles building model input from conversation history.

Main components:
- ContextEngine: Core context engine
- ContextMessage: Internal message format
- TokenCounter: Token counting
- ContextCompressor: Context compression (extensible)

Usage:
    from agent_runtime.context import ContextEngine

    context = ContextEngine(store)
    context.add_user_message("conv-1", "Hello")
    model_input = context.build_model_input("conv-1")
"""

from .compression import CompressionResult, ContextCompressor, ModelContextCompressor
from .engine import ContextEngine, ContextMessage, ContextOverflowError
from .tokens import (
    ContextBudget,
    DEFAULT_COMPACT_THRESHOLD_RATIO,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_RESERVED_OUTPUT_TOKENS,
    DEFAULT_SAFETY_MARGIN_TOKENS,
    TokenCounter,
    TokenCounterProtocol,
)

__all__ = [
    "CompressionResult",
    "ContextBudget",
    "ContextCompressor",
    "ContextEngine",
    "ContextMessage",
    "ContextOverflowError",
    "ModelContextCompressor",
    "DEFAULT_COMPACT_THRESHOLD_RATIO",
    "DEFAULT_CONTEXT_WINDOW_TOKENS",
    "DEFAULT_RESERVED_OUTPUT_TOKENS",
    "DEFAULT_SAFETY_MARGIN_TOKENS",
    "TokenCounter",
    "TokenCounterProtocol",
]
