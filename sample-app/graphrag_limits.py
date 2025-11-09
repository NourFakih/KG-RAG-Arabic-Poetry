from __future__ import annotations

import os
from typing import Any, Mapping


DEFAULT_LIMITS: dict[str, Any] = {
    "best_model_max_async": 1,
    "cheap_model_max_async": 1,
    "embedding_func_max_async": 1,
    "embedding_batch_num": 8,
    "best_model_max_token_size": 4096,
    "cheap_model_max_token_size": 2048,
}

ENV_KEY_MAP: dict[str, tuple[str, type]] = {
    "GRAPH_RAG_BEST_MODEL_MAX_ASYNC": ("best_model_max_async", int),
    "GRAPH_RAG_CHEAP_MODEL_MAX_ASYNC": ("cheap_model_max_async", int),
    "GRAPH_RAG_EMBEDDING_MAX_ASYNC": ("embedding_func_max_async", int),
    "GRAPH_RAG_EMBEDDING_BATCH": ("embedding_batch_num", int),
    "GRAPH_RAG_BEST_MODEL_MAX_TOKENS": ("best_model_max_token_size", int),
    "GRAPH_RAG_CHEAP_MODEL_MAX_TOKENS": ("cheap_model_max_token_size", int),
}


def _parse(value: str, caster: type) -> Any:
    try:
        return caster(value)
    except Exception:
        return None


def get_graphrag_limits(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """
    Returns throttling kwargs for GraphRAG, honoring overrides from env variables.
    """
    source = env or os.environ
    limits = dict(DEFAULT_LIMITS)
    for env_key, (param_name, caster) in ENV_KEY_MAP.items():
        raw = source.get(env_key)
        if raw is None:
            continue
        parsed = _parse(raw, caster)
        if parsed is not None:
            limits[param_name] = parsed
    return limits
