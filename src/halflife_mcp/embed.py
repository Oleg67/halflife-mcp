"""Optional local embeddings (the `[vector]` extra: fastembed, ONNX on CPU).

get_embedder() returns a callable list[str] -> list[list[float]] with a `.model` attribute,
or None when fastembed is not installed or HALFLIFE_EMBED=off. The model is downloaded
once into <data home>/models; later loads are offline.
"""

from __future__ import annotations

import os

MODEL = "BAAI/bge-small-en-v1.5"      # 384-d, ~65 MB; recall thresholds are calibrated for it


def get_embedder(model_dir: str):
    if os.environ.get("HALFLIFE_EMBED", "").lower() in ("off", "0", "false", "no"):
        return None
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return None
    model_name = os.environ.get("HALFLIFE_EMBED_MODEL", MODEL)
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never touch the network again
    os.makedirs(model_dir, exist_ok=True)
    model = TextEmbedding(model_name, cache_dir=model_dir)

    def embed(texts):
        return [[float(x) for x in v] for v in model.embed(list(texts))]

    embed.model = model_name
    return embed
