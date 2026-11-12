# -*- coding: utf-8 -*-
"""向量库工厂：按配置返回全局 MilvusLiteStore 单例。"""

from __future__ import annotations

import threading

from app.config import get_settings
from app.infrastructure.vectordb.milvus_lite import MilvusLiteStore

_store: MilvusLiteStore | None = None
_lock = threading.Lock()


def get_vector_store() -> MilvusLiteStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                settings = get_settings()
                _store = MilvusLiteStore(
                    uri=settings.milvus_uri,
                    collection=settings.milvus_collection_name,
                    dim=settings.embedding_dim,
                )
    return _store
