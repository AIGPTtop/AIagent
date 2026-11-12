# -*- coding: utf-8 -*-
"""Milvus Lite 向量库封装：本地文件模式，无需独立服务进程。

与 Milvus standalone 使用同一套 pymilvus 客户端语义，区别仅在
连接目标是本地 .db 文件（Milvus Lite），适合单机开发与演示。
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

from loguru import logger

try:  # pragma: no cover
    from pymilvus import MilvusClient
except ImportError:  # pragma: no cover
    MilvusClient = Any  # type: ignore[misc, assignment]


class MilvusLiteStore:
    """基于 Milvus Lite 的向量存储。

    集合 schema（显式指定，避免 auto schema 带来的字段类型漂移）：
      id          VARCHAR 主键，等于 document_chunks.id
      vector      FLOAT_VECTOR(dim)
      document_id VARCHAR
      chunk_index INT64
      content     VARCHAR
    """

    def __init__(
        self,
        uri: str,
        collection: str,
        dim: int,
        content_max_length: int = 65535,
    ) -> None:
        self._uri = uri
        self._collection = collection
        self._dim = dim
        self._content_max_length = content_max_length
        self._client: MilvusClient | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 内部
    def _ensure_client(self) -> MilvusClient:
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            uri_path = Path(self._uri)
            uri_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("连接 Milvus Lite uri={}", self._uri)
            client = MilvusClient(uri=self._uri)
            if not client.has_collection(self._collection):
                logger.info(
                    "创建集合 {} dim={} metric=IP",
                    self._collection,
                    self._dim,
                )
                client.create_collection(
                    collection_name=self._collection,
                    dimension=self._dim,
                    primary_field_name="id",
                    id_type="string",
                    max_length=64,
                    vector_field_name="vector",
                    metric_type="IP",
                    auto_id=False,
                )
            # Milvus Lite 的集合默认处于 released 状态，检索前必须 load 进内存
            client.load_collection(self._collection)
            self._client = client
            return client

    # ---------------------------------------------------------------- 写入
    def upsert(self, records: list[dict[str, Any]]) -> int:
        """写入/覆盖向量记录，返回写入条数。"""
        if not records:
            return 0
        client = self._ensure_client()
        payload: list[dict[str, Any]] = []
        for r in records:
            payload.append(
                {
                    "id": str(r["id"]),
                    "vector": [float(x) for x in r["vector"]],
                    "document_id": str(r.get("document_id", "")),
                    "chunk_index": int(r.get("chunk_index", 0)),
                    "content": str(r.get("content", ""))[: self._content_max_length],
                }
            )
        client.upsert(collection_name=self._collection, data=payload)
        return len(payload)

    async def aupsert(self, records: list[dict[str, Any]]) -> int:
        return await asyncio.to_thread(self.upsert, records)

    # ---------------------------------------------------------------- 检索
    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索，返回 [{id, distance, document_id, chunk_index, content}]。"""
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "collection_name": self._collection,
            "data": [query_vector],
            "limit": top_k,
            "output_fields": ["document_id", "chunk_index", "content"],
        }
        if expr:
            kwargs["filter"] = expr
        res = client.search(**kwargs)
        hits = res[0] if res else []
        return [
            {
                "id": h.get("id"),
                "distance": float(h.get("distance", 0.0)),
                "document_id": h.get("entity", {}).get("document_id"),
                "chunk_index": h.get("entity", {}).get("chunk_index"),
                "content": h.get("entity", {}).get("content", ""),
            }
            for h in hits
        ]

    async def asearch(
        self,
        query_vector: list[float],
        top_k: int = 20,
        expr: str | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.search, query_vector, top_k, expr)

    # ---------------------------------------------------------------- 维护
    def delete_by_document(self, document_id: str) -> None:
        """删除某文档的全部向量（重传文档时先清理）。"""
        client = self._ensure_client()
        client.delete(
            collection_name=self._collection,
            filter=f'document_id == "{document_id}"',
        )

    async def adelete_by_document(self, document_id: str) -> None:
        await asyncio.to_thread(self.delete_by_document, document_id)

    def count(self) -> int:
        """集合内实体数，用于校验入库是否完整。"""
        client = self._ensure_client()
        stats = client.get_collection_stats(self._collection)
        return int(stats.get("row_count", 0))

    async def acount(self) -> int:
        return await asyncio.to_thread(self.count)
