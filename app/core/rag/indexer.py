# -*- coding: utf-8 -*-
"""文档索引：把分块结果向量化并写入 Milvus Lite，回填 vector_id。"""

from __future__ import annotations

from typing import Sequence

from loguru import logger

from app.infrastructure.llm.embedding import EmbeddingService, get_embedding_service
from app.infrastructure.vectordb.milvus_lite import MilvusLiteStore
from app.infrastructure.vectordb.store import get_vector_store


class DocumentIndexer:
    """把文档块写入向量库。

    设计要点：
    - 先按 document_id 删除旧向量，保证同一文档重复上传不产生脏数据；
    - 向量 id 复用 document_chunks.id，双向可追溯（DB -> Milvus / Milvus -> DB）；
    - 批量编码，避免逐条推理导致的 GPU 利用率低下。
    """

    def __init__(
        self,
        embedder: EmbeddingService | None = None,
        store: MilvusLiteStore | None = None,
    ) -> None:
        self._embedder = embedder or get_embedding_service()
        self._store = store or get_vector_store()

    async def index(
        self,
        document_id: str,
        chunks: Sequence[tuple[str, str]],
    ) -> list[str]:
        """chunks 为 [(chunk_id, content), ...]，返回写入的向量 id 列表。"""
        if not chunks:
            return []

        texts = [c[1] for c in chunks]
        vectors = await self._embedder.aencode(texts)

        await self._store.adelete_by_document(document_id)

        records = [
            {
                "id": chunk_id,
                "vector": vectors[i].tolist(),
                "document_id": document_id,
                "chunk_index": i,
                "content": texts[i],
            }
            for i, (chunk_id, _) in enumerate(chunks)
        ]
        written = await self._store.aupsert(records)
        logger.info(
            "索引完成 document_id={} chunks={} written={}",
            document_id,
            len(chunks),
            written,
        )
        return [r["id"] for r in records]
