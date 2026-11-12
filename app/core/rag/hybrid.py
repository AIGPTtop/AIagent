# -*- coding: utf-8 -*-
"""混合检索：向量召回 + BM25 关键词召回 + RRF 融合。"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Literal

import jieba
from loguru import logger
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.infrastructure.database.models import DocumentChunk
from app.infrastructure.llm.embedding import get_embedding_service
from app.infrastructure.vectordb.store import get_vector_store

RetrievalMode = Literal["vector", "keyword", "hybrid"]

_STOPWORDS = {
    "的", "了", "和", "是", "在", "有", "我", "你", "他", "吗", "呢", "怎么", "什么",
    "如何", "多少", "几天", "一下", "请问", "可以", "需要", "这个", "那个", "以及",
}


def tokenize(text: str) -> list[str]:
    """中文检索分词：jieba 搜索引擎模式，去停用词与纯空白。"""
    tokens = [t.strip().lower() for t in jieba.cut_for_search(text or "")]
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


class HybridRetriever:
    """三路检索器。

    为什么不是纯向量：企业文档里大量查询带精确词（编号、金额、天数、专有名词），
    向量对这类 token 不敏感；BM25 正好互补。RRF 只依赖排名、不依赖分数尺度，
    免去了「向量 0.66 与 BM25 12.3」怎么加权这种拍脑袋问题。
    """

    def __init__(self, rrf_k: int = 60) -> None:
        self._embedder = get_embedding_service()
        self._store = get_vector_store()
        self._rrf_k = rrf_k
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._dirty = True
        self._reranker: Any = None

    # ---------------------------------------------------------------- 索引
    def invalidate(self) -> None:
        """文档增删后标记索引过期，下次检索时重建。"""
        self._dirty = True
        logger.info("BM25 索引已标记为待重建")

    async def refresh(self, session: AsyncSession) -> int:
        """从数据库重建 BM25 索引。"""
        async with self._lock:
            if not self._dirty and self._bm25 is not None:
                return len(self._chunks)
            rows = (
                await session.execute(
                    select(
                        DocumentChunk.id,
                        DocumentChunk.document_id,
                        DocumentChunk.chunk_index,
                        DocumentChunk.content,
                    )
                )
            ).all()
            self._chunks = [
                {
                    "id": str(r[0]),
                    "document_id": str(r[1]),
                    "chunk_index": int(r[2]),
                    "content": r[3] or "",
                }
                for r in rows
            ]
            if self._chunks:
                corpus = [tokenize(c["content"]) for c in self._chunks]
                self._bm25 = BM25Okapi(corpus)
            else:
                self._bm25 = None
            self._dirty = False
            logger.info("BM25 索引重建完成，chunks={}", len(self._chunks))
            return len(self._chunks)

    # ---------------------------------------------------------------- 检索
    async def _dense(self, query: str, top_k: int) -> list[dict[str, Any]]:
        vector = (await self._embedder.aencode([query]))[0].tolist()
        rows = await self._store.asearch(vector, top_k=top_k)
        return [
            {
                "id": str(r["id"]),
                "document_id": str(r.get("document_id") or ""),
                "chunk_index": int(r.get("chunk_index") or 0),
                "content": str(r.get("content") or ""),
                "score": float(r["distance"]),
            }
            for r in rows
        ]

    def _sparse(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self._bm25 is None or not self._chunks:
            return []
        terms = tokenize(query)
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out: list[dict[str, Any]] = []
        for i in order:
            if scores[i] <= 0:
                continue
            c = self._chunks[i]
            out.append({**c, "score": float(scores[i])})
        return out

    @staticmethod
    def _rrf_fuse(
        lists: list[list[dict[str, Any]]],
        top_k: int,
        k: int = 60,
        weights: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """加权 RRF：向量路权重略高于 BM25，避免关键词噪声把语义结果挤下去。"""
        fused: dict[str, dict[str, Any]] = {}
        for idx, lst in enumerate(lists):
            w = weights[idx] if weights and idx < len(weights) else 1.0
            for rank, item in enumerate(lst):
                rid = item["id"]
                entry = fused.setdefault(
                    rid,
                    {**item, "score": 0.0, "ranks": dict(item.get("ranks") or {})},
                )
                entry["score"] += w / (k + rank + 1)
        ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return ranked

    async def search(
        self,
        session: AsyncSession,
        query: str,
        top_k: int = 5,
        mode: RetrievalMode = "hybrid",
        dense_k: int | None = None,
        sparse_k: int | None = None,
        rerank: bool = False,
        rerank_candidates: int = 20,
    ) -> list[dict[str, Any]]:
        """执行检索，返回按相关度降序的块列表。

        rerank=True 时，先取候选集（默认 20 条），再用 Cross-Encoder 精排到 top_k。
        """
        settings = get_settings()
        dense_k = dense_k or settings.rag_top_k_dense
        sparse_k = sparse_k or settings.rag_top_k_sparse

        if mode == "vector":
            results = await self._dense(query, top_k if not rerank else rerank_candidates)
        elif mode == "keyword":
            await self.refresh(session)
            results = self._sparse(query, top_k if not rerank else rerank_candidates)
        else:
            await self.refresh(session)
            dense_res, sparse_res = await asyncio.gather(
                self._dense(query, dense_k),
                asyncio.to_thread(self._sparse, query, sparse_k),
                return_exceptions=True,
            )
            lists: list[list[dict[str, Any]]] = []
            if isinstance(dense_res, list):
                for rank, item in enumerate(dense_res):
                    item.setdefault("ranks", {})["vector"] = rank + 1
                lists.append(dense_res)
            else:
                logger.error("向量检索分支失败: {}", dense_res)
            if isinstance(sparse_res, list):
                for rank, item in enumerate(sparse_res):
                    item.setdefault("ranks", {})["bm25"] = rank + 1
                lists.append(sparse_res)
            else:
                logger.error("BM25 检索分支失败: {}", sparse_res)

            if not lists:
                return []
            if len(lists) == 1:
                results = lists[0][:top_k]
            else:
                weights = [1.0, 0.7][: len(lists)]
                fuse_k = max(top_k, rerank_candidates) if rerank else top_k
                results = self._rrf_fuse(lists, fuse_k, self._rrf_k, weights)

        if not rerank or not results:
            return results[:top_k]
        return await self._rerank(query, results, top_k)

    async def _rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Cross-Encoder 精排：把「召回」和「排序」拆成两步。"""
        from app.core.rag.reranker import Reranker
        from app.models.schemas import RetrievalResult

        if self._reranker is None:
            settings = get_settings()
            self._reranker = Reranker(
                model_name=settings.rerank_model_path,
                device=settings.rerank_device,
            )

        docs = [
            RetrievalResult(
                id=c["id"],
                content=c["content"],
                score=float(c.get("score", 0.0)),
                metadata={"ranks": c.get("ranks", {})},
                source="candidate",
            )
            for c in candidates
        ]
        ranked = await self._reranker.rerank(query, docs, top_k)
        by_id = {c["id"]: c for c in candidates}
        out: list[dict[str, Any]] = []
        for doc in ranked:
            base = dict(by_id[doc.id])
            base["retrieval_score"] = base.get("score", 0.0)
            base["score"] = float(doc.score)
            base["reranked"] = True
            out.append(base)
        return out


_retriever: HybridRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = HybridRetriever()
    return _retriever
