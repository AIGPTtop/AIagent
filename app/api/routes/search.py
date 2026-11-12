# -*- coding: utf-8 -*-
"""检索 API：向量 / BM25 / 混合（RRF），可选 Cross-Encoder 重排。"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rag.hybrid import get_retriever
from app.infrastructure.database.session import get_async_session

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    """检索请求。"""

    query: str = Field(min_length=1, description="查询文本")
    top_k: int = Field(default=5, ge=1, le=50, description="返回条数")
    mode: Literal["vector", "keyword", "hybrid"] = Field(
        default="hybrid",
        description="检索模式：vector=纯向量，keyword=BM25，hybrid=RRF 融合",
    )
    rerank: bool = Field(default=False, description="是否用 Cross-Encoder 重排")


class SearchHit(BaseModel):
    """单条命中。"""

    id: str
    document_id: str
    chunk_index: int
    score: float
    content: str
    ranks: dict[str, int] = Field(default_factory=dict, description="各路召回排名")
    retrieval_score: float | None = Field(default=None, description="重排前的召回分")
    reranked: bool = Field(default=False, description="是否经过重排")


class SearchResponse(BaseModel):
    """检索结果。"""

    query: str
    mode: str
    rerank: bool
    score_kind: str = Field(
        default="rrf",
        description="分数字段含义：cosine_ip（向量内积）| bm25 | rrf（倒数排名和）| rerank_logit（交叉编码器原始分）",
    )
    hits: list[SearchHit]


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    session: AsyncSession = Depends(get_async_session),
) -> SearchResponse:
    """执行检索；hybrid 模式先召回再可选重排。"""
    try:
        rows = await get_retriever().search(
            session,
            request.query,
            top_k=request.top_k,
            mode=request.mode,
            rerank=request.rerank,
        )
    except Exception as exc:
        logger.exception("检索失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"检索失败: {exc!s}") from exc

    hits = [
        SearchHit(
            id=str(r["id"]),
            document_id=str(r.get("document_id") or ""),
            chunk_index=int(r.get("chunk_index") or 0),
            score=round(float(r["score"]), 6),
            content=str(r.get("content") or ""),
            ranks={k: int(v) for k, v in (r.get("ranks") or {}).items()},
            retrieval_score=(
                round(float(r["retrieval_score"]), 6)
                if r.get("retrieval_score") is not None
                else None
            ),
            reranked=bool(r.get("reranked", False)),
        )
        for r in rows
    ]
    if request.rerank:
        score_kind = "rerank_logit"
    elif request.mode == "hybrid":
        score_kind = "rrf"
    elif request.mode == "keyword":
        score_kind = "bm25"
    else:
        score_kind = "cosine_ip"

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        rerank=request.rerank,
        score_kind=score_kind,
        hits=hits,
    )
