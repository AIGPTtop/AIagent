# -*- coding: utf-8 -*-
"""可观测性 API：Trace 查询、延迟分位统计、熔断器状态。"""

from __future__ import annotations

import statistics
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from sqlalchemy import select

from app.api.routes.chat import get_shared_router
from app.infrastructure.database.models import TraceLog
from app.infrastructure.database.session import get_async_session
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(tags=["observability"])


def _percentile(sorted_values: list[int], pct: float) -> int:
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * pct))
    return sorted_values[idx]


@router.get("/trace/{trace_id}")
async def get_trace(
    trace_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """按 trace_id 查完整链路。"""
    rows = (
        await session.execute(
            select(TraceLog).where(TraceLog.trace_id == trace_id).order_by(TraceLog.created_at)
        )
    ).scalars().all()
    if not rows:
        raise HTTPException(status_code=404, detail="trace 不存在")
    return {
        "trace_id": trace_id,
        "spans": [
            {
                "span_id": r.span_id,
                "operation": r.operation,
                "duration_ms": r.duration_ms,
                "error": r.error,
                "payload": r.payload,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.get("/metrics/latency")
async def latency_metrics(
    operation: str = Query(default="chat", description="按 operation 过滤"),
    limit: int = Query(default=500, ge=10, le=5000),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """基于 trace_logs 计算延迟分位与错误率。"""
    rows = (
        await session.execute(
            select(TraceLog.duration_ms, TraceLog.error)
            .where(TraceLog.operation == operation)
            .order_by(TraceLog.created_at.desc())
            .limit(limit)
        )
    ).all()
    if not rows:
        return {"operation": operation, "count": 0}

    durations = sorted(int(r[0] or 0) for r in rows)
    errors = sum(1 for r in rows if r[1])
    return {
        "operation": operation,
        "count": len(durations),
        "avg_ms": round(statistics.mean(durations), 1),
        "p50_ms": _percentile(durations, 0.50),
        "p95_ms": _percentile(durations, 0.95),
        "p99_ms": _percentile(durations, 0.99),
        "max_ms": durations[-1],
        "errors": errors,
        "error_rate": round(errors / len(durations) * 100, 2),
    }


@router.get("/metrics/circuit")
async def circuit_states() -> dict[str, Any]:
    """查看各模型熔断器状态。"""
    try:
        router_llm = get_shared_router()
        return {"models": router_llm.breaker_states()}
    except Exception as exc:  # noqa: BLE001
        logger.exception("读取熔断器状态失败: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@router.get("/metrics/kb")
async def kb_metrics(session: AsyncSession = Depends(get_async_session)) -> dict[str, Any]:
    """知识库规模与关键配置，供控制台展示。"""
    from sqlalchemy import func

    from app.config import get_settings
    from app.infrastructure.database.models import Document, DocumentChunk

    docs = await session.scalar(select(func.count()).select_from(Document)) or 0
    chunks = await session.scalar(select(func.count()).select_from(DocumentChunk)) or 0
    settings = get_settings()
    return {
        "documents": int(docs),
        "chunks": int(chunks),
        "model": settings.openai_model,
        "refusal_threshold": settings.rag_min_rerank_score,
        "rag_enabled": settings.rag_enabled,
    }
