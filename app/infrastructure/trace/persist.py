# -*- coding: utf-8 -*-
"""Trace 持久化：把关键 Span 写入 PostgreSQL trace_logs 表。"""

from __future__ import annotations

import uuid
from typing import Any

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models import TraceLog


async def save_span(
    session: AsyncSession,
    *,
    trace_id: str,
    span_id: str,
    operation: str,
    duration_ms: int,
    payload: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """写一条 span；失败只记日志，不影响业务请求。"""
    try:
        session.add(
            TraceLog(
                id=str(uuid.uuid4()),
                trace_id=trace_id[:64],
                span_id=span_id[:64],
                operation=operation[:256],
                payload=payload,
                error=(error or None),
                duration_ms=duration_ms,
            )
        )
        await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("trace 持久化失败 trace_id={}: {}", trace_id, exc)
        await session.rollback()
