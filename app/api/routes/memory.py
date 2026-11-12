# -*- coding: utf-8 -*-
"""记忆 API：查看某会话的短期记忆与摘要状态。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.core.memory.store import get_short_term_memory

router = APIRouter(tags=["memory"])


@router.get("/memory/{conversation_id}")
async def memory_stats(conversation_id: str) -> dict:
    """返回消息条数、剩余 TTL、摘要长度与摘要内容。"""
    try:
        return await get_short_term_memory().stats(conversation_id)
    except Exception as exc:
        logger.exception("读取记忆失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"读取记忆失败: {exc!s}") from exc
