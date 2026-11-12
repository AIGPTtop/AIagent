# -*- coding: utf-8 -*-
"""对话归档：把每轮消息写入 PostgreSQL，供审计与统计。"""

from __future__ import annotations

import uuid

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.memory.store import count_tokens
from app.infrastructure.database.models import Conversation, Message as MessageRow

# 固定命名空间：同一字符串 cid 永远映射到同一 UUID
_NS = uuid.UUID("6f1b1d2e-0a3c-4c9a-9f6f-1f0b1c2d3e4f")


def to_uuid(raw: str) -> str:
    """把任意字符串会话 ID 映射成稳定 UUID。

    conversations/messages 的主键是 UUID 类型，而对外暴露的会话 ID 是字符串，
    这里用 uuid5 做确定性映射，保证重启后仍能对上同一条会话。
    """
    try:
        return str(uuid.UUID(str(raw)))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(_NS, str(raw)))


async def ensure_conversation(session: AsyncSession, conversation_id: str) -> None:
    """会话不存在则创建，避免外键失败。"""
    uid = to_uuid(conversation_id)
    exists = await session.execute(select(Conversation.id).where(Conversation.id == uid))
    if exists.scalar_one_or_none():
        return
    session.add(
        Conversation(
            id=uid,
            title="对话会话",
            meta={"source": "chat", "external_id": conversation_id},
        )
    )
    await session.commit()


async def archive_turn(
    session: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
) -> None:
    """写入一条消息；失败只记日志，不影响主流程。"""
    try:
        await ensure_conversation(session, conversation_id)
        session.add(
            MessageRow(
                conversation_id=to_uuid(conversation_id),
                role=role,
                content=content,
                token_count=count_tokens(content),
            )
        )
        await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("归档消息失败 cid={}: {}", conversation_id, exc)
        await session.rollback()


async def conversation_tokens(session: AsyncSession, conversation_id: str) -> int:
    """该会话已归档消息的 token 总量（用于对比「全量历史 vs 实际提示」）。"""
    try:
        total = await session.scalar(
            select(func.coalesce(func.sum(MessageRow.token_count), 0)).where(
                MessageRow.conversation_id == to_uuid(conversation_id)
            )
        )
        return int(total or 0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("统计会话 token 失败 cid={}: {}", conversation_id, exc)
        await session.rollback()
        return 0
