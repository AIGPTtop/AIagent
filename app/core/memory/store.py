# -*- coding: utf-8 -*-
"""分层记忆：Redis 短期上下文（TTL）+ LLM 摘要压缩 + PostgreSQL 归档。

为什么分三层：
- **短期**：最近 N 轮原样保留，保证指代与追问不失真；
- **摘要**：超过阈值后把旧消息压成一段摘要，控制 token 成本；
- **归档**：全量消息落 PG，供审计与统计，Redis 过期不影响事后追溯。
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import redis.asyncio as aioredis
import tiktoken
from loguru import logger

from app.config import get_settings

try:
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENC = None


def count_tokens(text: str) -> int:
    if _ENC is None:
        return max(1, len(text) // 2)
    return len(_ENC.encode(text or ""))


class ShortTermMemory:
    """Redis 中的短期记忆。

    key 设计：
      mem:{cid}:msgs   list，元素为 {"role","content","ts"}
      mem:{cid}:sum    string，滚动摘要
    TTL 对两者都生效；摘要 TTL 更长，避免对话间隔稍长就丢上下文。
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 1800,
        summary_ttl_seconds: int = 7200,
        max_messages: int = 20,
        summary_trigger: int = 12,
        summary_keep: int = 6,
    ) -> None:
        self._url = redis_url
        self._ttl = ttl_seconds
        self._summary_ttl = summary_ttl_seconds
        self._max_messages = max_messages
        self._summary_trigger = summary_trigger
        self._summary_keep = summary_keep
        self._client: aioredis.Redis | None = None

    async def _redis(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(self._url, decode_responses=True)
        return self._client

    @staticmethod
    def _key_msgs(cid: str) -> str:
        return f"mem:{cid}:msgs"

    @staticmethod
    def _key_sum(cid: str) -> str:
        return f"mem:{cid}:sum"

    # ---------------------------------------------------------------- 读写
    async def append(self, cid: str, role: str, content: str) -> None:
        r = await self._redis()
        item = json.dumps(
            {"role": role, "content": content, "ts": int(time.time())},
            ensure_ascii=False,
        )
        key = self._key_msgs(cid)
        await r.rpush(key, item)
        await r.ltrim(key, -self._max_messages, -1)
        await r.expire(key, self._ttl)

    async def load(self, cid: str) -> tuple[str, list[dict[str, Any]]]:
        """返回 (摘要, 最近消息列表)。"""
        r = await self._redis()
        summary = await r.get(self._key_sum(cid)) or ""
        raw = await r.lrange(self._key_msgs(cid), 0, -1)
        msgs: list[dict[str, Any]] = []
        for item in raw:
            try:
                msgs.append(json.loads(item))
            except json.JSONDecodeError:  # pragma: no cover
                continue
        return summary, msgs

    async def stats(self, cid: str) -> dict[str, Any]:
        r = await self._redis()
        key = self._key_msgs(cid)
        n = await r.llen(key)
        ttl = await r.ttl(key)
        summary = await r.get(self._key_sum(cid)) or ""
        return {
            "conversation_id": cid,
            "messages": n,
            "ttl_seconds": ttl,
            "summary_chars": len(summary),
            "summary": summary[:400],
        }

    # ---------------------------------------------------------------- 压缩
    async def should_summarize(self, cid: str) -> bool:
        r = await self._redis()
        return int(await r.llen(self._key_msgs(cid))) > self._summary_trigger

    async def summarize(
        self,
        cid: str,
        llm: Any,
    ) -> str:
        """把最旧的一半消息压进摘要，只保留最近 summary_keep 条原文。"""
        r = await self._redis()
        key = self._key_msgs(cid)
        raw = await r.lrange(key, 0, -1)
        if len(raw) <= self._summary_keep:
            return ""

        cut = len(raw) - self._summary_keep
        old_msgs = []
        for item in raw[:cut]:
            try:
                old_msgs.append(json.loads(item))
            except json.JSONDecodeError:  # pragma: no cover
                continue

        prev_summary = await r.get(self._key_sum(cid)) or ""
        dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in old_msgs)
        prompt = (
            "把下面的多轮对话压缩成不超过 150 字的中文摘要，保留：用户身份/诉求、"
            "已确认的关键事实（数字、日期、单号）、未完成的事项。不要编造。\n\n"
            f"已有摘要：{prev_summary or '（无）'}\n\n新增对话：\n{dialogue}"
        )
        resp = await llm(
            [
                {"role": "system", "content": "你是对话摘要器，只输出摘要正文。"},
                {"role": "user", "content": prompt},
            ]
        )
        summary = str(getattr(resp, "content", "") or "").strip()
        if not summary:
            return prev_summary

        await r.set(self._key_sum(cid), summary, ex=self._summary_ttl)
        await r.ltrim(key, -self._summary_keep, -1)
        await r.expire(key, self._ttl)
        logger.info(
            "对话 {} 摘要压缩：{} 条 -> 保留 {} 条，摘要 {} 字",
            cid,
            cut,
            self._summary_keep,
            len(summary),
        )
        return summary


_mem: ShortTermMemory | None = None
_lock = threading.Lock()


def get_short_term_memory() -> ShortTermMemory:
    global _mem
    if _mem is None:
        with _lock:
            if _mem is None:
                s = get_settings()
                _mem = ShortTermMemory(
                    redis_url=s.redis_url,
                    ttl_seconds=s.memory_ttl_seconds,
                    summary_ttl_seconds=s.memory_summary_ttl_seconds,
                    max_messages=s.memory_max_messages,
                    summary_trigger=s.memory_summary_trigger,
                    summary_keep=s.memory_summary_keep,
                )
    return _mem
