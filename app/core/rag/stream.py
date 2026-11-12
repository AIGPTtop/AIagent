# -*- coding: utf-8 -*-
"""流式 RAG 生成器：检索完成先发引用，再逐 token 输出答案。"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from loguru import logger
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.memory.archive import archive_turn, conversation_tokens
from app.core.memory.store import count_tokens, get_short_term_memory
from app.core.rag.pipeline import REFUSAL_TEXT, get_rag_pipeline
from app.infrastructure.trace.persist import save_span
from app.models.schemas import ChatRequest


def _sse(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"


async def stream_rag(
    request: ChatRequest,
    trace_id: str,
    session: AsyncSession,
) -> AsyncIterator[bytes]:
    """SSE 事件序列：

    retrieval  → 检索结果与引用元数据（先让前端知道引了哪些段落）
    delta      → 逐 token 的答案增量
    refusal    → 证据不足，直接返回固定拒答文案（不调模型）
    done       → 结束事件，含 citations 与用量
    error      → 异常
    """
    settings = get_settings()
    user_text = request.messages[-1].content if request.messages else ""
    cid = request.conversation_id
    mem = get_short_term_memory()
    use_rag = settings.rag_enabled if request.use_rag is None else request.use_rag

    pipeline = get_rag_pipeline()
    hits: list[dict[str, Any]] = []

    try:
        # ---------------- 上下文与检索 ----------------
        summary, history = ("", []) if not cid else await mem.load(cid)
        history_msgs = [
            {"role": h.get("role", "user"), "content": h.get("content", "")}
            for h in history
            if h.get("role") in ("user", "assistant")
        ]

        if use_rag:
            hits, refuse = await pipeline.retrieve(session, user_text)
            yield _sse(
                {
                    "type": "retrieval",
                    "refused": refuse,
                    "hits": [
                        {
                            "index": i + 1,
                            "id": h.get("id"),
                            "document_id": h.get("document_id"),
                            "score": round(float(h.get("score", 0.0)), 4),
                            "snippet": str(h.get("content", ""))[:120],
                        }
                        for i, h in enumerate(hits)
                    ],
                }
            )
            if refuse:
                yield _sse({"type": "refusal", "content": REFUSAL_TEXT})
                yield _sse({"type": "done", "citations": [], "usage": None})
                return
            messages = pipeline.build_messages(user_text, hits, history_msgs)
        else:
            messages = [m.model_dump() for m in request.messages]
            if summary:
                messages.insert(0, {"role": "system", "content": f"【会话摘要】{summary}"})

        # ---------------- 流式生成 ----------------
        client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base or None,
        )
        stream = await client.chat.completions.create(
            model=request.model or settings.openai_model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )

        parts: list[str] = []
        first_token_at: float | None = None
        import time as _time

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if not delta:
                continue
            if first_token_at is None:
                first_token_at = _time.perf_counter()
            parts.append(delta)
            yield _sse({"type": "delta", "content": delta})

        answer = "".join(parts).strip()
        citations = pipeline.extract_citations(answer, hits) if use_rag else []
        yield _sse({"type": "done", "citations": citations, "usage": None})

        # ---------------- 记忆与归档 ----------------
        if cid:
            await mem.append(cid, "user", user_text)
            await mem.append(cid, "assistant", answer)
            await archive_turn(session, cid, "user", user_text)
            await archive_turn(session, cid, "assistant", answer)
            if await mem.should_summarize(cid):
                from app.api.routes.chat import _llm_call

                await mem.summarize(cid, _llm_call)

    except Exception as exc:  # noqa: BLE001
        logger.exception("chat_stream 失败: {}", exc)
        yield _sse({"type": "error", "message": str(exc)})


async def record_stream_trace(
    trace_id: str,
    session: AsyncSession,
    *,
    operation: str,
    duration_ms: int,
    payload: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    await save_span(
        session,
        trace_id=trace_id,
        span_id=str(uuid.uuid4()),
        operation=operation,
        duration_ms=duration_ms,
        payload=payload,
        error=error,
    )
