# -*- coding: utf-8 -*-
"""对话 API：RAG 增强 + 分层记忆（Redis 短期 / 摘要压缩 / PG 归档）。"""

from __future__ import annotations

import json
import threading
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.intent.recognizer import IntentRecognizer
from app.core.memory.archive import archive_turn, conversation_tokens
from app.core.memory.store import count_tokens, get_short_term_memory
from app.core.rag.pipeline import get_rag_pipeline
from app.core.rag.stream import record_stream_trace, stream_rag
from app.infrastructure.database.session import get_async_session
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter
from app.infrastructure.trace.persist import save_span
from app.infrastructure.trace.tracer import Tracer
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

_tracer = Tracer()
_intent = IntentRecognizer()


def _build_router() -> ModelRouter:
    """根据配置构造模型路由器。"""
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="未配置 OPENAI_API_KEY，无法调用模型")
    cfg = ModelConfig(
        model_id=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base or None,
        priority=0,
        weight=1.0,
    )
    return ModelRouter([cfg])


async def _llm_call(messages: list[dict[str, str]]) -> Any:
    return await get_shared_router().chat(messages, temperature=0.2, max_tokens=300)


_ROUTER: ModelRouter | None = None
_ROUTER_LOCK = threading.Lock()


def get_shared_router() -> ModelRouter:
    """全局共享一个 ModelRouter，保证熔断计数跨请求累计。"""
    global _ROUTER
    if _ROUTER is None:
        with _ROUTER_LOCK:
            if _ROUTER is None:
                _ROUTER = _build_router()
    return _ROUTER


async def _persist_trace(
    session: AsyncSession,
    trace_id: str,
    span: Any,
    operation: str,
    payload: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    duration_ms = int((span.end_time - span.start_time) * 1000) if span.end_time else 0
    await save_span(
        session,
        trace_id=trace_id,
        span_id=span.span_id,
        operation=operation,
        duration_ms=duration_ms,
        payload=payload,
        error=error,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session: AsyncSession = Depends(get_async_session),
) -> ChatResponse:
    """对话入口：默认走 RAG（检索 + 引用 + 拒答）；带 conversation_id 时启用分层记忆。"""
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "chat")
    settings = get_settings()
    user_text = request.messages[-1].content if request.messages else ""
    cid = request.conversation_id
    mem = get_short_term_memory()

    try:
        intent = await _intent.recognize(user_text)
        clarify = await _intent.clarify(user_text, intent)
        logger.info(
            "意图 intent={} conf={} sub={}",
            intent.intent,
            intent.confidence,
            intent.sub_intent,
        )

        # ---------------- 组装上下文：摘要 + 历史 + 本轮 ----------------
        summary, history = ("", []) if not cid else await mem.load(cid)
        base = [m.model_dump() for m in request.messages]
        if history:
            history_msgs = [
                {"role": h.get("role", "user"), "content": h.get("content", "")}
                for h in history
                if h.get("role") in ("user", "assistant")
            ]
            base = history_msgs + base[-1:]

        full_history_tokens = sum(count_tokens(m["content"]) for m in base)
        prompt_messages: list[dict[str, str]] = []
        if summary:
            prompt_messages.append({"role": "system", "content": f"【会话摘要】{summary}"})
        prompt_messages.extend(base)
        prompt_tokens = sum(count_tokens(m["content"]) for m in prompt_messages)
        if cid:
            # 全量历史 = 已归档消息 + 本轮输入（不受 Redis 裁剪影响）
            full_history_tokens = await conversation_tokens(session, cid) + count_tokens(user_text)

        router_llm = get_shared_router()
        use_rag = settings.rag_enabled if request.use_rag is None else request.use_rag

        if use_rag:
            async def _call_llm(messages: list[dict[str, str]]):
                return await router_llm.chat(
                    messages,
                    model_preference=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )

            result = await get_rag_pipeline().answer(
                session,
                user_text,
                _call_llm,
                history=prompt_messages[:-1],
            )
            answer = result.answer
            response = ChatResponse(
                id=str(uuid.uuid4()),
                model=result.model or settings.openai_model,
                content=answer,
                trace_id=trace_id,
                usage=result.usage,
                citations=result.citations,
                refused=result.refused,
                retrieved=len(result.hits),
            )
        else:
            messages = list(prompt_messages)
            if clarify and intent.confidence < _intent.confidence_threshold:
                messages.append({"role": "system", "content": f"（系统提示：{clarify}）"})
            resp = await router_llm.chat(
                messages,
                model_preference=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            answer = resp.content
            response = ChatResponse(
                id=str(uuid.uuid4()),
                model=resp.model_id,
                content=answer,
                trace_id=trace_id,
                usage=resp.usage,
            )

        # ---------------- 记忆：写入短期 + 归档 + 触发摘要 ----------------
        if cid:
            await mem.append(cid, "user", user_text)
            await mem.append(cid, "assistant", answer)
            await archive_turn(session, cid, "user", user_text)
            await archive_turn(session, cid, "assistant", answer)
            if await mem.should_summarize(cid):
                await mem.summarize(cid, _llm_call)

            stats = await mem.stats(cid)
            response.memory = {
                "conversation_id": cid,
                "messages": stats["messages"],
                "summary_chars": stats["summary_chars"],
                "full_history_tokens": full_history_tokens,
                "prompt_tokens": prompt_tokens,
                "saved_tokens": max(0, full_history_tokens - prompt_tokens),
            }

        _tracer.end_span(
            span,
            result={"model": response.model, "usage": response.usage, "memory": cid is not None},
        )
        await _persist_trace(
            session,
            trace_id,
            span,
            "chat",
            payload={
                "model": response.model,
                "refused": response.refused,
                "citations": len(response.citations),
                "retrieved": response.retrieved,
                "usage": response.usage,
                "memory": cid is not None,
            },
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat 失败: {}", exc)
        _tracer.end_span(span, error=str(exc))
        await _persist_trace(session, trace_id, span, "chat", error=str(exc)[:2000])
        raise HTTPException(status_code=500, detail=f"对话失败: {exc!s}") from exc


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    session: AsyncSession = Depends(get_async_session),
) -> StreamingResponse:
    """流式输出（SSE）：先发检索结果与引用，再逐 token 输出答案。"""
    import time as _time

    trace_id = str(uuid.uuid4())
    started = _time.perf_counter()

    async def _gen():
        async for chunk in stream_rag(request, trace_id, session):
            yield chunk
        await record_stream_trace(
            trace_id,
            session,
            operation="chat_stream",
            duration_ms=int((_time.perf_counter() - started) * 1000),
            payload={"mode": "rag_stream"},
        )

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Trace-Id": trace_id,
        },
    )

