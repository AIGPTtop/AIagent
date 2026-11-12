# -*- coding: utf-8 -*-
"""RAG 编排：检索 -> 证据阈值判定 -> 组装提示 -> 生成 -> 抽取引用。"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.rag.hybrid import get_retriever

REFUSAL_TEXT = (
    "抱歉，知识库中没有检索到与该问题足够相关的资料，我无法据此作答。"
    "建议把问题描述得更具体一些，或直接联系 HR／IT 确认。"
)

_SYSTEM_PROMPT = """你是企业内部的制度与业务知识助手。请严格遵守以下规则：
1. 只依据【参考资料】作答，不得引入资料之外的知识，也不得编造数字、日期或人名。
2. 引用资料时在句末标注编号，例如 [1]、[2]；同一句可标注多个。
3. 如果参考资料不能回答问题，直接说明「根据现有资料无法回答」，不要猜测。
4. 回答用中文，先给结论，再给必要细节，控制在 300 字以内。"""

_CITE_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class RagResult:
    """一次 RAG 问答的结果。"""

    answer: str
    refused: bool
    citations: list[dict[str, Any]] = field(default_factory=list)
    hits: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None
    usage: dict[str, Any] | None = None


class RAGPipeline:
    """把检索与生成串起来的编排层。

    关键设计：**证据不足时明确拒答**。企业场景里「编一个看起来合理的答案」
    比「说不知道」危害大得多，因此用重排分数的绝对阈值做门禁。
    """

    def __init__(self) -> None:
        self._retriever = get_retriever()

    # ---------------------------------------------------------------- 检索
    async def retrieve(
        self,
        session: AsyncSession,
        query: str,
    ) -> tuple[list[dict[str, Any]], bool]:
        """返回 (命中列表, 是否应拒答)。"""
        settings = get_settings()
        hits = await self._retriever.search(
            session,
            query,
            top_k=settings.rag_top_k_final,
            mode="hybrid",
            rerank=True,
            rerank_candidates=settings.rag_rerank_candidates,
        )
        if not hits:
            return [], True
        best = float(hits[0].get("score", 0.0))
        if best < settings.rag_min_rerank_score:
            logger.info(
                "证据不足拒答：best_score={} 阈值={}",
                round(best, 4),
                settings.rag_min_rerank_score,
            )
            return hits, True
        return hits, False

    # ---------------------------------------------------------------- 提示
    def build_messages(
        self,
        query: str,
        hits: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        """把命中的块编号后拼进提示，编号与 citations 一一对应。"""
        settings = get_settings()
        lines: list[str] = []
        budget = settings.rag_context_max_chars
        used = 0
        for i, h in enumerate(hits, start=1):
            text = str(h.get("content") or "").strip()
            if used + len(text) > budget:
                text = text[: max(0, budget - used)]
            if not text:
                continue
            used += len(text)
            lines.append(f"[{i}] {text}")
            if used >= budget:
                break

        history_text = ""
        if history:
            recent = [m for m in history if m.get("role") in ("user", "assistant")][-4:]
            history_text = "\n".join(
                f"{m['role']}: {str(m.get('content', ''))[:200]}" for m in recent
            )

        user_content = (
            f"【参考资料】\n{chr(10).join(lines) if lines else '（无）'}\n\n"
            f"【最近对话】\n{history_text or '（无）'}\n\n"
            f"【问题】\n{query}"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # ---------------------------------------------------------------- 引用
    @staticmethod
    def extract_citations(answer: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从答案里解析 [n]，映射回命中块；忽略越界编号与重复。"""
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for raw in _CITE_RE.findall(answer or ""):
            idx = int(raw)
            if idx in seen or idx < 1 or idx > len(hits):
                continue
            seen.add(idx)
            h = hits[idx - 1]
            content = str(h.get("content") or "")
            out.append(
                {
                    "index": idx,
                    "result_id": str(h.get("id")),
                    "document_id": str(h.get("document_id") or ""),
                    "chunk_index": int(h.get("chunk_index") or 0),
                    "score": round(float(h.get("score", 0.0)), 4),
                    "snippet": content[:160] + ("…" if len(content) > 160 else ""),
                }
            )
        return out

    # ---------------------------------------------------------------- 主流程
    async def answer(
        self,
        session: AsyncSession,
        query: str,
        llm: Callable[[list[dict[str, str]]], Awaitable[Any]],
        history: list[dict[str, Any]] | None = None,
    ) -> RagResult:
        """完整链路：检索 -> 门禁 -> 生成 -> 引用。"""
        hits, refuse = await self.retrieve(session, query)
        if refuse:
            return RagResult(answer=REFUSAL_TEXT, refused=True, hits=hits[:3])

        messages = self.build_messages(query, hits, history)
        try:
            resp = await llm(messages)
        except Exception as exc:  # pragma: no cover - 交给上层处理
            logger.exception("RAG 生成失败: {}", exc)
            raise

        answer = str(getattr(resp, "content", "") or "").strip()
        citations = self.extract_citations(answer, hits)
        return RagResult(
            answer=answer,
            refused=False,
            citations=citations,
            hits=hits,
            model=getattr(resp, "model_id", None),
            usage=getattr(resp, "usage", None),
        )


_pipeline: RAGPipeline | None = None
_lock = threading.Lock()


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                _pipeline = RAGPipeline()
    return _pipeline
