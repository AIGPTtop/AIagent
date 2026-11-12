# -*- coding: utf-8 -*-
"""启动预热：把 embedding 与 reranker 模型提前加载进显存，避免首个请求等 19 秒。"""

from __future__ import annotations

import asyncio

from loguru import logger

from app.config import get_settings


def _load_embedding() -> None:
    from app.infrastructure.llm.embedding import get_embedding_service

    service = get_embedding_service()
    service.encode(["预热文本"])


def _load_reranker() -> None:
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    model = CrossEncoder(settings.rerank_model_path, device=settings.rerank_device, max_length=512)
    model.predict([("预热查询", "预热文档内容")])


async def warmup() -> None:
    """并行加载两个模型；失败只记日志，不影响服务启动。"""
    try:
        await asyncio.gather(
            asyncio.to_thread(_load_embedding),
            asyncio.to_thread(_load_reranker),
        )
        logger.info("模型预热完成（embedding + reranker）")
    except Exception as exc:  # noqa: BLE001
        logger.warning("模型预热失败: {}", exc)
