# -*- coding: utf-8 -*-
"""Embedding 服务：基于 sentence-transformers 加载 bge-m3，支持 GPU 批处理与异步调用。"""

from __future__ import annotations

import asyncio
import threading
from typing import Sequence

import numpy as np
from loguru import logger

from app.config import get_settings


class EmbeddingService:
    """向量化服务（进程内单例，模型懒加载）。

    bge-m3 输出 1024 维稠密向量；开启 normalize_embeddings 后
    向量已 L2 归一化，Milvus 里用 IP 即等价于余弦相似度。
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None
        self._lock = threading.Lock()
        self._dim: int | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            logger.info(
                "加载 embedding 模型 path={} device={}",
                self._model_path,
                self._device,
            )
            model = SentenceTransformer(self._model_path, device=self._device)
            model.max_seq_length = self._max_length
            self._dim = int(model.get_sentence_embedding_dimension())
            self._model = model
            logger.info("embedding 模型就绪 dim={}", self._dim)

    @property
    def dim(self) -> int:
        self._ensure_model()
        assert self._dim is not None
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """同步批量编码，返回 (n, dim) 的 float32 归一化向量。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_model()
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    async def aencode(self, texts: Sequence[str]) -> np.ndarray:
        """异步包装：把阻塞的模型推理放到线程池，避免卡事件循环。"""
        return await asyncio.to_thread(self.encode, texts)


_service: EmbeddingService | None = None
_service_lock = threading.Lock()


def get_embedding_service() -> EmbeddingService:
    """按配置返回全局 EmbeddingService。"""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                settings = get_settings()
                _service = EmbeddingService(
                    model_path=settings.embedding_model_path,
                    device=settings.embedding_device,
                    batch_size=settings.embedding_batch_size,
                    max_length=settings.embedding_max_length,
                )
    return _service
