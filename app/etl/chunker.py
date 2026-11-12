# -*- coding: utf-8 -*-
"""文档分块：固定长度 / 递归 / 按段落 / Markdown 标题感知。"""

from __future__ import annotations

import re
from enum import Enum
from typing import List

import tiktoken
from loguru import logger


class ChunkStrategy(str, Enum):
    """分块策略枚举。"""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    PARAGRAPH = "paragraph"
    MARKDOWN = "markdown"


_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+)$")


class DocumentChunker:
    """文档分块器。

    默认 256 token、64 token 重叠。企业知识库问答里块太大是精度杀手：
    512 token 会把两三个章节揉进同一块，检索分数彼此接近、生成答案上下文串味。
    """

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = max(32, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            logger.warning("tiktoken 编码不可用，回退字符切分: {}", exc)
            self._encoding = None

    def _len(self, text: str) -> int:
        if self._encoding is None:
            return len(text)
        return len(self._encoding.encode(text))

    def chunk(self, text: str, strategy: ChunkStrategy = ChunkStrategy.MARKDOWN) -> List[str]:
        """按策略将全文切分为块列表。"""
        text = text.strip()
        if not text:
            return []

        if strategy == ChunkStrategy.FIXED:
            return self._chunk_fixed(text)
        if strategy == ChunkStrategy.PARAGRAPH:
            return self._chunk_paragraph(text)
        if strategy == ChunkStrategy.MARKDOWN:
            return self._chunk_markdown(text)
        return self._chunk_recursive(text, _depth=0)

    # ---------------------------------------------------------------- 策略
    def _chunk_fixed(self, text: str) -> List[str]:
        out: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + self.chunk_size)
            out.append(text[start:end])
            if end >= n:
                break
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        return out

    def _chunk_paragraph(self, text: str) -> List[str]:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        return self._merge(paras)

    def _chunk_markdown(self, text: str) -> List[str]:
        """按 Markdown 标题切段，每段内部再按长度切分。

        每块都保留所属标题（拼在正文前），这样即便块被切碎，
        模型仍知道这段属于哪一章，BM25 也能命中标题里的关键词。
        """
        sections: list[tuple[str, str]] = []
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return self._merge([p.strip() for p in text.split("\n\n") if p.strip()])

        # 第一个标题之前的内容（比如文档标题）
        if matches[0].start() > 0:
            prefix = text[: matches[0].start()].strip()
            if prefix:
                sections.append(("", prefix))

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((m.group(2).strip(), text[start:end].strip()))

        chunks: list[str] = []
        for heading, body in sections:
            if self._len(body) <= self.chunk_size:
                chunks.append(body)
                continue
            for piece in self._split_by_size(body):
                if heading and not piece.lstrip().startswith("#"):
                    chunks.append(f"## {heading}\n\n{piece}")
                else:
                    chunks.append(piece)
        return [c for c in chunks if c.strip()]

    def _chunk_recursive(self, text: str, _depth: int = 0) -> List[str]:
        """递归切分：分块仍超长时进一步按句号切。"""
        if self._len(text) <= self.chunk_size:
            return [text]
        if _depth >= 2:
            return self._split_by_size(text)
        parts = [p.strip() for p in re.split(r"(?<=[。！？；\n])", text) if p.strip()]
        merged = self._merge(parts, recursive=_depth + 1)
        return merged

    # ---------------------------------------------------------------- 工具
    def _split_by_size(self, text: str) -> List[str]:
        """按 token 预算硬切，带重叠，优先在换行处断开。"""
        out: list[str] = []
        buf = ""
        for line in text.splitlines(keepends=True):
            if self._len(buf + line) <= self.chunk_size:
                buf += line
                continue
            if buf:
                out.append(buf.strip())
            buf = line
            while self._len(buf) > self.chunk_size:
                # 超长单行：按字符比例截断
                keep = max(1, int(len(buf) * self.chunk_size / max(1, self._len(buf))))
                out.append(buf[:keep].strip())
                buf = buf[max(0, keep - self.chunk_overlap) :]
        if buf.strip():
            out.append(buf.strip())
        return out

    def _merge(self, parts: list[str], recursive: int = 0) -> List[str]:
        """把碎段落合并到目标长度，超长的递归再切。"""
        out: list[str] = []
        buf = ""
        for p in parts:
            candidate = (buf + "\n\n" + p).strip() if buf else p
            if self._len(candidate) <= self.chunk_size:
                buf = candidate
                continue
            if buf:
                out.append(buf)
            buf = p
            if self._len(buf) > self.chunk_size:
                if recursive:
                    out.extend(self._chunk_recursive(buf, recursive))
                else:
                    out.extend(self._split_by_size(buf))
                buf = ""
        if buf:
            out.append(buf)
        return [c for c in out if c.strip()]
