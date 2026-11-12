# -*- coding: utf-8 -*-
"""文档管理 API：上传、向量化入库与列表。"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rag.hybrid import get_retriever
from app.core.rag.indexer import DocumentIndexer
from app.etl import ETLPipeline
from app.etl.chunker import ChunkStrategy
from app.infrastructure.database.models import Document, DocumentChunk
from app.infrastructure.database.session import get_async_session
from app.models.schemas import DocumentInfo, DocumentUploadResponse

router = APIRouter(tags=["documents"])


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="上传的文件"),
    session: AsyncSession = Depends(get_async_session),
) -> DocumentUploadResponse:
    """上传文档：解析 -> 标题感知分块 -> 向量化入库 -> 写数据库。"""
    upload_root = Path("uploads")
    upload_root.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    safe_name = file.filename or "unnamed"
    dest = upload_root / f"{doc_id}_{safe_name}"

    try:
        raw = await file.read()
        await asyncio.to_thread(dest.write_bytes, raw)
    except Exception as exc:
        logger.exception("保存上传文件失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"保存文件失败: {exc!s}") from exc

    pipeline = ETLPipeline()
    try:
        etl = await pipeline.run_bytes(
            raw,
            filename=safe_name,
            mime_type=file.content_type,
            strategy=ChunkStrategy.MARKDOWN,
        )
    except Exception as exc:
        logger.exception("ETL 失败: {}", exc)
        raise HTTPException(status_code=422, detail=f"文档解析失败: {exc!s}") from exc

    doc = Document(
        id=doc_id,
        filename=safe_name,
        mime_type=file.content_type,
        storage_path=str(dest),
        status="indexing",
        meta={"chunk_count": len(etl.chunks)},
    )
    session.add(doc)

    chunk_ids: list[str] = []
    chunk_rows: list[DocumentChunk] = []
    for i, chunk_text in enumerate(etl.chunks):
        chunk_id = str(uuid.uuid4())
        chunk_ids.append(chunk_id)
        chunk = DocumentChunk(
            id=chunk_id,
            document_id=doc_id,
            chunk_index=i,
            content=chunk_text[:65000],
            vector_id=None,
            meta=None,
        )
        chunk_rows.append(chunk)
        session.add(chunk)
    await session.commit()

    # ---- 向量化并写入 Milvus Lite ----
    try:
        indexer = DocumentIndexer()
        pairs = [(cid, row.content) for cid, row in zip(chunk_ids, chunk_rows)]
        vector_ids = await indexer.index(doc_id, pairs)
    except Exception as exc:
        logger.exception("向量化失败 document_id={}: {}", doc_id, exc)
        await session.execute(
            update(Document).where(Document.id == doc_id).values(status="index_failed")
        )
        await session.commit()
        raise HTTPException(status_code=500, detail=f"向量化入库失败: {exc!s}") from exc

    for cid in vector_ids:
        await session.execute(
            update(DocumentChunk)
            .where(DocumentChunk.id == cid)
            .values(vector_id=cid)
        )
    await session.execute(
        update(Document).where(Document.id == doc_id).values(status="ready")
    )
    await session.commit()

    # 新文档入库后，BM25 索引需要重建
    get_retriever().invalidate()

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=safe_name,
        status="ready",
        chunk_count=len(etl.chunks),
        message="上传、分块并向量化成功",
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(
    session: AsyncSession = Depends(get_async_session),
) -> list[DocumentInfo]:
    """列出已入库文档元数据。"""
    try:
        result = await session.execute(select(Document).order_by(Document.created_at.desc()))
        rows = result.scalars().all()
        out: list[DocumentInfo] = []
        for d in rows:
            out.append(
                DocumentInfo(
                    id=d.id,
                    filename=d.filename,
                    mime_type=d.mime_type,
                    status=d.status,
                    created_at=d.created_at.isoformat() if d.created_at else None,
                )
            )
        return out
    except Exception as exc:
        logger.exception("查询文档列表失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"查询失败: {exc!s}") from exc
