# -*- coding: utf-8 -*-
"""演示控制台：把单页 HTML 挂到根路径，避免用户直接面对 Swagger。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse

router = APIRouter(tags=["ui"], include_in_schema=False)

_INDEX = Path(__file__).resolve().parent.parent.parent / "static" / "index.html"


@router.get("/")
async def console() -> FileResponse:
    """返回可视化控制台。"""
    return FileResponse(_INDEX, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-store"})


@router.get("/ui")
async def console_alias() -> RedirectResponse:
    return RedirectResponse(url="/")

