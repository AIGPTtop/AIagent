# -*- coding: utf-8 -*-
"""FastAPI 应用入口：生命周期内初始化数据库并预热模型。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.api.routes import agent, chat, document, health, memory, observability, search, ui
from app.config import get_settings
from app.core.warmup import warmup
from app.infrastructure.database.session import configure_session, init_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("启动 {} ({})", settings.app_name, settings.app_env)
    engine = init_engine(settings.database_url)
    configure_session(engine)
    app.state.engine = engine
    # 后台预热：不阻塞启动，但能让第一个真实请求避开 15–20 秒的模型加载
    app.state.warmup_task = asyncio.create_task(warmup())
    yield
    task = getattr(app.state, "warmup_task", None)
    if task and not task.done():
        task.cancel()
    await engine.dispose()
    logger.info("关闭 {}", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.include_router(health.router, prefix=settings.api_prefix)
    application.include_router(chat.router, prefix=settings.api_prefix)
    application.include_router(document.router, prefix=settings.api_prefix)
    application.include_router(search.router, prefix=settings.api_prefix)
    application.include_router(agent.router, prefix=settings.api_prefix)
    application.include_router(memory.router, prefix=settings.api_prefix)
    application.include_router(observability.router, prefix=settings.api_prefix)
    application.include_router(ui.router)
    return application


app = create_app()
