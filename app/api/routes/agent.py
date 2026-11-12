# -*- coding: utf-8 -*-
"""Agent API：多步工具调用 + 人工审批恢复。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.core.agent.graph import resume_agent, run_agent

router = APIRouter(tags=["agent"])


class AgentChatRequest(BaseModel):
    """一轮 Agent 对话。"""

    message: str = Field(min_length=1, description="用户输入")
    thread_id: str | None = Field(default=None, description="会话线程 ID；不传则新建")
    employee_id: str = Field(default="E1001", description="员工工号（演示用）")


class AgentResumeRequest(BaseModel):
    """人工审批结果。"""

    thread_id: str = Field(description="挂起的线程 ID")
    approved: bool = Field(description="是否批准执行")


class AgentResponse(BaseModel):
    """Agent 响应：完成或待审批。"""

    status: str
    thread_id: str
    answer: str | None = None
    intent: str | None = None
    tool: str | None = None
    tool_result: dict[str, Any] | None = None
    tool_error: str | None = None
    approval: dict[str, Any] | None = None
    attempts: int = 0


@router.post("/agent/chat", response_model=AgentResponse)
async def agent_chat(request: AgentChatRequest) -> AgentResponse:
    """走 LangGraph 状态机；写操作会返回 pending_approval。"""
    try:
        out = await run_agent(request.message, request.thread_id, request.employee_id)
    except Exception as exc:
        logger.exception("agent 执行失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {exc!s}") from exc
    return AgentResponse(**out)


@router.post("/agent/resume", response_model=AgentResponse)
async def agent_resume(request: AgentResumeRequest) -> AgentResponse:
    """按人工审批结论从断点恢复执行。"""
    try:
        out = await resume_agent(request.thread_id, request.approved)
    except Exception as exc:
        logger.exception("agent 恢复失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"恢复失败: {exc!s}") from exc
    return AgentResponse(**out)
