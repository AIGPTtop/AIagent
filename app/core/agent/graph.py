# -*- coding: utf-8 -*-
"""LangGraph 编排：意图 -> 工具（写操作先人工审批）-> 结果 -> 回答。

状态机的价值不在于「看起来高级」，而在于三件事：
1. 工具失败可重试并带着错误继续往下走，不会让整个请求 500；
2. 写操作在 interrupt 处挂起，人工确认后从断点恢复，而不是重跑一遍；
3. 全流程状态可落 checkpoint，重启/换实例仍能续跑（当前用内存版）。
"""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from loguru import logger

from app.config import get_settings
from app.core.tools.base import BaseTool
from app.core.tools.builtin.hr import _parse_weekday, build_default_tools
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter

WRITE_TOOLS = {"submit_leave_request"}


class AgentState(TypedDict, total=False):
    """图状态：每个节点读一部分、写一部分。"""

    query: str
    employee_id: str
    intent: str
    tool_name: str | None
    tool_args: dict[str, Any]
    tool_result: dict[str, Any] | None
    tool_error: str | None
    attempts: int
    needs_approval: bool
    approval: dict[str, Any] | None
    answer: str


# --------------------------------------------------------------------- 工具注册
_TOOLS: dict[str, BaseTool] = {t.name: t for t in build_default_tools()}


def _get_tool(name: str) -> BaseTool | None:
    return _TOOLS.get(name)


# --------------------------------------------------------------------- LLM 调用
def _router() -> ModelRouter:
    settings = get_settings()
    cfg = ModelConfig(
        model_id=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base or None,
        priority=0,
        weight=1.0,
    )
    return ModelRouter([cfg])


async def _say(prompt: str) -> str:
    """一次性生成，用于把工具结果写成自然语言。"""
    resp = await _router().chat(
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )
    return str(resp.content).strip()


# --------------------------------------------------------------------- 节点
async def plan_node(state: AgentState) -> dict[str, Any]:
    """规则优先的意图路由；没命中工具就交给通用回答。"""
    query = state.get("query", "")
    text = query.strip()

    ticket_match = re.search(r"T-\d{4}", text, flags=re.IGNORECASE)
    if "工单" in text or ticket_match:
        return {
            "intent": "ticket_status",
            "tool_name": "get_ticket_status",
            "tool_args": {"ticket_id": ticket_match.group(0).upper() if ticket_match else "T-1001"},
            "needs_approval": False,
        }

    if re.search(r"(请|休).{0,8}假|调休|申请年假", text):
        target = _parse_weekday(text)
        days_match = re.search(r"(\d+)\s*天", text)
        days = int(days_match.group(1)) if days_match else 1
        return {
            "intent": "submit_leave",
            "tool_name": "submit_leave_request",
            "tool_args": {
                "employee_id": state.get("employee_id", "E1001"),
                "start_date": target.isoformat(),
                "days": days,
            },
            "needs_approval": True,
        }

    if any(k in text for k in ("年假", "还剩", "余额", "几天假", "剩余")):
        return {
            "intent": "leave_balance",
            "tool_name": "get_leave_balance",
            "tool_args": {"employee_id": state.get("employee_id", "E1001")},
            "needs_approval": False,
        }

    return {"intent": "chat", "tool_name": None, "needs_approval": False}


async def approval_node(state: AgentState) -> dict[str, Any]:
    """写操作挂起点：把动作详情抛给人，等 resume 决定。"""
    payload = {
        "tool": state.get("tool_name"),
        "args": state.get("tool_args", {}),
        "reason": "该操作会修改业务数据，需要人工确认",
    }
    logger.info("等待人工审批: {}", payload)
    decision = interrupt(payload)
    approved = bool((decision or {}).get("approved"))
    if not approved:
        return {
            "approval": {"approved": False},
            "tool_error": "用户取消了本次操作",
            "needs_approval": False,
        }
    return {"approval": {"approved": True}, "needs_approval": False}


async def act_node(state: AgentState) -> dict[str, Any]:
    """执行工具：失败重试 2 次，仍失败则带着错误进入回答节点。"""
    if state.get("tool_error"):
        return {"attempts": state.get("attempts", 0)}

    name = state.get("tool_name")
    tool = _get_tool(name) if name else None
    if tool is None:
        return {"tool_error": f"未注册的工具 {name}"}

    args = dict(state.get("tool_args") or {})
    last_error: str | None = None
    for attempt in range(1, 3):
        try:
            result = await tool.execute(**args)
            logger.info("工具 {} 第 {} 次执行成功", name, attempt)
            return {"tool_result": result, "attempts": attempt, "tool_error": None}
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("工具 {} 第 {} 次失败: {}", name, attempt, last_error)
            await asyncio.sleep(0.2 * attempt)
    return {"tool_error": last_error, "attempts": 2}


async def respond_node(state: AgentState) -> dict[str, Any]:
    """把工具结果润色成回答；失败或闲聊走兜底话术。"""
    if state.get("tool_error"):
        return {
            "answer": f"操作未完成：{state['tool_error']}。如果需要，我可以换一种方式帮你。"
        }

    result = state.get("tool_result")
    if not result:
        return {
            "answer": (
                "我可以帮你查询年假余额、工单状态，或者提交请假申请。"
                "例如：「我还有几天年假」「T-1001 现在什么状态」「帮我请下周五的假」。"
            )
        }

    prompt = (
        "你是企业 HR／客服助手。根据下面的工具执行结果，用中文简洁回答用户，"
        "保留关键数字与日期，不要编造结果里没有的信息。\n\n"
        f"用户问题：{state.get('query')}\n"
        f"工具执行结果：{result}\n"
        "回答控制在 120 字以内。"
    )
    answer = await _say(prompt)
    return {"answer": answer}


# --------------------------------------------------------------------- 组图
def _route_after_plan(state: AgentState) -> str:
    if not state.get("tool_name"):
        return "respond"
    if state.get("needs_approval"):
        return "approval"
    return "act"


def build_graph():
    """构建并编译状态机（内存 checkpoint）。"""
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("approval", approval_node)
    graph.add_node("act", act_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan",
        _route_after_plan,
        {"approval": "approval", "act": "act", "respond": "respond"},
    )
    graph.add_edge("approval", "act")
    graph.add_edge("act", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=MemorySaver())


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def _extract(result: dict[str, Any]) -> dict[str, Any]:
    """把图的输出整理成 API 返回结构。"""
    interrupts = result.get("__interrupt__") or []
    if interrupts:
        payload: Any = interrupts[0]
        value = getattr(payload, "value", payload)
        return {
            "status": "pending_approval",
            "approval": value,
            "tool": result.get("tool_name"),
            "args": result.get("tool_args"),
        }
    return {
        "status": "completed",
        "answer": result.get("answer", ""),
        "intent": result.get("intent"),
        "tool": result.get("tool_name"),
        "tool_result": result.get("tool_result"),
        "tool_error": result.get("tool_error"),
        "attempts": result.get("attempts", 0),
        "approval": result.get("approval"),
    }


async def run_agent(query: str, thread_id: str | None = None, employee_id: str = "E1001") -> dict:
    """执行一轮对话；写操作会停在审批处并返回 pending_approval。"""
    thread_id = thread_id or f"thread-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    result = await get_graph().ainvoke(
        {"query": query, "employee_id": employee_id, "attempts": 0},
        config=config,
    )
    out = _extract(result)
    out["thread_id"] = thread_id
    return out


async def resume_agent(thread_id: str, approved: bool) -> dict:
    """人工审批后从断点继续。"""
    config = {"configurable": {"thread_id": thread_id}}
    result = await get_graph().ainvoke(Command(resume={"approved": approved}), config=config)
    out = _extract(result)
    out["thread_id"] = thread_id
    return out
