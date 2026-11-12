# -*- coding: utf-8 -*-
"""模拟 HR / 工单工具：演示 Agent 工具调用、写操作审批与幂等。

数据放在进程内存里，重启即复位——这是演示骨架，接真实 HR 系统时
只需替换 store 的实现，工具接口保持不变。
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Any

from app.core.tools.base import BaseTool, ToolParameter

# --------------------------------------------------------------------- 模拟数据
EMPLOYEES: dict[str, dict[str, Any]] = {
    "E1001": {
        "name": "张三",
        "department": "研发中心",
        "annual_leave_total": 10,
        "annual_leave_used": 3,
        "carry_over": 2,
    },
    "E1002": {
        "name": "李四",
        "department": "客户成功部",
        "annual_leave_total": 5,
        "annual_leave_used": 5,
        "carry_over": 0,
    },
}

TICKETS: dict[str, dict[str, Any]] = {
    "T-1001": {
        "title": "报表导出超时",
        "priority": "P1",
        "status": "处理中",
        "owner": "王工",
        "updated_at": "2026-11-10 18:20",
        "sla_left_hours": 5,
    },
    "T-1002": {
        "title": "登录验证码收不到",
        "priority": "P2",
        "status": "待客户确认",
        "owner": "陈工",
        "updated_at": "2026-11-11 09:05",
        "sla_left_hours": 20,
    },
}

LEAVE_REQUESTS: dict[str, dict[str, Any]] = {}
_IDEMPOTENCY: dict[str, str] = {}


def _parse_weekday(text: str) -> date:
    """把「下周五」「本周三」「明天」这类口语映射成日期。"""
    today = date.today()
    weekdays = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6}
    if "明天" in text:
        return today + timedelta(days=1)
    if "今天" in text:
        return today
    target = None
    for ch, idx in weekdays.items():
        if ch in text:
            target = idx
            break
    if target is None:
        return today + timedelta(days=7)
    base = today - timedelta(days=today.weekday())
    if "下" in text:
        base += timedelta(days=7)
    elif "本周" not in text and "这周" not in text:
        # 未指定周次：取下一个该星期几
        candidate = base + timedelta(days=target)
        if candidate <= today:
            candidate += timedelta(days=7)
        return candidate
    return base + timedelta(days=target)


class LeaveBalanceTool(BaseTool):
    """查询员工年假余额。"""

    name = "get_leave_balance"
    description = "查询员工当年年假总额、已用、剩余与结转天数"

    def __init__(self) -> None:
        super().__init__()
        self.parameters = [
            ToolParameter(name="employee_id", description="员工工号，如 E1001"),
        ]

    async def execute(self, **kwargs: Any) -> Any:
        employee_id = str(kwargs.get("employee_id", "E1001"))
        emp = EMPLOYEES.get(employee_id)
        if not emp:
            raise ValueError(f"未找到员工 {employee_id}")
        remaining = emp["annual_leave_total"] - emp["annual_leave_used"] + emp["carry_over"]
        return {
            "employee_id": employee_id,
            "name": emp["name"],
            "department": emp["department"],
            "total": emp["annual_leave_total"],
            "used": emp["annual_leave_used"],
            "carry_over": emp["carry_over"],
            "remaining": remaining,
        }


class SubmitLeaveTool(BaseTool):
    """提交请假申请（写操作，需人工审批）。"""

    name = "submit_leave_request"
    description = "提交年假申请；写操作，调用前必须经过人工确认"

    def __init__(self) -> None:
        super().__init__()
        self.parameters = [
            ToolParameter(name="employee_id", description="员工工号"),
            ToolParameter(name="start_date", description="开始日期，YYYY-MM-DD"),
            ToolParameter(name="days", type="integer", description="请假天数"),
        ]

    async def execute(self, **kwargs: Any) -> Any:
        employee_id = str(kwargs.get("employee_id", "E1001"))
        start = str(kwargs.get("start_date") or "")
        days = int(kwargs.get("days") or 1)

        emp = EMPLOYEES.get(employee_id)
        if not emp:
            raise ValueError(f"未找到员工 {employee_id}")
        remaining = emp["annual_leave_total"] - emp["annual_leave_used"] + emp["carry_over"]
        if days > remaining:
            raise ValueError(f"余额不足：申请 {days} 天，仅剩 {remaining} 天")

        key = f"{employee_id}:{start}:{days}"
        if key in _IDEMPOTENCY:
            return {"request_id": _IDEMPOTENCY[key], "duplicated": True}

        request_id = f"LR-{uuid.uuid4().hex[:8].upper()}"
        LEAVE_REQUESTS[request_id] = {
            "request_id": request_id,
            "employee_id": employee_id,
            "name": emp["name"],
            "start_date": start,
            "days": days,
            "status": "已提交",
        }
        _IDEMPOTENCY[key] = request_id
        emp["annual_leave_used"] += days
        return {"request_id": request_id, "duplicated": False, **LEAVE_REQUESTS[request_id]}


class TicketStatusTool(BaseTool):
    """查询工单状态。"""

    name = "get_ticket_status"
    description = "按工单号查询当前状态、处理人与剩余 SLA"

    def __init__(self) -> None:
        super().__init__()
        self.parameters = [
            ToolParameter(name="ticket_id", description="工单号，如 T-1001"),
        ]

    async def execute(self, **kwargs: Any) -> Any:
        ticket_id = str(kwargs.get("ticket_id", "")).strip().upper()
        ticket = TICKETS.get(ticket_id)
        if not ticket:
            raise ValueError(f"未找到工单 {ticket_id}")
        return {"ticket_id": ticket_id, **ticket}


def build_default_tools() -> list[BaseTool]:
    return [LeaveBalanceTool(), SubmitLeaveTool(), TicketStatusTool()]


__all__ = [
    "EMPLOYEES",
    "LEAVE_REQUESTS",
    "TICKETS",
    "LeaveBalanceTool",
    "SubmitLeaveTool",
    "TicketStatusTool",
    "build_default_tools",
    "_parse_weekday",
]
