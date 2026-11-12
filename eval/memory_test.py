"""分层记忆验收：多轮对话下的 token 增长、摘要触发与归档。"""
from __future__ import annotations

import json

import requests

BASE = "http://127.0.0.1:8000/api/v1"
CID = "conv-mem-demo-4"

TURNS = [
    "年假有几天",
    "那我满三年了算多少天",
    "出差住宿一线城市标准是多少",
    "发票税号是多少",
    "P0 工单多久响应",
    "加班能调休吗",
    "每个月加班上限多少小时",
    "备份保留多少天",
    "离职要提前多久申请",
    "竞业限制期限多长",
]


def ask(text: str) -> dict:
    r = requests.post(
        f"{BASE}/chat",
        json={
            "messages": [{"role": "user", "content": text}],
            "conversation_id": CID,
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    print(f"{'轮次':<5}{'消息数':<8}{'摘要字数':<10}{'prompt_tokens':<15}{'全量tokens':<12}{'节省':<8}")
    for i, q in enumerate(TURNS, start=1):
        d = ask(q)
        m = d.get("memory") or {}
        print(
            f"{i:<5}{m.get('messages', 0):<8}{m.get('summary_chars', 0):<10}"
            f"{m.get('prompt_tokens', 0):<15}{m.get('full_history_tokens', 0):<12}"
            f"{m.get('saved_tokens', 0):<8}"
        )

    stats = requests.get(f"{BASE}/memory/{CID}", timeout=30).json()
    print("\n最终记忆状态：")
    print(json.dumps({k: v for k, v in stats.items() if k != "summary"}, ensure_ascii=False, indent=2))
    print("\n摘要内容：", stats.get("summary") or "（无）")


if __name__ == "__main__":
    main()
