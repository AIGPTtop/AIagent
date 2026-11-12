"""20 轮长对话：观察摘要压缩带来的 token 节省趋势。"""
from __future__ import annotations

import requests

BASE = "http://127.0.0.1:8000/api/v1"
CID = "conv-mem-long-1"
QUESTIONS = [
    "年假有几天",
    "满三年算多少天",
    "出差住宿一线标准",
    "发票税号是多少",
    "P0 工单多久响应",
    "加班能调休吗",
    "每月加班上限",
    "备份保留多少天",
    "离职提前多久",
    "竞业限制多久",
]


def main() -> None:
    header = ("轮", "消息", "摘要字", "prompt", "全量", "节省", "节省率")
    print("{:<4}{:<6}{:<8}{:<9}{:<8}{:<8}{:<8}".format(*header))
    for i in range(1, 21):
        q = QUESTIONS[(i - 1) % len(QUESTIONS)]
        resp = requests.post(
            f"{BASE}/chat",
            json={"messages": [{"role": "user", "content": q}], "conversation_id": CID},
            timeout=180,
        )
        data = resp.json()
        m = data.get("memory") or {}
        prompt_tokens = m.get("prompt_tokens", 0)
        full_tokens = m.get("full_history_tokens", 0)
        saved = m.get("saved_tokens", 0)
        rate = f"{saved / full_tokens * 100:.0f}%" if full_tokens else "-"
        print(
            "{:<4}{:<6}{:<8}{:<9}{:<8}{:<8}{:<8}".format(
                i, m.get("messages", 0), m.get("summary_chars", 0),
                prompt_tokens, full_tokens, saved, rate,
            )
        )
    stats = requests.get(f"{BASE}/memory/{CID}", timeout=30).json()
    print("\n最终状态：", {k: v for k, v in stats.items() if k != "summary"})


if __name__ == "__main__":
    main()
