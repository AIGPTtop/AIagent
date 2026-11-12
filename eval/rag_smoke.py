"""RAG 对话冒烟测试：验证引用、拒答与端到端延迟。"""
from __future__ import annotations

import time

import requests

QUESTIONS = [
    ("年假有几天", "应答"),
    ("P0 工单多久响应", "应答"),
    ("加班能调休吗", "应答"),
    ("公司有股权激励吗", "拒答"),
    ("公司食堂在哪", "拒答"),
    ("今天上海天气怎么样", "拒答"),
]


def ask(q: str) -> dict:
    start = time.perf_counter()
    r = requests.post(
        "http://127.0.0.1:8000/api/v1/chat",
        json={"messages": [{"role": "user", "content": q}]},
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    data["_latency_ms"] = (time.perf_counter() - start) * 1000
    return data


def main() -> None:
    for q, expect in QUESTIONS:
        d = ask(q)
        cites = d.get("citations", [])
        ok = ("拒答" if d.get("refused") else "应答") == expect
        print(f"[{'OK ' if ok else 'XX '}] {q}  ({expect})")
        print(
            f"      refused={d.get('refused')} retrieved={d.get('retrieved')} "
            f"cites={len(cites)} latency={d['_latency_ms']:.0f}ms"
        )
        print(f"      {d.get('content', '')[:120]}")


if __name__ == "__main__":
    main()
