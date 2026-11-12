"""检索延迟基准：四种模式的均值 / P50 / P95。"""
from __future__ import annotations

import statistics
import time

import requests

QUESTIONS = [
    "年假有几天",
    "发票税号是多少",
    "工单超时怎么处理",
    "实习生住宿能报多少",
    "密码多久换一次",
    "备份保留多少天",
    "客户反复催同一个问题",
    "上线出问题怎么退回去",
]

MODES = [
    ("vector", "vector", False),
    ("keyword", "keyword", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]


def main() -> None:
    for name, mode, rerank in MODES:
        lat = []
        for q in QUESTIONS:
            start = time.perf_counter()
            requests.post(
                "http://127.0.0.1:8000/api/v1/search",
                json={"query": q, "top_k": 5, "mode": mode, "rerank": rerank},
                timeout=180,
            )
            lat.append((time.perf_counter() - start) * 1000)
        lat.sort()
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        print(
            f"{name:<16} 均值 {statistics.mean(lat):7.1f}ms   "
            f"P50 {statistics.median(lat):7.1f}ms   P95 {p95:7.1f}ms"
        )


if __name__ == "__main__":
    main()
