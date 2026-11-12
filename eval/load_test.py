"""压测：/chat 全链路（RAG + 重排）在不同并发下的吞吐与延迟。"""
from __future__ import annotations

import asyncio
import statistics
import time

import httpx

BASE = "http://127.0.0.1:8000/api/v1/chat"
QUESTIONS = [
    "年假有几天",
    "发票税号是多少",
    "P0 工单多久响应",
    "出差住宿标准是多少",
    "加班能调休吗",
    "离职要提前多久",
    "备份保留多少天",
    "竞业限制期限多长",
]


async def worker(client: httpx.AsyncClient, deadline: float, lat: list[float], err: list[int]) -> None:
    i = 0
    while time.perf_counter() < deadline:
        q = QUESTIONS[i % len(QUESTIONS)]
        i += 1
        start = time.perf_counter()
        try:
            r = await client.post(
                BASE,
                json={"messages": [{"role": "user", "content": q}]},
                timeout=120,
            )
            if r.status_code != 200:
                err.append(r.status_code)
            else:
                lat.append((time.perf_counter() - start) * 1000)
        except Exception:
            err.append(-1)


async def run(concurrency: int, duration: float) -> None:
    lat: list[float] = []
    err: list[int] = []
    async with httpx.AsyncClient() as client:
        # 预热，避免冷启动（模型加载）污染数据
        await client.post(BASE, json={"messages": [{"role": "user", "content": "年假有几天"}]}, timeout=180)
        deadline = time.perf_counter() + duration
        start = time.perf_counter()
        await asyncio.gather(*[worker(client, deadline, lat, err) for _ in range(concurrency)])
        elapsed = time.perf_counter() - start

    if not lat:
        print(f"并发 {concurrency}: 全部失败 ({len(err)} 次)")
        return
    lat.sort()
    p = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))]  # noqa: E731
    print(
        f"并发 {concurrency:<3} 请求 {len(lat)+len(err):<4} 成功 {len(lat):<4} "
        f"QPS {len(lat)/elapsed:6.2f}  "
        f"均值 {statistics.mean(lat):7.1f}ms  P50 {p(0.50):7.1f}ms  "
        f"P95 {p(0.95):7.1f}ms  P99 {p(0.99):7.1f}ms  错误 {len(err)}"
    )


def main() -> None:
    for concurrency in (5, 20, 50):
        asyncio.run(run(concurrency, 25))


if __name__ == "__main__":
    main()
