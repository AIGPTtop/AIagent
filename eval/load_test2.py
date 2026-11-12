"""参数化压测：python load_test2.py <url> <concurrency> <seconds>"""
from __future__ import annotations

import asyncio
import statistics
import sys
import time

import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/api/v1/chat"
CONCURRENCY = int(sys.argv[2]) if len(sys.argv) > 2 else 20
DURATION = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0

QUESTIONS = [
    "年假有几天", "发票税号是多少", "P0 工单多久响应", "出差住宿标准是多少",
    "加班能调休吗", "离职要提前多久", "标准版单空间多少人", "夜班补贴多少",
]


async def worker(client: httpx.AsyncClient, deadline: float, lat: list[float], err: list[int]) -> None:
    i = 0
    while time.perf_counter() < deadline:
        q = QUESTIONS[i % len(QUESTIONS)]
        i += 1
        start = time.perf_counter()
        try:
            r = await client.post(URL, json={"messages": [{"role": "user", "content": q}]}, timeout=180)
            if r.status_code == 200:
                lat.append((time.perf_counter() - start) * 1000)
            else:
                err.append(r.status_code)
        except Exception:
            err.append(-1)


async def main() -> None:
    lat: list[float] = []
    err: list[int] = []
    async with httpx.AsyncClient() as client:
        await client.post(URL, json={"messages": [{"role": "user", "content": "年假有几天"}]}, timeout=300)
        deadline = time.perf_counter() + DURATION
        start = time.perf_counter()
        await asyncio.gather(*[worker(client, deadline, lat, err) for _ in range(CONCURRENCY)])
        elapsed = time.perf_counter() - start

    if not lat:
        print(f"全部失败: {len(err)}")
        return
    lat.sort()
    p = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))]  # noqa: E731
    print(
        f"并发 {CONCURRENCY:<3} 成功 {len(lat):<4} 失败 {len(err):<3} "
        f"QPS {len(lat)/elapsed:5.2f} | 均值 {statistics.mean(lat):7.1f}ms | "
        f"P50 {p(0.5):7.1f} P95 {p(0.95):7.1f} P99 {p(0.99):7.1f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
