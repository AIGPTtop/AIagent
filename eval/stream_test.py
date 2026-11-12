"""流式 RAG 冒烟：验证检索事件、逐 token 输出、引用与 TTFT。"""
from __future__ import annotations

import json
import time

import requests

URL = "http://127.0.0.1:8000/api/v1/chat/stream"
QUESTIONS = ["年假有几天", "公司食堂在哪"]


def run(question: str) -> None:
    t0 = time.perf_counter()
    first_delta = None
    retrieval_at = None
    answer_parts: list[str] = []
    citations: list[dict] = []
    events = 0

    with requests.post(URL, json={"messages": [{"role": "user", "content": question}]}, stream=True, timeout=180) as r:
        print(f"\n=== {question}   status={r.status_code}")
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            payload = json.loads(raw[6:])
            events += 1
            etype = payload.get("type")
            if etype == "retrieval":
                retrieval_at = (time.perf_counter() - t0) * 1000
                print(f"  [retrieval] {retrieval_at:6.0f}ms  hits={len(payload['hits'])} refused={payload['refused']}")
                if payload["hits"]:
                    print(f"              top1={payload['hits'][0]['snippet'][:40]!r} score={payload['hits'][0]['score']}")
            elif etype == "refusal":
                print(f"  [refusal] {payload['content'][:60]}")
            elif etype == "delta":
                if first_delta is None:
                    first_delta = (time.perf_counter() - t0) * 1000
                answer_parts.append(payload["content"])
            elif etype == "done":
                citations = payload.get("citations") or []
                total = (time.perf_counter() - t0) * 1000
                print(f"  [done]      {total:6.0f}ms  events={events}  citations={len(citations)}")
                if citations:
                    print(f"              引用: {[(c['index'], c['result_id'][:8]) for c in citations]}")
                ttft = f"{first_delta:.0f}ms" if first_delta else "-"
                print(f"  TTFT={ttft}  总耗时={total:.0f}ms")
            elif etype == "error":
                print(f"  [error] {payload.get('message')}")

    print(f"  答案：{''.join(answer_parts)[:120]}")


if __name__ == "__main__":
    for q in QUESTIONS:
        run(q)
