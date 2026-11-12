"""端到端验收：一次跑完 12 项检查，输出 PASS/FAIL 与关键指标。"""
from __future__ import annotations

import json
import sys

import requests

BASE = "http://127.0.0.1:8000/api/v1"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def main() -> int:
    try:
        # 1 存活
        r = requests.get(f"{BASE}/health", timeout=10).json()
        check("健康检查", r.get("status") == "ok", str(r))

        # 2 就绪
        r = requests.get(f"{BASE}/health/ready", timeout=10).json()
        check("就绪检查", r.get("database") == "up", str(r))

        # 3 文档数
        docs = requests.get(f"{BASE}/documents", timeout=10).json()
        check("知识库文档", len(docs) >= 10, f"{len(docs)} docs")

        # 4 检索（混合 + 重排）
        r = requests.post(
            f"{BASE}/search",
            json={"query": "年假有几天", "top_k": 3, "mode": "hybrid", "rerank": True},
            timeout=120,
        ).json()
        hits = r["hits"]
        ok = bool(hits) and "年假" in hits[0]["content"]
        check("混合检索+重排", ok, f"top1 score={hits[0]['score'] if hits else '-'}")

        # 5 RAG 域内问答
        r = requests.post(
            f"{BASE}/chat",
            json={"messages": [{"role": "user", "content": "发票税号是多少"}]},
            timeout=180,
        ).json()
        ok = (not r.get("refused")) and len(r.get("citations", [])) >= 1
        check("RAG 引用", ok, f"refused={r.get('refused')} cites={len(r.get('citations', []))}")

        # 6 RAG 域外拒答
        r = requests.post(
            f"{BASE}/chat",
            json={"messages": [{"role": "user", "content": "今天上海天气怎么样"}]},
            timeout=180,
        ).json()
        check("域外拒答", bool(r.get("refused")), f"refused={r.get('refused')}")

        # 7 流式
        got_retrieval = got_delta = False
        with requests.post(
            f"{BASE}/chat/stream",
            json={"messages": [{"role": "user", "content": "夜班补贴多少钱"}]},
            stream=True,
            timeout=180,
        ) as resp:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                etype = json.loads(line[6:]).get("type")
                got_retrieval = got_retrieval or etype == "retrieval"
                got_delta = got_delta or etype == "delta"
                if etype == "done":
                    break
        check("流式 RAG", got_retrieval and got_delta, f"retrieval={got_retrieval} delta={got_delta}")

        # 8 Agent 工具
        r = requests.post(
            f"{BASE}/agent/chat", json={"message": "T-1001 现在什么状态"}, timeout=180
        ).json()
        ok = r.get("tool") == "get_ticket_status" and r.get("status") == "completed"
        check("Agent 工具调用", ok, f"tool={r.get('tool')} status={r.get('status')}")

        # 9 人工审批 + 恢复
        r = requests.post(f"{BASE}/agent/chat", json={"message": "帮我请下周五的假"}, timeout=180).json()
        tid = r.get("thread_id")
        pending = r.get("status") == "pending_approval"
        r2 = requests.post(
            f"{BASE}/agent/resume", json={"thread_id": tid, "approved": True}, timeout=180
        ).json()
        ok = pending and r2.get("status") == "completed" and bool(r2.get("tool_result"))
        check("人工审批与恢复", ok, f"pending={pending} final={r2.get('status')}")

        # 10 记忆
        requests.post(
            f"{BASE}/chat",
            json={"messages": [{"role": "user", "content": "年假有几天"}], "conversation_id": "acceptance-1"},
            timeout=180,
        )
        m = requests.get(f"{BASE}/memory/acceptance-1", timeout=10).json()
        check("分层记忆", m.get("messages", 0) >= 2 and m.get("ttl_seconds", 0) > 0,
              f"messages={m.get('messages')} ttl={m.get('ttl_seconds')}")

        # 11 指标
        met = requests.get(f"{BASE}/metrics/latency?operation=chat&limit=50", timeout=10).json()
        check("延迟指标", met.get("count", 0) > 0, f"count={met.get('count')} p95={met.get('p95_ms')}ms")

        # 12 trace
        tr = requests.post(
            f"{BASE}/chat", json={"messages": [{"role": "user", "content": "加班能调休吗"}]}, timeout=180
        ).json()
        trace = requests.get(f"{BASE}/trace/{tr['trace_id']}", timeout=10)
        check("链路追踪", trace.status_code == 200, trace.json().get("spans", [{}])[0].get("operation", ""))

    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 验收异常: {exc}")
        return 1

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n结果：{passed}/{total} 通过")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
