"""RAG 检索评测 v3：精确型 vs 语义型，对比四路（含重排）。

用法：python eval_retrieval.py [base_url]
"""
from __future__ import annotations

import sys
from typing import Any

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

CASES: list[tuple[str, str, str]] = [
    # ---- 精确型 ----
    ("精确", "发票税号是多少", "91310000MA1FL8XX2Y"),
    ("精确", "P0 工单多久响应", "15 分钟"),
    ("精确", "每月加班时长上限是多少小时", "36 小时"),
    ("精确", "VPN 账号有效期多少天", "90 天"),
    ("精确", "备份保留多少天", "30 天"),
    ("精确", "租车每公里补贴多少钱", "1.2 元"),
    ("精确", "RTO 和 RPO 分别是多少", "RTO"),
    ("精确", "报销单超过多少金额需要财务经理复核", "5000 元"),
    # ---- 语义型 ----
    ("语义", "出差打车一次好几百能全报吗", "100 元"),
    ("语义", "报销交上去很久了钱还没到", "15 个工作日"),
    ("语义", "密码用久了会不会强制我改", "90 天"),
    ("语义", "半夜系统报警没人看会怎样", "值班"),
    ("语义", "客户反复催同一个问题该怎么处理", "升级"),
    ("语义", "新同事第一天电脑和账号谁给", "IT 部门"),
    ("语义", "实习生出差住宿能报多少", "80%"),
    ("语义", "上线出问题怎么快速退回去", "回滚"),
]

# (显示名, mode, rerank)
MODES: list[tuple[str, str, bool]] = [
    ("vector", "vector", False),
    ("keyword", "keyword", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]


def search(query: str, mode: str, rerank: bool, top_k: int = 5) -> list[dict[str, Any]]:
    resp = requests.post(
        f"{BASE}/api/v1/search",
        json={"query": query, "top_k": top_k, "mode": mode, "rerank": rerank},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["hits"]


def evaluate(cases: list[tuple[str, str, str]]) -> dict[str, dict[str, float]]:
    stats = {name: {"hit1": 0, "recall5": 0, "mrr": 0.0} for name, _, _ in MODES}
    for _, question, keyword in cases:
        for name, mode, rerank in MODES:
            hits = search(question, mode, rerank)
            contents = [h["content"] for h in hits]
            rank = next((i + 1 for i, c in enumerate(contents) if keyword in c), 0)
            if rank == 1:
                stats[name]["hit1"] += 1
            if rank:
                stats[name]["recall5"] += 1
                stats[name]["mrr"] += 1.0 / rank
    return stats


def show(stats: dict[str, dict[str, float]], n: int, label: str) -> None:
    print(f"\n[{label}] 样本数={n}")
    print(f"{'模式':<16}{'Hit@1':<10}{'Recall@5':<12}{'MRR':<8}")
    for name, _, _ in MODES:
        s = stats[name]
        print(f"{name:<16}{int(s['hit1'])}/{n:<8}{int(s['recall5'])}/{n:<10}{s['mrr'] / n:.3f}")


def main() -> None:
    exact = [c for c in CASES if c[0] == "精确"]
    semantic = [c for c in CASES if c[0] == "语义"]
    show(evaluate(CASES), len(CASES), "全部")
    show(evaluate(exact), len(exact), "精确型")
    show(evaluate(semantic), len(semantic), "语义型")


if __name__ == "__main__":
    main()
