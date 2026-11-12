"""拒答阈值校准 + 端到端负样本评测。

1) 用 /search 拿到「域内 / 域外」问题的首条重排分分布；
2) 扫多个阈值，统计误拒（域内被拒）与正确拒答（域外被拒）率；
3) 用选定阈值跑 /chat，验证端到端行为。
"""
from __future__ import annotations

import json
import statistics
import sys

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/api/v1"

IN_DOMAIN = [
    "年假有几天",
    "发票税号是多少",
    "P0 工单多久响应",
    "出差住宿标准是多少",
    "加班能调休吗",
    "离职要提前多久",
    "标准版单空间多少人",
    "开放 API 限流多少",
    "销售代表最高几折",
    "采购三十万以上走什么流程",
    "供应商准入要多久",
    "付款日是周几",
    "核心工作时段",
    "夜班补贴多少",
    "绩效 S 档比例",
    "晋升要满多少个月",
    "审计日志保留多久",
    "跨境评估周期",
    "里程碑延期多少天升级",
    "Ⅰ级事件多久上报",
]

OUT_DOMAIN = [
    "今天上海天气怎么样",
    "公司食堂在哪",
    "有股权激励吗",
    "帮我写一首诗",
    "推荐一家附近的餐厅",
    "比特币现在多少钱",
    "怎么学 Python 最快",
    "明天股市会涨吗",
    "公司附近有地铁站吗",
    "Python 和 Java 哪个好",
    "如何申请美国签证",
    "世界杯冠军是谁",
    "给我讲个笑话",
    "北京明天限行吗",
    "双十一有什么优惠",
    "怎么减肥最有效",
    "公司有没有班车",
    "附近停车场收费多少",
    "如何提高英语口语",
    "推荐一部电影",
]

THRESHOLDS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def top_score(question: str) -> float:
    r = requests.post(
        f"{BASE}/search",
        json={"query": question, "top_k": 1, "mode": "hybrid", "rerank": True},
        timeout=120,
    )
    r.raise_for_status()
    hits = r.json()["hits"]
    return float(hits[0]["score"]) if hits else -99.0


def ask(question: str) -> bool:
    r = requests.post(
        f"{BASE}/chat",
        json={"messages": [{"role": "user", "content": question}]},
        timeout=180,
    )
    r.raise_for_status()
    return bool(r.json().get("refused"))


def main() -> None:
    print("采集重排分分布……")
    in_scores = [top_score(q) for q in IN_DOMAIN]
    out_scores = [top_score(q) for q in OUT_DOMAIN]

    in_sorted = sorted(in_scores)
    out_sorted = sorted(out_scores)
    print(f"域内 top1 分：min={in_sorted[0]:.3f} p10={in_sorted[1]:.3f} "
          f"median={statistics.median(in_sorted):.3f} max={in_sorted[-1]:.3f}")
    print(f"域外 top1 分：min={out_sorted[0]:.3f} median={statistics.median(out_sorted):.3f} "
          f"p90={out_sorted[-2]:.3f} max={out_sorted[-1]:.3f}")

    print("\n阈值扫描：")
    print(f"{'阈值':<8}{'误拒(域内被拒)':<18}{'正确拒答(域外)':<18}{'合计准确率':<10}")
    best = None
    for t in THRESHOLDS:
        false_refuse = sum(1 for s in in_scores if s < t)
        correct_refuse = sum(1 for s in out_scores if s < t)
        total = (len(in_scores) - false_refuse + correct_refuse) / (len(in_scores) + len(out_scores))
        print(f"{t:<8}{false_refuse}/{len(in_scores):<14}{correct_refuse}/{len(out_scores):<14}{total*100:.1f}%")
        if best is None or total > best[1]:
            best = (t, total)
    assert best is not None
    print(f"\n最优阈值 ≈ {best[0]}（准确率 {best[1]*100:.1f}%）")

    print("\n端到端验证（/chat）：")
    in_refused = sum(ask(q) for q in IN_DOMAIN)
    out_refused = sum(ask(q) for q in OUT_DOMAIN)
    print(f"  域内 20 题：误拒 {in_refused} 题")
    print(f"  域外 20 题：正确拒答 {out_refused} 题")
    print(json.dumps({
        "in_domain_false_refusal_rate": round(in_refused / len(IN_DOMAIN) * 100, 1),
        "out_domain_refusal_rate": round(out_refused / len(OUT_DOMAIN) * 100, 1),
        "in_domain_scores": [round(s, 3) for s in in_scores],
        "out_domain_scores": [round(s, 3) for s in out_scores],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
