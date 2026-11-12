"""RAG 检索评测 v4：14 份文档 / 114 块，40 题（精确 20 + 语义 20）。

用法：python eval_retrieval.py [base_url]
"""
from __future__ import annotations

import sys
from typing import Any

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

CASES: list[tuple[str, str, str]] = [
    # ---------------- 精确型（关键词/数字直接命中） ----------------
    ("精确", "标准版单空间最多多少人", "200 人"),
    ("精确", "开放 API 默认每分钟限流多少次", "600 次"),
    ("精确", "旗舰版的可用性 SLA 是多少", "99.95%"),
    ("精确", "私有化部署起售价格多少", "50 万元"),
    ("精确", "销售代表最高可以打几折", "9 折"),
    ("精确", "渠道返点 300 到 500 万的比例", "15%"),
    ("精确", "三十万以上的采购要走什么流程", "公开招标"),
    ("精确", "供应商准入审核要多久", "10 个工作日"),
    ("精确", "公司付款日是每周哪几天", "周二、周四"),
    ("精确", "核心工作时段是几点到几点", "10:00–17:00"),
    ("精确", "夜班补贴每晚多少钱", "80 元"),
    ("精确", "绩效 S 档比例上限是多少", "10%"),
    ("精确", "晋升需要当前职级满多少个月", "18 个月"),
    ("精确", "审计日志保留多长时间", "1 年"),
    ("精确", "跨境传输安全评估周期多久", "20 个工作日"),
    ("精确", "项目里程碑延期超过多少天要升级", "15 个工作日"),
    ("精确", "Ⅰ级安全事件多久内上报 CEO", "30 分钟"),
    ("精确", "审计日志和业务日志分别保留多久", "90 天"),
    ("精确", "发票认证抵扣的期限", "30 天"),
    ("精确", "资产标签编码规则举例", "XC-2026-00123"),

    # ---------------- 语义型（口语化，不含原文关键词） ----------------
    ("语义", "客户要的折扣超出我的权限怎么办", "审批"),
    ("语义", "供应商送货晚了会怎么罚", "违约金"),
    ("语义", "报销单金额特别大要谁签字", "财务总监"),
    ("语义", "上班忘打卡了还能补救吗", "补卡"),
    ("语义", "培训学时不够会影响我评优吗", "A 及以上"),
    ("语义", "年终奖一般什么时候到账", "1 月 20 日"),
    ("语义", "能不能把线上数据拷到测试环境用", "脱敏"),
    ("语义", "系统被入侵了第一步该干什么", "取证"),
    ("语义", "项目做完要交给客户哪些材料", "交付物"),
    ("语义", "客户的数据可以传到境外吗", "备案"),
    ("语义", "节假日加班是给钱还是给假", "3 倍"),
    ("语义", "迟到几次开始扣钱", "30 元"),
    ("语义", "培训没过考试会怎么样", "延期转正"),
    ("语义", "绩效连续两次垫底会怎样", "PIP"),
    ("语义", "用户信息泄露了多久要报监管部门", "72 小时"),
    ("语义", "服务器被攻击了能先把日志删掉吗", "镜像"),
    ("语义", "采购多少钱以上要找三家比价", "5 万元"),
    ("语义", "和客户签合同账期最多能放多久", "90 天"),
    ("语义", "客户一直拖着不付钱会有什么后果", "观察名单"),
    ("语义", "员工离职后电脑怎么处理", "数据擦除"),
]

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
    stats = {name: {"hit1": 0, "recall3": 0, "recall5": 0, "mrr": 0.0} for name, _, _ in MODES}
    misses: list[str] = []
    for kind, question, keyword in cases:
        row_miss = []
        for name, mode, rerank in MODES:
            hits = search(question, mode, rerank)
            contents = [h["content"] for h in hits]
            rank = next((i + 1 for i, c in enumerate(contents) if keyword in c), 0)
            if rank == 1:
                stats[name]["hit1"] += 1
            if rank and rank <= 3:
                stats[name]["recall3"] += 1
            if rank:
                stats[name]["recall5"] += 1
                stats[name]["mrr"] += 1.0 / rank
            else:
                row_miss.append(name)
        if row_miss:
            misses.append(f"  [{kind}] {question} 未命中: {', '.join(row_miss)}")
    if misses:
        print("\n未命中的题目：")
        print("\n".join(misses))
    return stats


def show(stats: dict[str, dict[str, float]], n: int, label: str) -> None:
    print(f"\n[{label}] 样本数={n}")
    print(f"{'模式':<16}{'Hit@1':<10}{'Recall@3':<10}{'Recall@5':<12}{'MRR':<8}")
    for name, _, _ in MODES:
        s = stats[name]
        print(f"{name:<16}{int(s['hit1'])}/{n:<8}{int(s['recall3'])}/{n:<8}{int(s['recall5'])}/{n:<10}{s['mrr'] / n:.3f}")


def main() -> None:
    exact = [c for c in CASES if c[0] == "精确"]
    semantic = [c for c in CASES if c[0] == "语义"]
    show(evaluate(CASES), len(CASES), "全部")
    show(evaluate(exact), len(exact), "精确型")
    show(evaluate(semantic), len(semantic), "语义型")


if __name__ == "__main__":
    main()

