# 阶段 2 报告：分块重构 + 混合检索 + 重排

日期：2026-11-11　　环境：Ubuntu 18.04 / Python 3.12.11 / torch 2.2.2+cu118 / RTX 3090

## 1. 语料

| 文档 | 块数 |
|---|---|
| employee_handbook.md（员工手册） | 11 |
| expense_policy.md（报销细则） | 8 |
| it_security.md（IT 安全规范） | 8 |
| support_faq.md（客服 FAQ） | 7 |
| **合计** | **34** |

改造前后对比：员工手册从 4 块（512 token 硬切、跨章节串味）变为 11 块（按 `##` 标题切、256 token、保留标题前缀）。

## 2. 分块策略变更

| 项 | 改造前 | 改造后 |
|---|---|---|
| 策略 | 固定长度递归切分 | Markdown 标题感知 |
| chunk_size | 512 token | 256 token |
| overlap | 64 | 64 |
| 块内上下文 | 可能横跨 2–3 章 | 单章为主，块首保留标题 |

## 3. 检索评测（16 题，Hit@1 / Recall@5 / MRR）

| 模式 | Hit@1 | Recall@5 | MRR |
|---|---|---|---|
| vector | 13/16 | 15/16 | 0.875 |
| keyword (BM25) | 14/16 | 15/16 | 0.906 |
| hybrid (RRF) | 14/16 | 15/16 | 0.906 |
| **hybrid + rerank** | **15/16** | **16/16** | **0.958** |

分组结果：

| 子集 | vector | keyword | hybrid | hybrid+rerank |
|---|---|---|---|---|
| 精确型 Hit@1 | 7/8 | 8/8 | 7/8 | 8/8 |
| 精确型 MRR | 0.938 | 1.000 | 0.938 | 1.000 |
| 语义型 Hit@1 | 6/8 | 6/8 | 7/8 | 7/8 |
| 语义型 Recall@5 | 7/8 | 7/8 | 7/8 | 8/8 |
| 语义型 MRR | 0.812 | 0.812 | 0.875 | 0.917 |

结论：

1. 精确型问题（税号、天数、金额）BM25 全中，向量漏 1 条 —— 与「向量对精确 token 不敏感」的预期一致。
2. 语义型问题混合检索优于单路（Hit@1 7/8 vs 6/8），说明 RRF 确实补了短板。
3. 重排把 Recall@5 从 15/16 提到 16/16、Hit@1 从 14 提到 15，是收益最大的一步。
4. 未调参的朴素 RRF（等权、全量融合）曾使 Hit@1 降到 11/16；改为加权（向量 1.0 / BM25 0.7）+ 候选集上限后才优于单路 —— 融合策略本身需要调。

## 4. 延迟（8 个查询，单并发）

| 模式 | 均值 | P50 | P95 |
|---|---|---|---|
| vector | 76.6 ms | 76.6 ms | 85.0 ms |
| keyword | 3.2 ms | 2.4 ms | 5.8 ms |
| hybrid | 84.6 ms | 85.2 ms | 87.0 ms |
| hybrid+rerank | 308.9 ms | 308.9 ms | 325.3 ms |

重排增加约 225 ms（20 候选 × 512 token Cross-Encoder，单卡 3090）。可优化方向：候选数降到 10、fp16 推理、max_length 降到 256。

## 5. 新增/修改文件

| 文件 | 说明 |
|---|---|
| `app/etl/chunker.py` | 重写：新增 MARKDOWN 策略，默认 256 token |
| `app/core/rag/hybrid.py` | 新增：jieba + BM25Okapi + 加权 RRF + Cross-Encoder 精排 |
| `app/api/routes/search.py` | 改：支持 mode=vector/keyword/hybrid 与 rerank 开关 |
| `app/api/routes/document.py` | 改：使用 MARKDOWN 分块，入库后失效 BM25 缓存 |
| `eval/eval_retrieval.py` | 新增：16 题四路对比评测 |
| `eval/bench_latency.py` | 新增：延迟基准 |

## 6. 已知问题

1. Milvus Lite 单进程独占锁，评测脚本只能走 HTTP 接口，无法直接读库。
2. BM25 索引在进程内存中重建，文档多时需要持久化（阶段 5 处理）。
3. 语料仅 34 块，指标易饱和；需扩到千级块才能反映真实差距。
