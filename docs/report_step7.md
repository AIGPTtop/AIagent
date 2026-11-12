# 阶段 7 报告：可观测性与压测

日期：2026-11-11

## 1. Trace 落地

- 每次 `/chat` 结束把根 Span 写入 PostgreSQL `trace_logs`（trace_id / span_id / operation / duration_ms / payload / error）
- 新增 `GET /api/v1/trace/{trace_id}`：返回该 trace 的全部 span
- 写入失败只记 warning，不影响业务请求（可观测性不能反向拖垮主链路）

payload 记录：模型名、是否拒答、引用数、检索条数、token 用量、是否带记忆。

## 2. 延迟指标

`GET /api/v1/metrics/latency?operation=chat&limit=300` 直接从 trace_logs 聚合：

```json
{"operation":"chat","count":300,"avg_ms":7165.9,"p50_ms":5617,
 "p95_ms":17998,"p99_ms":19561,"max_ms":19815,"errors":0,"error_rate":0.0}
```

（这批样本包含压测流量，所以 p95 接近 18 s。）

## 3. 熔断器验证

用不可达的 base_url 构造 router，连续调用 5 次：

| 次数 | 耗时 | 熔断状态 |
|---|---|---|
| 1 | 1445 ms | closed |
| 2 | 1420 ms | closed |
| 3 | 1391 ms | **open** |
| 4 | **0 ms** | open |
| 5 | **0 ms** | open |

失败 3 次后进入 OPEN，后续请求立即失败（0 ms），不再白等 1.4 秒超时。`GET /metrics/circuit` 可查看实时状态。

## 4. 压测结果（`/chat` 全链路：检索 + 重排 + 生成）

单机单 worker，每档跑 25 秒：

| 并发 | 请求数 | 成功 | QPS | 均值 | P50 | P95 | P99 | 错误 |
|---|---|---|---|---|---|---|---|---|
| 5 | 75 | 75 | 2.86 | 1719 ms | 1675 ms | 2699 ms | 3194 ms | 0 |
| 20 | 106 | 106 | 3.81 | 5035 ms | 4756 ms | 8198 ms | 8414 ms | 0 |
| 50 | 116 | 116 | 3.45 | 13061 ms | 11881 ms | 20036 ms | 20601 ms | 0 |

## 5. 分析：瓶颈在哪

- 并发 5 时 QPS 2.86、均值 1.72 s，符合 Little 定律（5 / 1.72 ≈ 2.9）。
- 并发升到 20、50，QPS 不再上升（3.8 → 3.45），延迟近似线性增长 —— 这是**典型饱和**：系统吞吐已到上限，多余并发只是在排队。
- 单请求耗时拆解（估算）：检索 0.3 s（embedding 0.03 + 向量检索 0.01 + BM25 0.01 + 重排 0.22）+ 生成 1.2–1.4 s。**生成占大头，且是外部 API，不受本机扩容影响。**

优化方向（按性价比排序）：

1. **多 worker**：`uvicorn --workers 4`，把 CPU 侧（分词、序列化、JSON）摊开；生成仍是串行，但队列会变短。
2. **重排降本**：候选 20 → 10、max_length 512 → 256，预计重排从 220 ms 降到 100 ms 以内。
3. **同问缓存**：企业知识库问答重复率很高，Redis 缓存 (query → answer) 命中即可秒回。
4. **流式输出**：把「等完整答案」变成「边生成边吐」，用户感知延迟从 1.7 s 降到首 token 约 0.4 s。

## 6. 新增/修改文件

| 文件 | 说明 |
|---|---|
| `app/infrastructure/trace/persist.py` | 新增：Span 写 PG |
| `app/api/routes/observability.py` | 新增：`/trace/{id}`、`/metrics/latency`、`/metrics/circuit` |
| `app/infrastructure/llm/model_router.py` | 增：`breaker_states()` 导出熔断状态 |
| `app/api/routes/chat.py` | 改：共享 ModelRouter 单例（熔断计数跨请求累计）、请求结束时持久化 trace |
| `eval/load_test.py` | 新增：并发 5/20/50 压测 |
| `eval/breaker_test.py` | 新增：熔断状态机验证 |

## 7. 已知问题

1. 单 worker 是压测上限的主因之一，但本次刻意保留单 worker 以测出真实单实例能力。
2. trace 写入与业务共用同一个 DB session，高并发下会增加连接压力；生产应改为异步队列 + 批量写。
3. 压测脚本用 httpx 同步等待，没有做连接池调优，客户端可能也是瓶颈之一（但错误率为 0，说明不是主要因素）。
4. 压测期间 GPU 与其他用户共享，重排延迟有波动。
