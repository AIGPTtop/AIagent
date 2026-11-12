# 阶段 9 报告：流式 RAG（SSE）

日期：2026-11-11

## 1. 事件协议

`POST /api/v1/chat/stream` 返回 `text/event-stream`，事件顺序固定：

| 事件 | 时机 | 内容 |
|---|---|---|
| `retrieval` | 检索完成、生成开始前 | 命中的块（序号、id、分数、摘要）与是否拒答 |
| `delta` | 生成过程中 | 逐 token 的文本增量 |
| `refusal` | 证据不足 | 固定拒答文案，**不调用模型** |
| `done` | 结束 | 引用列表（citations）与用量 |
| `error` | 异常 | 错误信息 |

先发 `retrieval` 的好处：前端可以在答案开始前就把「引用了哪几段」显示出来，用户等待时的焦虑感明显降低；也让「引用」不再是答案出来后才补的装饰。

## 2. 实测（预热后，连续三次）

| 指标 | 数值 |
|---|---|
| 检索完成（retrieval 事件） | 328 / 354 / 369 ms |
| 首 token（TTFT） | 1254 / 1205 / 1432 ms |
| 完整答案 | 1819 / 1714 / 1952 ms |
| 拒答路径总耗时 | 342 / 337 / 309 ms |

对比非流式 `/chat`（1.6–1.9 s 才返回第一句话）：**流式让用户 1.2 s 看到首字、0.33 s 就看到引用，而不是 1.8 s 后一次性出现。** 拒答场景因为不调模型，只花 0.3 s 左右。

## 3. 冷启动问题与修复

修复前，服务重启后的第一个请求：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 检索完成 | 19 283 ms | 328 ms |
| 首 token | 20 046 ms | 1254 ms |

原因是首个请求同步加载 bge-m3（约 12 s）与 bge-reranker（约 6 s）。修复方式：在 `lifespan` 里起后台任务并行预热两个模型，不阻塞启动。

```python
app.state.warmup_task = asyncio.create_task(warmup())
```

日志确认：`模型预热完成（embedding + reranker）`。

这个问题在本地调试时看不出来（第二次请求就快了），但生产上「发布后第一个用户等 20 秒」是实打实的故障。

## 4. 新增/修改文件

| 文件 | 说明 |
|---|---|
| `app/core/rag/stream.py` | 新增：SSE 生成器（retrieval / delta / refusal / done / error） |
| `app/api/routes/chat.py` | 改：`/chat/stream` 接入 RAG，流结束后补写 trace |
| `app/core/warmup.py` | 新增：启动预热 embedding + reranker |
| `app/main.py` | 改：lifespan 启动预热任务 |
| `eval/stream_test.py` | 新增：流式冒烟（事件顺序、TTFT、引用） |

## 5. 已知问题

1. **流式响应没有 usage**：OpenAI 兼容接口的流式用量需要 `stream_options={"include_usage": true}`，DeepSeek 当前未见支持，因此 `done` 事件里 usage 为 null。
2. **记忆的摘要触发在流结束后**：用户感知不到，但会在响应收尾时多一次模型调用（约 1 s）。如果摘要慢，客户端可能已收到 `done` 而连接未关闭。
3. **没有背压控制**：客户端读取慢时，服务端仍会持续生成；生产应结合队列长度做限流。
4. **反向代理**：已设置 `X-Accel-Buffering: no`，否则 Nginx 会缓冲整个响应，流式失效。
