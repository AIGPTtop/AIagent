# 阶段 10 报告：负样本评测、多 worker 实验、容器化与交付

日期：2026-11-11

## 1. 拒答阈值校准（负样本评测）

构造 20 条域内问题与 20 条域外问题，取首条重排分：

| 分组 | min | 中位数 | max |
|---|---|---|---|
| 域内 | 0.160 | 0.966 | 0.998 |
| 域外 | 0.000 | 0.006 | 0.064 |

阈值扫描（准确率 = 域内不误拒 + 域外正确拒）：

| 阈值 | 误拒（域内） | 正确拒答（域外） | 准确率 |
|---|---|---|---|
| 0.02 | 0/20 | 16/20 | 90.0% |
| 0.05 | 0/20 | 18/20 | 95.0% |
| **0.10** | **0/20** | **20/20** | **100%** |
| 0.20 | 1/20 | 20/20 | 97.5% |

结论：0.1 落在「域外最高 0.064」与「域内最低 0.160」的间隙内，裕度约 ±0.05。
已将 `RAG_MIN_RERANK_SCORE` 从 0.05 调整为 **0.1**，端到端复测：域内误拒 0/20，域外拒答 20/20。

> 之前 0.05 是拍脑袋定的，实测会漏掉 2 条域外问题（被当成有证据回答了）。这类“看起来能用”的默认值，必须用负样本验证。

## 2. 多 worker 实验（失败但结论有价值）

| 配置 | 结果 |
|---|---|
| 1 worker @20 并发 | QPS 3.00，0 错误，P50 5.95 s |
| 2 worker @20 并发 | 31 次 500，GPU 显存打满 |
| 4 worker @20 并发 | 43 次 500 |

根因：**每个 worker 进程独立加载 bge-m3 与 bge-reranker，各占约 4 GB 显存**；本机 GPU 还被其他用户占用（GPU0 15.7 GB、GPU1 9.9 GB），两个副本即 OOM：

```
RuntimeError: 重排序失败: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 1 has a total capacity of 23.70 GiB of which 11.56 MiB is free.
```

结论与正确做法：

1. 本机（模型进程内加载）**不适合横向扩 worker**；
2. 生产应把 embedding / rerank 拆成独立推理服务（或多副本 + 模型服务常驻），API 层无状态扩展；
3. 短期替代：减小模型（bge-small）、降低重排候选、开启 fp16。

## 3. 容器化交付（静态，未运行）

- `project-python/Dockerfile`：python:3.12-slim + torch 2.2.2(cu118) + 依赖，模型通过 volume 挂载，带 HEALTHCHECK
- `docker-compose.yml`：app / postgres / redis / etcd / minio / milvus 六个服务，postgres 与 redis 带健康检查，app 依赖 `service_healthy`，GPU 通过 `deploy.resources.reservations.devices` 申请
- 环境差异通过 `MILVUS_URI=http://milvus:19530` 覆盖（代码用 MilvusClient，同一套逻辑兼容 Lite 与服务端）

已用 yaml 解析校验结构，符合 compose 规范；**未实际 `docker compose up`**，因为本机无 Docker 且无 root。

## 4. 交付物

| 文件 | 说明 |
|---|---|
| `README.md` | 项目总览、能力矩阵、接口清单、评测数据、限制 |
| `eval/acceptance.py` | 12 项端到端验收脚本 |
| `eval/report_step*.md` | 各阶段报告（2/4/5/6/7/8/9/10） |
| `eval/*.py` | 检索、拒答、RAG、流式、记忆、压测、熔断测试 |

## 5. 最终验收

```
[PASS] 健康检查          {'status': 'ok'}
[PASS] 就绪检查          database=up
[PASS] 知识库文档        14 docs
[PASS] 混合检索+重排      top1 score=0.812
[PASS] RAG 引用          cites=2
[PASS] 域外拒答          refused=True
[PASS] 流式 RAG          retrieval=True delta=True
[PASS] Agent 工具调用    tool=get_ticket_status
[PASS] 人工审批与恢复    pending=True final=completed
[PASS] 分层记忆          messages=2 ttl=1800
[PASS] 延迟指标          count=50
[PASS] 链路追踪          operation=chat

结果：12/12 通过
```
