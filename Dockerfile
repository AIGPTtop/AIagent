# syntax=docker/dockerfile:1
# 说明：本镜像未在本机验证（该环境无 Docker / 无 root），仅做静态检查。
# 模型权重不打进镜像，运行时通过 volume 挂载到 /models。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) 先装 torch（cu118 轮子，与驱动 455+ 匹配），再装其余依赖，
#    避免 pip 解析时把 numpy 升到 2.x 导致 torch 2.2 不可用。
RUN pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

RUN pip install \
    "fastapi>=0.115" "uvicorn[standard]>=0.34" "pydantic>=2.10" "pydantic-settings>=2.7" \
    "sqlalchemy>=2.0" asyncpg greenlet psycopg2-binary \
    "pymilvus>=2.5" "milvus-lite>=2.4" "pyarrow==16.1.0" "numpy<2" \
    "redis>=5.2" "openai>=1.60" "httpx>=0.28" loguru tenacity python-dotenv \
    "langchain>=0.3" "langchain-openai>=0.3" "langgraph>=0.3" \
    "sentence-transformers==3.4.1" "transformers==4.49.0" "tiktoken>=0.8" \
    "rank-bm25>=0.2.2" jieba pypdf python-docx python-multipart

COPY requirements.txt /app/requirements.txt
COPY app /app/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
