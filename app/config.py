"""应用配置：基于 pydantic-settings，支持环境变量与 .env 文件。"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="enterprise-ai-agent", description="服务名称")
    app_env: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=False, description="调试模式")
    api_prefix: str = Field(default="/api/v1", description="API 前缀")
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8000, description="监听端口")

    openai_api_key: str = Field(default="", description="OpenAI API Key")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI 兼容 API Base",
    )
    openai_model: str = Field(default="gpt-4o-mini", description="默认对话模型")

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/agent_db",
        description="SQLAlchemy 异步数据库 URL（推荐 postgresql+asyncpg）",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")

    milvus_host: str = Field(default="localhost", description="Milvus 主机")
    milvus_port: int = Field(default=19530, description="Milvus 端口")
    milvus_user: str = Field(default="", description="Milvus 用户名")
    milvus_password: str = Field(default="", description="Milvus 密码")
    milvus_collection_name: str = Field(
        default="agent_knowledge",
        description="默认向量集合名",
    )
    milvus_uri: str = Field(
        default="/public/FWX/AI_Agent/milvus/agent_knowledge.db",
        description="Milvus Lite 本地文件路径（本地开发/演示用）",
    )

    # --- 向量化 ---
    embedding_model_path: str = Field(
        default="/public/FWX/AI_Agent/models/BAAI/bge-m3",
        description="embedding 模型本地路径",
    )
    embedding_device: str = Field(default="cuda:0", description="embedding 使用的 GPU")
    embedding_batch_size: int = Field(default=16, description="embedding 批大小")
    embedding_max_length: int = Field(default=512, description="embedding 最大 token 数")
    embedding_dim: int = Field(default=1024, description="向量维度（bge-m3 为 1024）")

    # --- 检索与重排 ---
    rag_enabled: bool = Field(default=False, description="是否启用 RAG 检索增强")
    rag_top_k_dense: int = Field(default=20, description="向量召回条数")
    rag_top_k_sparse: int = Field(default=20, description="BM25 召回条数")
    rag_top_k_final: int = Field(default=5, description="重排后保留条数")
    rag_rerank_candidates: int = Field(default=20, description="进入重排的候选条数")
    rag_min_rerank_score: float = Field(
        default=0.05,
        description="重排分下限，低于该值判定为证据不足并拒答",
    )
    rag_context_max_chars: int = Field(default=6000, description="拼入提示的证据字符上限")

    # --- 分层记忆 ---
    memory_ttl_seconds: int = Field(default=1800, description="短期记忆 TTL（秒）")
    memory_summary_ttl_seconds: int = Field(default=7200, description="摘要 TTL（秒）")
    memory_max_messages: int = Field(default=20, description="短期记忆最多保留条数")
    memory_summary_trigger: int = Field(default=12, description="超过多少条触发摘要压缩")
    memory_summary_keep: int = Field(default=6, description="摘要后保留的原文条数")
    rerank_model_path: str = Field(
        default="/public/FWX/AI_Agent/models/BAAI/bge-reranker-v2-m3",
        description="reranker 模型本地路径",
    )
    rerank_device: str = Field(default="cuda:1", description="reranker 使用的 GPU")
    rerank_threshold: float = Field(default=0.0, description="重排分数阈值，低于则拒答")

    log_level: str = Field(default="INFO", description="日志级别")


@lru_cache
def get_settings() -> Settings:
    return Settings()
