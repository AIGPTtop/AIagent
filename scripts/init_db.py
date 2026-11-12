"""创建数据库表（SQLAlchemy metadata.create_all）。

用法：python scripts/init_db.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.infrastructure.database.models import Base  # noqa: E402
from app.infrastructure.database.session import init_engine  # noqa: E402


async def main() -> None:
    settings = get_settings()
    engine = init_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    print("表创建完成：", ", ".join(sorted(Base.metadata.tables)))


if __name__ == "__main__":
    asyncio.run(main())
