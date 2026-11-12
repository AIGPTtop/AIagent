"""熔断器验证：连续失败后应进入 OPEN，并快速失败而不是一直等超时。"""
from __future__ import annotations

import asyncio
import time

from app.infrastructure.llm.model_router import ModelConfig, ModelRouter


async def main() -> None:
    router = ModelRouter(
        [
            ModelConfig(
                model_id="dead-model",
                api_key="sk-invalid",
                base_url="http://127.0.0.1:9/v1",  # 不可达端口
                priority=0,
            )
        ],
        failure_threshold=3,
        recovery_timeout=30.0,
    )

    for i in range(5):
        start = time.perf_counter()
        try:
            await router.chat([{"role": "user", "content": "hi"}], max_tokens=8)
            print(f"第 {i+1} 次：意外成功")
        except Exception as exc:
            cost = (time.perf_counter() - start) * 1000
            state = router.breaker_states()[0]["state"]
            print(f"第 {i+1} 次：失败 {cost:6.0f}ms  state={state}  {type(exc).__name__}")


if __name__ == "__main__":
    asyncio.run(main())
