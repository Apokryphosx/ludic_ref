from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from ludic.types import Message, SamplingArgs
from ludic.inference.client import ChatClient, ChatResponse

class Agent:
    """
    Concrete agent that defines an interface that can be relied on.
    Defers sampling-arg validation and completion to the underlying client.
    """
    name: str = "agent"

    def __init__(self, *, client: ChatClient, model: str) -> None:
        self._client = client
        self._model = model
        self.last_info: Dict[str, Any] = {}

    async def act(
        self,
        messages: List[Message],
        sampling_args: SamplingArgs,
        timeout_s: Optional[float] = None,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        coro = self._client.complete(
            model=self._model,
            messages=messages,
            sampling_args=sampling_args,
        )
        if timeout_s is None:
            resp, info = await coro
        else:
            resp, info = await asyncio.wait_for(coro, timeout=timeout_s)

        self.last_info = dict(info)
        return resp, info
