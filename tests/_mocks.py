from __future__ import annotations
from typing import Any, Optional, List
from ludic.types import Message
from ludic.agent.base import Agent

class MockAgent(Agent):
    """
    A trivial agent for testing. Always replies with "1".
    """

    async def act(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        return "1"
