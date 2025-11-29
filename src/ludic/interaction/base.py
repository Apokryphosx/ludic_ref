from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, Dict, Optional

from ludic.env import Env
from ludic.agent import Agent
from ludic.types import Rollout, SamplingArgs

class InteractionProtocol(ABC):
    """
    Abstract base class for all interaction protocols.
    
    A protocol defines the "rules of the game" for how Agent(s) and an
    EnvKernel interact to produce a history of experience (a Rollout).
    """
    
    @abstractmethod
    async def run(
        self,
        *,
        env: Env,
        agents: Union[Agent, Dict[str, Agent]],
        max_steps: int,
        seed: Optional[int] = None,
        sampling_args: Optional[SamplingArgs] = None,
        timeout_s: Optional[float] = None,
    ) -> Rollout:
        """
        Executes one full episode according to the protocol's rules
        and returns the complete Rollout.
        """
        ...