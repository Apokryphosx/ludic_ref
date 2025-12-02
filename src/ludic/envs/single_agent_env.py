from __future__ import annotations
from abc import abstractmethod
from typing import List, Dict, Tuple, Optional

from ludic.envs.env import LudicEnv
from ludic.types import Observation, Info, StepOutcome

class SingleAgentEnv(LudicEnv[str, str, str]):
    """
    A base class for creating simple single-agent environments.

    It handles all the multi-agent dict-based logic internally,
    exposing a simple API for you to implement.
    """
    
    _DEFAULT_ID = "agent_0"

    def __init__(self, agent_id: str = _DEFAULT_ID) -> None:
        """
        Initializes the environment.
        Args:
            agent_id: The role name for the single agent.
                      This is only needed if you want to
                      reference it in a multi-agent protocol.
        """
        self._agent_id = agent_id
        super().__init__()

    # --- This is the API you implement ---

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        """Optionally, provide a default system prompt."""
        return None
    
    @abstractmethod
    def env_reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        """
        Implement your environment's reset logic here.
        Returns:
            A (str, dict) tuple for the single agent.
        """
        ...

    @abstractmethod
    def env_step(self, action: str) -> StepOutcome:
        """
        Implement your environment's step logic here.
        Args:
            action: The string action from the single agent.
        Returns:
            A StepOutcome for the single agent.
        """
        ...
    
    @abstractmethod
    def env_current_obs(self) -> Observation:
        """
        Implement your logic for getting the current observation.
        Returns:
            An Observation string for the single agent.
        """
        ...

    # --- This is the boilerplate we handle for you ---

    @property
    def agent_ids(self) -> List[str]:
        """(Implemented) Returns the single agent ID."""
        return [self._agent_id]

    @property
    def active_agents(self) -> List[str]:
        """(Implemented) The single agent is always active."""
        return [self._agent_id]

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, Tuple[str, Info]]:
        """(Implemented) Wraps your env_reset in a dict."""
        obs, info = self.env_reset(seed=seed)
        return {self._agent_id: (obs, info)}

    def step(self, actions: Dict[str, str]) -> Dict[str, StepOutcome]:
        """(Implemented) Unwraps the action and wraps your outcome."""
        # Get the action for our single agent, default to "" if missing
        action_str = actions.get(self._agent_id, "")
        outcome = self.env_step(action_str)
        return {self._agent_id: outcome}

    def current_obs(self) -> Dict[str, str]:
        """(Implemented) Wraps your env_current_obs in a dict."""
        return {self._agent_id: self.env_current_obs()}