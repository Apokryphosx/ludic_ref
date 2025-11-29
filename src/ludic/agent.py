from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Mapping

import torch

from ludic.types import SamplingArgs, Observation, Info
from ludic.inference.client import ChatClient
from ludic.inference.sampling import SamplingConfig, resolve_sampling_args
from ludic.context.base import ContextStrategy
from ludic.context.full_dialog import FullDialog
from ludic.parsers import Parser, ParseResult, xml_move_parser

class Agent:
    """
    A stateful, logical actor that bundles inference, context, and parsing.
    
    It holds a reference to a (potentially shared) ChatClient and manages
    its own internal state via its ContextStrategy.
    """
    name: str = "agent"

    def __init__(
        self, 
        *, 
        client: ChatClient, 
        model: str,
        ctx: ContextStrategy,
        parser: Parser
    ) -> None:
        """
        Initializes the Agent.
        
        Args:
            client: The ChatClient for inference.
            model: The model name this agent should use.
            ctx: An instance of a ContextStrategy for managing memory.
            parser: An instance of a Parser for decoding actions.
        """
        self._client = client
        self._model = model
        self._ctx = ctx
        self._parser = parser
        self.last_info: Dict[str, Any] = {}

    def reset(self, system_prompt: Optional[str] = None) -> None:
        """
        Resets the agent's internal memory (ContextStrategy)
        with an optional system prompt.
        """
        self._ctx.reset(system_prompt=system_prompt)

    async def act(
        self,
        obs: Observation,
        info: Info,
        sampling_args: SamplingArgs,
        timeout_s: Optional[float] = None,
    ) -> Tuple[ParseResult, str, Dict[str, Any]]:
        """
        Runs the full observe -> think -> act -> parse cycle.
        
        This is called by an InteractionProtocol.
        
        Args:
            obs: The observation from the environment.
            info: The info dict from the environment.
            sampling_args: The sampling configuration for this step.
            timeout_s: Optional timeout for the inference call.
            
        Returns:
            A tuple of (ParseResult, raw_action_text, client_info_dict).
        """
        # 1. Observe (update memory with the latest env state)
        self._ctx.on_after_step(obs, info) 
        
        # 2. Think (prepare prompt messages from context)
        messages = self._ctx.on_before_act()
        
        # 3. Act (run inference)
        sampling: SamplingConfig = resolve_sampling_args(sampling_args)
        coro = self._client.complete(
            model=self._model,
            messages=messages,
            sampling=sampling,
        )
        if timeout_s is None:
            resp, client_info = await coro
        else:
            resp, client_info = await asyncio.wait_for(coro, timeout=timeout_s)

        self.last_info = dict(client_info)
        
        # 4. Update memory with the agent's own response
        self._ctx.on_after_act(resp)
        
        # 5. Parse (format the raw text action)
        raw_action = resp.text
        parse_result = self._parser(raw_action)
        
        return parse_result, raw_action, self.last_info

    def push_policy_update(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        reset_cache: bool = True,
        version: Optional[str] = None,
    ) -> str:
        """
        Push updated policy parameters into the underlying runtime.

        Delegates to the ChatClient's push_update_atomic implementation.
        """
        if not hasattr(self._client, "push_update_atomic"):
            raise RuntimeError(
                "Underlying ChatClient does not support policy weight updates "
                "(missing push_update_atomic)."
            )

        return self._client.push_update_atomic(
            params,
            timeout_s=timeout_s,
            reset_cache=reset_cache,
            version=version,
        )