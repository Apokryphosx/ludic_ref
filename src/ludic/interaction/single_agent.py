from __future__ import annotations
from typing import Optional, Union, Dict

from ludic.env import Env
from ludic.agent import Agent
from ludic.types import Rollout, Step, StepOutcome, SamplingArgs
from .base import InteractionProtocol

class SingleAgentSyncProtocol(InteractionProtocol):
    """
    Implements the standard single-agent, synchronous interaction loop.
    
    This protocol expects a single "heavy" Agent instance that
    manages its own context and parsing.
    """
    
    async def run(
        self,
        *,
        env: Env,
        agents: Union[Agent, Dict[str, Agent]],
        max_steps: int,
        seed: Optional[int] = None,
        sampling_args: Optional[SamplingArgs] = None,
        timeout_s: Optional[float] = None,
        use_env_sysprompt: bool = False,
        system_prompt: Optional[str] = None
    ) -> Rollout:
        
        # 1. --- Setup ---
        if not isinstance(agents, Agent):
            raise ValueError(
                "SingleAgentSyncProtocol requires a single Agent instance, "
                "not a dict. Use a multi-agent protocol instead."
            )
        agent = agents 
        sargs: SamplingArgs = sampling_args or {}

        # 2. --- Reset Agent and Env ---
        
        # Choose system prompt priority: explicit override > env-suggested
        # If both are None, the agent will use its own default.
        if use_env_sysprompt:
            agent.reset(system_prompt=env.suggested_sysprompt)
        elif system_prompt:
            agent.reset(system_prompt=system_prompt)
        else:
            agent.reset()

        # Pass the seed to env.reset()
        obs, info = env.reset(seed=seed)
        
        rollout = Rollout(meta={
            "agent_name": getattr(agent, "name", "unknown"),
            "env_name": env.__class__.__name__,
        })

        # 3. --- Run Interaction Loop ---
        for t in range(max_steps):
            
            # --- A. Call the Agent ---
            # The agent handles its own context and parsing
            parse_result, raw_action, client_info = await agent.act(
                obs=obs,
                info=info,
                sampling_args=sargs,
                timeout_s=timeout_s
            )

            # --- B. Handle Parser Failure ---
            if parse_result.action is None:
                synthetic_obs = parse_result.obs or "Invalid action."
                parser_reward = parse_result.reward

                rollout.steps.append(Step(
                    index=t,
                    prev_obs=obs,
                    action=raw_action,
                    next_obs=synthetic_obs,
                    reward=parser_reward,
                    truncated=False,
                    terminated=False,
                    info={
                        "parse_error": True,
                        "raw_action": raw_action,
                        **client_info
                    },
                ))

                obs = synthetic_obs
                info = {"parse_error": True} # Update info for next agent loop
                continue

            # --- C. Handle Parser Success (Step Env) ---
            parsed_action = parse_result.action
            parser_reward = parse_result.reward

            outcome: StepOutcome = env.step(parsed_action)

            # Build info dict
            step_info = {
                **client_info,
                **outcome.info,
                "parsed_action": parsed_action,
            }

            # Total reward = env reward + parser reward
            total_reward = outcome.reward + parser_reward

            # For logging: terminal/truncated steps have no next_obs
            logged_next_obs = None
            if not (outcome.terminated or outcome.truncated):
                logged_next_obs = outcome.obs

            rollout.steps.append(Step(
                index=t,
                prev_obs=obs,
                action=raw_action,
                next_obs=logged_next_obs,
                reward=total_reward,
                truncated=outcome.truncated,
                terminated=outcome.terminated,
                info=step_info,
            ))

            # Update obs and info for the next agent.act() call
            obs = outcome.obs
            info = outcome.info

            if outcome.terminated or outcome.truncated:
                break

        return rollout