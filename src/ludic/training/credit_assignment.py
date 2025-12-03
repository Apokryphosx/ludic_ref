from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import torch
from ludic.types import Rollout
from ludic.training.types import RolloutStepKey


# ---- Helper for Grouping (FIXED) ----

def _group_rollouts_by_prompt(
    rollouts: List[Rollout],
) -> Dict[str, List[Rollout]]:
    """
    Groups rollouts by their initial observation (prompt).
    This is the 'q' in the GRPO paper and is the correct way
    to group rollouts for group-normalized advantage.
    """
    groups: Dict[str, List[Rollout]] = defaultdict(list)
    for r in rollouts:
        if not r.steps:
            continue  # Skip empty rollouts
        
        # The first observation is the prompt
        prompt_key = r.steps[0].prev_obs
        groups[prompt_key].append(r)
    return groups


# ---- Credit Assigners ----

@dataclass
class GroupNormalizedReturn:
    """
    Computes advantage as (Episodic Return - Group-Mean Episodic Return).

    This is the core advantage estimation for GRPO. It assumes:
    1.  A "group" of rollouts corresponds to multiple outputs for a
        single prompt.
    2.  All rollouts from the same prompt are grouped by their
        identical `rollout.steps[0].prev_obs`.
    3.  The `total_reward` of a rollout is the scalar score from the
        Reward Model.
    4.  All tokens in a single rollout (output) receive the same
        advantage value.
    """
    normalize_adv: bool = False

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        
        out: Dict[RolloutStepKey, float] = {}
        
        # Group by the initial prompt
        groups = _group_rollouts_by_prompt(rollouts)

        for _, group_rollouts in groups.items():
            if not group_rollouts:
                continue

            # 1. Get total reward (RM score) for each rollout in the group
            rewards = torch.tensor(
                [r.total_reward for r in group_rollouts],
                dtype=torch.float32
            )

            # 2. Compute the baseline (group mean)
            baseline = rewards.mean()

            # 3. Compute advantages (A_i = R_i - b)
            advantages = rewards - baseline

            # 4. (Optional) Normalize advantages (zero mean, unit std)
            if self.normalize_adv:
                std = advantages.std(unbiased=False)
                # Add epsilon to prevent divide-by-zero if std is 0
                advantages = advantages / (std + 1e-8) 

            # 5. Assign the computed advantage to *every step*
            #    in the corresponding rollout.
            for i, r in enumerate(group_rollouts):
                adv = advantages[i].item()
                for step in r.steps:
                    key: RolloutStepKey = (r.id, step.index)
                    out[key] = adv
        
        return out


@dataclass
class MonteCarloReturn:
    """
    Monte Carlo return per step:

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    For each rollout and each step, assigns the discounted sum of *future*
    rewards including the current step.

    This is the standard REINFORCE-style return (optionally discounted).
    """

    gamma: float = 1.0

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        out: Dict[RolloutStepKey, float] = {}

        for r in rollouts:
            # process steps in reverse to accumulate returns
            G = 0.0
            returns: List[float] = []

            for step in reversed(r.steps):
                G = float(step.reward) + self.gamma * G
                returns.append(G)

            returns.reverse()  # now aligned with r.steps order

            for step, g in zip(r.steps, returns):
                key: RolloutStepKey = (r.id, step.index)
                out[key] = g

        return out


@dataclass
class PerStepReward:
    """
    Assigns each step's weight equal to its immediate reward:

        w_t = r_t

    This is sometimes useful for simple bandit-style or myopic settings.
    """

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        out: Dict[RolloutStepKey, float] = {}

        for r in rollouts:
            for step in r.steps:
                key: RolloutStepKey = (r.id, step.index)
                out[key] = float(step.reward)

        return out

@dataclass
class EpisodicReturn:
    """
    Assigns the same episodic return to every step in a rollout:

        R_ep = sum_t r_t  (undiscounted total reward for the episode)
        w_t  = R_ep       for all steps t in that rollout

    This is useful when you care only about the overall episode score and
    want each action in a successful episode to receive identical credit.
    """

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        out: Dict[RolloutStepKey, float] = {}

        for r in rollouts:
            R_ep = float(r.total_reward)
            for step in r.steps:
                key: RolloutStepKey = (r.id, step.index)
                out[key] = R_ep

        return out