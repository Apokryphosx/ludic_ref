import pytest

from ludic.context.full_dialog import FullDialog
from ludic.interaction import run_episode
from ludic.agent import Agent
from ludic.inference.client import ChatResponse
from ludic.parsers import (
    cot_prefix_parser,
    xml_move_parser,
    compose_parsers,
)
from tests._mocks import MockEnv, MockClient


# ---------------------------------------------------------------------
# Basic env/agent termination cases
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_terminates_immediately():
    env = MockEnv(max_steps=3, target="1")
    agent = Agent(client=MockClient(text="1"), model="mock")

    rollout = await run_episode(
        env=env,
        agent=agent,
        max_steps=5,
        sampling_args={},
        ctx=FullDialog(),
    )

    assert rollout.steps[-1].terminated is True
    assert rollout.total_reward == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_truncation_when_agent_is_wrong():
    class WrongClient(MockClient):
        async def complete(self, *, model, messages, sampling):
            return ChatResponse(text="nope"), {"used_args": sampling}

    env = MockEnv(max_steps=2, target="1")
    agent = Agent(client=WrongClient(), model="mock")

    rollout = await run_episode(
        env=env,
        agent=agent,
        max_steps=10,
        sampling_args={},
        ctx=FullDialog(),
    )

    assert rollout.steps[-1].truncated is True
    assert rollout.total_reward < 0.0


# ---------------------------------------------------------------------
# Parser integration test
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_episode_uses_action_parser_and_logs_parsed_action():
    """
    Ensure that:
      - run_episode uses the semantic Parser API
      - Step.action keeps the raw LLM text
      - Step.info['parsed_action'] is the parsed action
      - parser reward is added to env reward
      - env.step receives the parsed action
    """

    env = MockEnv(max_steps=3, target="A1")

    # LLM emits a valid CoT-prefixed XML move
    raw_llm_output = "<think>some reasoning</think>\n<move>  A1  </move>"
    agent = Agent(client=MockClient(text=raw_llm_output), model="mock")

    # Compose strict semantic parsers:
    # 1. cot_prefix_parser -> extracts everything after </think>
    # 2. xml_move_parser   -> extracts inner <move>
    action_parser = compose_parsers(
        cot_prefix_parser,
        xml_move_parser,
    )

    rollout = await run_episode(
        env=env,
        agent=agent,
        max_steps=5,
        sampling_args={},
        ctx=FullDialog(),
        action_parser=action_parser,
    )

    assert rollout.length >= 1
    step = rollout.steps[-1]

    # Raw LLM text must be preserved
    assert "<think>" in step.action
    assert "<move>" in step.action

    # Parsed action must be logged
    assert step.info["parsed_action"] == "A1"

    # Env should terminate because parsed action == target "A1"
    assert step.terminated is True

    # Parser reward: 0.0 ; Env reward: 1.0
    assert rollout.total_reward == pytest.approx(1.0)
