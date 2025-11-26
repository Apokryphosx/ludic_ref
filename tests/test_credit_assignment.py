import pytest

from ludic.types import Rollout, Step
from ludic.training.credit_assignment import (
    MonteCarloReturn,
    EpisodicReturn,
    PerStepReward,
)

# ---- Helper to build a simple rollout ----

def _make_rollout(id: str, rewards: list[float]) -> Rollout:
    """Creates a simple rollout with the given reward sequence."""
    rollout = Rollout(id=id)
    obs = "start"
    for i, reward in enumerate(rewards):
        next_obs = f"obs_{i+1}" if i < len(rewards) - 1 else None
        rollout.steps.append(Step(
            index=i,
            prev_obs=obs,
            action=f"action_{i}",
            next_obs=next_obs,
            reward=reward,
            truncated=False,
            terminated=(i == len(rewards) - 1),
        ))
        obs = next_obs or ""
    return rollout


# ---- Tests ----

def test_monte_carlo_return_gamma_1():
    """Test standard (gamma=1.0) return-to-go."""
    r1 = _make_rollout("r1", rewards=[0.0, 0.0, 1.0]) # len 3
    r2 = _make_rollout("r2", rewards=[-1.0])          # len 1

    assigner = MonteCarloReturn(gamma=1.0)
    weights = assigner.compute([r1, r2])

    # G_t = r_t + r_{t+1} + ...
    # r1: [0.0, 0.0, 1.0]
    # G:  [1.0, 1.0, 1.0]
    assert weights[("r1", 0)] == pytest.approx(1.0)
    assert weights[("r1", 1)] == pytest.approx(1.0)
    assert weights[("r1", 2)] == pytest.approx(1.0)

    # r2: [-1.0]
    # G:  [-1.0]
    assert weights[("r2", 0)] == pytest.approx(-1.0)

    assert len(weights) == 4


def test_monte_carlo_return_gamma_0_9():
    """Test discounted (gamma=0.9) return-to-go."""
    r1 = _make_rollout("r1", rewards=[1.0, 2.0, 4.0]) # len 3

    assigner = MonteCarloReturn(gamma=0.9)
    weights = assigner.compute([r1])

    # r1: [1.0, 2.0, 4.0]
    # G_2 = 4.0
    # G_1 = 2.0 + 0.9 * G_2 = 2.0 + 0.9 * 4.0 = 2.0 + 3.6 = 5.6
    # G_0 = 1.0 + 0.9 * G_1 = 1.0 + 0.9 * 5.6 = 1.0 + 5.04 = 6.04
    assert weights[("r1", 0)] == pytest.approx(6.04)
    assert weights[("r1", 1)] == pytest.approx(5.6)
    assert weights[("r1", 2)] == pytest.approx(4.0)
    assert len(weights) == 3


def test_episodic_return():
    """Test that all steps in a rollout get the total sum."""
    r1 = _make_rollout("r1", rewards=[0.0, 0.0, 1.0]) # total = 1.0
    r2 = _make_rollout("r2", rewards=[-1.0, -0.5])    # total = -1.5

    assigner = EpisodicReturn()
    weights = assigner.compute([r1, r2])

    assert weights[("r1", 0)] == pytest.approx(1.0)
    assert weights[("r1", 1)] == pytest.approx(1.0)
    assert weights[("r1", 2)] == pytest.approx(1.0)

    assert weights[("r2", 0)] == pytest.approx(-1.5)
    assert weights[("r2", 1)] == pytest.approx(-1.5)
    assert len(weights) == 5


def test_per_step_reward():
    """Test that all steps get their own immediate reward."""
    r1 = _make_rollout("r1", rewards=[1.0, 2.0, 4.0])

    assigner = PerStepReward()
    weights = assigner.compute([r1])

    assert weights[("r1", 0)] == pytest.approx(1.0)
    assert weights[("r1", 1)] == pytest.approx(2.0)
    assert weights[("r1", 2)] == pytest.approx(4.0)
    assert len(weights) == 3