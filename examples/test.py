import os
import time
import asyncio
import torch
from transformers import TrainingArguments, AutoModelForCausalLM

# Ludic Imports
from ludic.agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.inference.vllm_client import VLLMChatClient
from ludic.training.rollout_engine import RolloutEngine, GRPOBatchSource
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest, SamplingArgs
from ludic.training.algorithm import make_reinforce_baseline
from ludic.training.credit_assignment import GroupNormalizedReturn
from ludic.training.config import TrainerConfig
from ludic.training.hf_trainer import LudicTrainer
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.envs.single_agent_env import SingleAgentEnv
from ludic.types import Observation, Info, StepOutcome
from ludic.parsers import ParseResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VLLM_HOST = "127.0.0.1" # Change if server is on another node
VLLM_PORT = 8000

# ---------------------------------------------------------------------------
# 1. Inline Mock Definitions
# ---------------------------------------------------------------------------

def simple_parser(raw: str) -> ParseResult:
    """Pass-through parser."""
    return ParseResult(action=raw, reward=0.0, obs=None)

class MockEnv(SingleAgentEnv):
    """
    A simple environment that asks the agent to output a specific number.
    """
    def __init__(self, target: str = "1", max_steps: int = 3):
        super().__init__()
        self.target = target
        self.max_steps = max_steps
        self._t = 0
        self._obs = f"Please output the number {self.target}."

    @property
    def suggested_sysprompt(self) -> str | None:
        return "You are a helpful assistant. Reply with only the requested number."

    def env_reset(self, *, seed: int | None = None) -> tuple[Observation, Info]:
        self._t = 0
        self._obs = f"Please output the number {self.target}."
        return self._obs, {}

    def env_step(self, action: str) -> StepOutcome:
        self._t += 1
        # Simple reward logic
        if self.target in action:
            reward = 1.0
            terminated = True
            obs = "Correct!"
        else:
            reward = -0.1
            terminated = False
            obs = f"Wrong. Try again ({self._t}/{self.max_steps})"

        truncated = self._t >= self.max_steps
        return StepOutcome(
            obs=obs, 
            reward=reward, 
            truncated=truncated, 
            terminated=terminated, 
            info={}
        )

    def env_current_obs(self) -> Observation:
        return self._obs

# ---------------------------------------------------------------------------
# 2. Main Execution
# ---------------------------------------------------------------------------

def main():
    print(f"🚀 Connecting to vLLM at {VLLM_HOST}:{VLLM_PORT}...")

    # -------------------------------------------------------------------
    # Setup Training Components
    # -------------------------------------------------------------------
    
    print("🧠 Loading Training Model...")
    # Note: device_map="auto" will use available GPUs. 
    # Since you have A100s, bf16 is definitely the way to go.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.train()
    model.gradient_checkpointing_enable()

    # Connect to the external vLLM server
    # (Ensure you launched it with: python -m ludic.inference.vllm_server ...)
    try:
        client = VLLMChatClient(
            host=VLLM_HOST,
            port=VLLM_PORT,
            enable_weight_updates=True,
            connection_timeout_s=10
        )
    except Exception as e:
        print(f"❌ Failed to connect to vLLM: {e}")
        print("   Make sure the server is running!")
        return

    # -- Factories --
    def protocol_factory():
        agent = Agent(
            client=client, 
            model=MODEL_NAME, 
            ctx=FullDialog(), 
            parser=simple_parser
        )
        return SingleAgentSyncProtocol(agent=agent)

    # -- Registries --
    env_registry = {"mock_env": lambda **kwargs: MockEnv(**kwargs)}
    protocol_registry = {"mock_protocol": protocol_factory}
    
    engine = RolloutEngine(
        env_registry=env_registry, 
        protocol_registry=protocol_registry
    )

    # -- Batch Source --
    def requests_fn():
        # Request 4 rollouts (Group Size 4) for our Mock Environment
        return [
            RolloutRequest(
                env=EnvSpec(kind="mock_env", kwargs={"target": "42"}),
                protocol=ProtocolSpec(kind="mock_protocol"),
                sampling_args={
                    "temperature": 1.0, 
                    "max_tokens": 16,
                    "extras": {"extra_body": {"return_token_ids": True}}
                },
                num_episodes=1
            )
        ]

    batch_source = GRPOBatchSource(
        orchestrator=engine,
        credit_assigner=GroupNormalizedReturn(normalize_adv=True),
        requests_fn=requests_fn,
        group_size=4,
        max_steps=5,
        concurrency=4,
        retokenize=False
    )

    # -- Config --
    ludic_config = TrainerConfig(
        pad_token_id=model.config.eos_token_id,
        model_device="cuda", # Let HF handle specific placement via device_map
        sync_every_steps=1
    )

    hf_args = TrainingArguments(
        output_dir="./ludic_test_results",
        max_steps=5, 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=10,
        remove_unused_columns=False,
        bf16=True,
        report_to="none",
        dataloader_pin_memory=False
    )

    trainer = LudicTrainer(
        model=model,
        algo=make_reinforce_baseline(normalize_adv=True),
        batch_source=batch_source,
        client=client,
        ludic_config=ludic_config,
        args=hf_args
    )

    print("🥊 Starting Training Run...")
    trainer.train()
    print("🏆 Success! Training loop completed.")

if __name__ == "__main__":
    main()