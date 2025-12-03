import os
import asyncio
import torch
import wandb
from transformers import TrainingArguments, AutoModelForCausalLM

# Ludic Imports
from ludic.agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.inference.vllm_client import VLLMChatClient
from ludic.training.rollout_engine import RolloutEngine, GRPOBatchSource
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
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
VLLM_HOST = "127.0.0.1" 
VLLM_PORT = 8000

# ---------------------------------------------------------------------------
# 0. WANDB Setup
# ---------------------------------------------------------------------------
# Replace with your actual key or set WANDB_API_KEY environment variable
MY_WANDB_KEY = os.getenv("WANDB_API_KEY", "YOUR_WANDB_API_KEY_HERE")

if MY_WANDB_KEY and MY_WANDB_KEY != "YOUR_WANDB_API_KEY_HERE":
    wandb.login(key=MY_WANDB_KEY)
    os.environ["WANDB_PROJECT"] = "ludic-test-run" # Group your runs here

# ---------------------------------------------------------------------------
# 1. Inline Mock Definitions (Same as before)
# ---------------------------------------------------------------------------

def simple_parser(raw: str) -> ParseResult:
    return ParseResult(action=raw, reward=0.0, obs=None)

class MockEnv(SingleAgentEnv):
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
            obs=obs, reward=reward, truncated=truncated, terminated=terminated, info={}
        )

    def env_current_obs(self) -> Observation:
        return self._obs

# ---------------------------------------------------------------------------
# 2. Main Execution
# ---------------------------------------------------------------------------

def main():
    print(f"🚀 Connecting to vLLM at {VLLM_HOST}:{VLLM_PORT}...")

    # --- Model Setup ---
    print("🧠 Loading Training Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.train()
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable() 

    # --- Client Setup ---
    try:
        client = VLLMChatClient(
            host=VLLM_HOST,
            port=VLLM_PORT,
            enable_weight_updates=True,
            connection_timeout_s=10
        )
    except Exception as e:
        print(f"❌ Failed to connect to vLLM: {e}")
        return

    # --- Components ---
    def protocol_factory():
        agent = Agent(
            client=client, 
            model=MODEL_NAME, 
            ctx=FullDialog(), 
            parser=simple_parser
        )
        return SingleAgentSyncProtocol(agent=agent)

    env_registry = {"mock_env": lambda **kwargs: MockEnv(**kwargs)}
    protocol_registry = {"mock_protocol": protocol_factory}
    
    engine = RolloutEngine(
        env_registry=env_registry, 
        protocol_registry=protocol_registry
    )

    # --- Batch Source ---
    def requests_fn():
        return [
            RolloutRequest(
                env=EnvSpec(kind="mock_env", kwargs={"target": "42"}),
                protocol=ProtocolSpec(kind="mock_protocol"),
                sampling_args={
                    "temperature": 1.0, 
                    "max_tokens": 16,
                    # Important: get token IDs for 'retokenize=False' optimization
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

    # --- Trainer Config ---
    ludic_config = TrainerConfig(
        pad_token_id=model.config.eos_token_id,
        model_device="cuda", 
        sync_every_steps=1
    )

    # HF Trainer Arguments
    # Note: Since we use an IterableDataset (infinite stream), 'epochs' don't apply.
    # We use 'max_steps' to control how long we train.
    hf_args = TrainingArguments(
        output_dir="./ludic_test_results",
        max_steps=50,                  # Run for 50 steps (approx "a few epochs" of updates)
        per_device_train_batch_size=1, # Micro-batch size
        gradient_accumulation_steps=4, # Macro-batch size (accumulate 4 micro-batches)
        learning_rate=1e-5,
        logging_steps=1,               # Log every step so we see data immediately
        save_steps=25,                 # Save checkpoint halfway
        remove_unused_columns=False,
        bf16=True,
        report_to="wandb",             # <--- Enable WandB logging
        dataloader_pin_memory=False,
        run_name="ludic-mock-run-01"
    )

    trainer = LudicTrainer(
        model=model,
        algo=make_reinforce_baseline(normalize_adv=True),
        batch_source=batch_source,
        client=client,                 # <--- Passing client correctly now
        ludic_config=ludic_config,
        args=hf_args
    )

    print("🥊 Starting Training Run...")
    trainer.train()
    print("🏆 Success! Training loop completed.")

if __name__ == "__main__":
    main()