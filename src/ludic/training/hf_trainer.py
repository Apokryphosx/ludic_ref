import asyncio
import torch
from torch.utils.data import IterableDataset
from ludic.training.types import BatchSource
from ludic.training.trainer import _collate_saw_items
from transformers import Trainer
from torch.utils.data import DataLoader
from ludic.training.algorithm import RLAlgorithm
from ludic.training.config import TrainerConfig
from ludic.inference.client import ChatClient


class LudicDataset(IterableDataset):
    def __init__(
        self, 
        batch_source: BatchSource, 
        pad_token_id: int,
    ):
        self.batch_source = batch_source
        self.pad_token_id = pad_token_id
        # We generally keep collation on CPU for DataLoaders to avoid 
        # multiprocessing issues, but with num_workers=0, GPU is fine.
        # Letting HF handle the device placement usually works best.
        self.device = torch.device("cpu")

    def __iter__(self):
        # We rely on the main process loop (asyncio.run) because 
        # DataLoaders are synchronous.
        while True:
            # 1. Fetch the raw SAWBatch (async)
            saw_batch = asyncio.run(self.batch_source.next_batch())
            
            # 2. Collate into PyTorch tensors
            # We explicitly pass the device here.
            batch_tensors = _collate_saw_items(
                saw_batch.items,
                pad_token_id=self.pad_token_id,
                device=self.device
            )
            
            # 3. Yield the dictionary of tensors
            yield batch_tensors



class LudicTrainer(Trainer):
    def __init__(
        self, 
        model, 
        algo: RLAlgorithm, 
        batch_source, 
        ludic_config: TrainerConfig,
        client: Optional[ChatClient] = None,
        **kwargs
    ):
        # Ensure we don't accidentally pass a dataset to the super init
        # which might trigger HF's internal length checks.
        kwargs["train_dataset"] = None
        kwargs["data_collator"] = None
        super().__init__(model=model, **kwargs)
        self.algo = algo
        self.batch_source = batch_source
        self.ludic_config = ludic_config
        self.client = client

        # Safety Check: IterableDatasets require max_steps to be set
        if self.args.max_steps <= 0:
            raise ValueError(
                "When using LudicTrainer (RL), you must set 'max_steps' to a positive integer "
                "in TrainingArguments. Epoch-based training is not supported for infinite rollouts."
            )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader that yields data from the Ludic BatchSource.
        """
        # We ignore self.args.device here and use CPU for the dataset
        dataset = LudicDataset(
            batch_source=self.batch_source,
            pad_token_id=self.ludic_config.pad_token_id,
        )

        return DataLoader(
            dataset,
            # The BatchSource already produces a full batch.
            # We set batch_size=1 to prevent the DataLoader from trying to 
            # batch our batches.
            batch_size=1,
            # This collate_fn removes the extra dimension added by batch_size=1
            collate_fn=lambda x: x[0], 
            # Must be 0 because we run an event loop inside __iter__
            num_workers=0,
            # Pin memory speeds up transfer to GPU in the main loop
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Delegates loss computation to the Ludic RLAlgorithm.
        """
        # 1. Compute loss using your existing infrastructure
        # stats contains things like 'adv_mean', 'entropy', etc.
        loss, stats = self.algo.compute_loss(model, inputs)

        # 2. Log the internal RL stats
        # We check if we should log based on step count to avoid spamming logs
        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            # We prefix metrics with 'rl/' to keep them organized in WandB/Tensorboard
            logs = {f"rl/{k}": v for k, v in stats.items()}
            self.log(logs)

        # 3. Handle return signature required by HF
        if return_outputs:
            return (loss, {"logits": None}) 
        
        return loss