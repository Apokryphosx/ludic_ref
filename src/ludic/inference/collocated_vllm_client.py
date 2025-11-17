from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch  # type: ignore
from vllm import SamplingParams  # type: ignore
from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from ludic.types import Message
from ludic.inference.client import ChatClient, ChatResponse
from ludic.inference.sampling import SamplingConfig

log = logging.getLogger(__name__)


class CollocatedVLLMChatClient(ChatClient):
    """
    vLLM ChatClient backed by an *in-process* AsyncLLMEngine.

    This is the "collocated" / TRL-style setup:
      - Training loop owns a HF model on the same GPU(s).
      - vLLM runs in the same Python process on the same device(s).
      - The RL/environment side only sees a ChatClient.

    Responsibilities:
      * map (messages, SamplingConfig) -> vLLM SamplingParams
      * run generation via AsyncLLMEngine
      * return a normalized ChatResponse + info dict
      * optionally mirror trainer weights into the vLLM engine
        via push_update_atomic().

    Assumptions:
      * `engine` is already constructed (e.g. AsyncLLMEngine.from_vllm_config)
        with whatever logits processors, KV cache config etc. you want.
      * `tokenizer` matches the model inside vLLM and uses a chat template
        compatible with how you trained / want to prompt the model.
    """

    def __init__(
        self,
        *,
        engine: AsyncLLMEngine,
        tokenizer: PreTrainedTokenizerBase,
        chat_template: Optional[str] = None,
        enable_weight_updates: bool = False,
    ) -> None:
        self._engine = engine
        self._tokenizer = tokenizer
        self.enable_weight_updates = enable_weight_updates

        # If you want a non-default chat template, you can pass it in; otherwise
        # we rely on tokenizer.chat_template.
        if chat_template is not None:
            # This is how HF tokenizers usually allow overriding templates.
            # If your tokenizer does it differently, adjust accordingly.
            self._tokenizer.chat_template = chat_template  # type: ignore[attr-defined]

    # ----------------------------------------------------------------------
    # internal helpers
    # ----------------------------------------------------------------------

    def _build_sampling_params(
        self,
        sampling: SamplingConfig,
        *,
        interrupt_thinking: Optional[int],
        return_token_ids: bool,
    ) -> SamplingParams:
        """
        Map our SamplingConfig -> vLLM SamplingParams.

        This is analogous to .to_openai_kwargs() in the HTTP client, but
        targeting vLLM's native SamplingParams instead of OpenAI kwargs.
        """

        # Note: vLLM ignores some of these; we still thread them through
        # to make "policy" only live in SamplingConfig.
        params = SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            presence_penalty=sampling.presence_penalty,
            frequency_penalty=sampling.frequency_penalty,
            stop=sampling.stop or None,
            # If you want logprobs for training, you can set logprobs>0 here.
            # For now we use return_token_ids as a hint that caller cares.
            logprobs=1 if return_token_ids else None,
        )

        # vLLM V1 side-channel arguments for logits processors, etc.
        extra_args: Dict[str, Any] = dict(sampling.extras or {})

        # Our custom GlobalThinkProcessor uses "max_think"
        if interrupt_thinking is not None:
            if not isinstance(interrupt_thinking, int) or interrupt_thinking <= 0:
                raise ValueError("interrupt_thinking must be a positive integer")
            extra_args["max_think"] = interrupt_thinking

        if extra_args:
            # new vLLM versions hang this off SamplingParams.extra_args
            setattr(params, "extra_args", extra_args)

        return params

    def _build_prompt_and_ids(
        self,
        messages: List[Message],
    ) -> Tuple[str, List[int]]:
        """
        Convert a list of OpenAI-style chat messages into:
          - a single string prompt (for debugging/logging)
          - the corresponding token IDs.

        We always go through the tokenizer's chat_template, so:
          - what vLLM sees
          - what we store as prompt_token_ids
        are guaranteed to be aligned (no post-hoc retokenization drift).
        """
        # `messages` is a List[Dict[str, str]] with keys "role"/"content"
        # HF's apply_chat_template expects a slightly different structure
        # (list of {"role": ..., "content": ...}), which matches ours.
        # If your tokenizer wants a different schema, adapt here.

        # Text form
        prompt_text: str = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Token IDs form (exactly what we feed into vLLM)
        prompt_ids: List[int] = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        return prompt_text, prompt_ids

    # ----------------------------------------------------------------------
    # ChatClient.complete
    # ----------------------------------------------------------------------

    async def complete(
        self,
        *,
        model: str,  # ignored; model is baked into AsyncLLMEngine
        messages: List[Message],
        sampling: SamplingConfig,
        interrupt_thinking: Optional[int] = None,
        return_token_ids: bool = False,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        """
        High-level LLM invocation using the *collocated* vLLM engine.

        Args:
            model:
                Logical model name; vLLM side already knows its weights.
                We keep this parameter for compatibility but don't use it.

            messages:
                List of OpenAI-style chat messages.

            sampling:
                Fully-resolved SamplingConfig from ludic.

            interrupt_thinking:
                Same semantics as the HTTP client:
                  -> SamplingParams.extra_args["max_think"]

            return_token_ids:
                If True, we try to populate:
                  - ChatResponse.token_ids
                  - ChatResponse.prompt_token_ids

        Returns:
            (ChatResponse, info)
        """
        # 1) Build prompt text + token IDs via chat template
        prompt_text, prompt_token_ids = self._build_prompt_and_ids(messages)

        # 2) SamplingConfig -> SamplingParams
        sampling_params = self._build_sampling_params(
            sampling,
            interrupt_thinking=interrupt_thinking,
            return_token_ids=return_token_ids,
        )

        # 3) Dispatch to AsyncLLMEngine
        #
        # The exact API surface of AsyncLLMEngine differs between versions.
        # Conceptually, we want:
        #   - to pass `prompt_token_ids`
        #   - to pass `sampling_params`
        #   - to await the first RequestOutput
        #
        # Pseudocode for vLLM >=0.5.x:
        #
        #   outputs = await self._engine.generate(
        #       prompt_token_ids=[prompt_token_ids],
        #       sampling_params=sampling_params,
        #       request_id=f"ludic-{time.time_ns()}",
        #   )
        #
        # where `outputs` is a list[RequestOutput].
        #
        # Here we write it in that spirit; adapt to your exact engine version.
        request_id = f"ludic-{time.time_ns()}"

        outputs = await self._engine.generate(  # type: ignore[attr-defined]
            prompt_token_ids=[prompt_token_ids],
            sampling_params=sampling_params,
            request_id=request_id,
        )

        if not outputs:
            raise RuntimeError("vLLM returned no RequestOutput for collocated call")

        # We asked for a single prompt → single RequestOutput
        out = outputs[0]
        if not out.outputs:
            raise RuntimeError("vLLM RequestOutput has no candidate outputs")

        # First candidate
        candidate = out.outputs[0]

        # Text + token IDs + logprobs
        text: str = getattr(candidate, "text", "") or ""
        completion_ids: Optional[List[int]] = getattr(candidate, "token_ids", None)
        logprobs: Optional[List[float]] = None

        # vLLM usually exposes per-token logprobs as a list of TokenLogprobs
        # objects; we flatten to a float list if present.
        token_logprobs = getattr(candidate, "logprobs", None)
        if token_logprobs is not None:
            # e.g. candidate.logprobs is List[TokenLogprobs], each has .logprob
            try:
                logprobs = [t.logprob for t in token_logprobs]  # type: ignore[attr-defined]
            except Exception:
                # Be resilient across vLLM versions
                log.warning("Could not flatten vLLM logprobs; leaving as None")

        # Finish reason, if vLLM exposes it (varies by version)
        finish_reason: Optional[str] = getattr(candidate, "finish_reason", None)
        if finish_reason is None:
            # Some versions use a status enum; you can map it if you care.
            finish_reason = getattr(out, "finished_reason", None)

        chat_resp = ChatResponse(
            text=text,
            token_ids=completion_ids,
            logprobs=logprobs,
            finish_reason=finish_reason,
            # These are *exactly* the IDs we gave vLLM, so no alignment issues.
            prompt_token_ids=prompt_token_ids if return_token_ids else None,
        )

        info: Dict[str, Any] = {
            "used_args": {
                "request_id": request_id,
                "sampling_params": sampling_params,
                # We include prompt text mainly for debugging; you may want to
                # drop this in production if prompts are large.
                "prompt_preview": prompt_text[:256],
            },
            # Raw RequestOutput is not trivially JSON-serializable; you might
            # want to store a stripped-down version.
            "raw_output": out,
        }

        return chat_resp, info

    # ----------------------------------------------------------------------
    # ChatClient.push_update_atomic
    # ----------------------------------------------------------------------

    def push_update_atomic(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        reset_cache: bool = True,
        version: Optional[str] = None,
    ) -> str:
        """
        Push updated model parameters into the collocated vLLM engine.

        For the collocated setup, there is no NCCL / HTTP dance:
          - `params` is typically a (possibly partial) state dict coming
            from your training model (e.g. HF Transformer or LoRA adapter).
          - You translate those into the naming / sharding scheme vLLM expects.
          - Then you call into the underlying engine/model to load them.

        This is exactly the place to mirror what TRL does:

            * You may only push a subset of weights (e.g. LoRA).
            * You may need a name mapping: trainer_name -> vllm_name.
            * You may want to coerce dtype (bf16 in trainer, fp16 in vLLM).

        For now this is a skeleton that assumes you have a method
        `self._update_single_param(name: str, tensor: torch.Tensor)`.
        """

        if not self.enable_weight_updates:
            raise RuntimeError(
                "push_update_atomic() called but enable_weight_updates=False. "
                "Either enable it at construction time or avoid calling this."
            )

        start = time.time()

        # Example: naive loop over params. In a real implementation you’ll:
        #   - translate names
        #   - maybe slice tensors for tensor parallelism
        #   - ensure correct dtype/device
        for name, tensor in params.items():
            if (time.time() - start) > timeout_s:
                raise TimeoutError(f"push_update_atomic exceeded {timeout_s}s")

            self._update_single_param(name, tensor)

        if reset_cache:
            self._reset_prefix_cache()

        return version or f"collocated-vllm-{int(time.time())}"

    # ---- Internal weight-update helpers ---------------------------------

    def _update_single_param(self, name: str, tensor: torch.Tensor) -> None:
        """
        Skeleton hook where you actually write into vLLM's weights.

        In spirit, you want to mirror what the WeightSyncWorkerExtension does
        on the server side:

            self.model_runner.model.load_weights(weights=[(name, weight)])

        For the collocated case you'll need to reach into AsyncLLMEngine's
        internal model runner. The exact attribute path depends on the vLLM
        version; something like:

            runner = self._engine.engine.model_executor.model_runner
            runner.model.load_weights(weights=[(vllm_name, tensor_on_correct_device)])

        This is intentionally left as a TODO because it is tightly coupled
        to the vLLM internals / your model config.
        """
        raise NotImplementedError(
            "CollocatedVLLMChatClient._update_single_param must be implemented "
            "to write tensors into the vLLM engine's model weights."
        )

    def _reset_prefix_cache(self) -> None:
        """
        Reset KV/prefix caches after a weight update, so future generations
        see fresh weights.

        AsyncLLMEngine exposes a reset_prefix_cache() coroutine; in a pure
        sync context you may need to run it via asyncio.run() or schedule it
        on your loop.
        """
        try:
            # If you're calling push_update_atomic from a non-async context,
            # you likely have no loop; in that case you might want a more
            # elaborate integration. For now, keep it simple.
            import asyncio

            async def _reset():
                await self._engine.reset_prefix_cache()  # type: ignore[attr-defined]

            asyncio.run(_reset())
        except Exception as exc:
            log.warning("Failed to reset vLLM prefix cache after weight update: %s", exc)
